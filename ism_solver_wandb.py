# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

import wandb # Added wandb import (if not already imported globally)

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *
from opt_functions.Solver_functions.projected_gradient import *
from opt_functions.Solver_functions.regularizations import *
from opt_functions.Solver_functions.white_opt_princ import *
from opt_functions.Solver_functions.Kulback_libler import *

from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares
import time
from matplotlib import cm


torch.manual_seed(0)
#%%
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv= TVLoss()

hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
    'lam': 0
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'

dataset = prepare_ism_data(
    is_real = hparams['IS_REAL'],
    real_name= hparams['real_name'],
    load_path = None if not hparams['LOAD_FROM_FILE'] else hparams['path'],
    phantom_type= hparams['real_name'],
    Nx = 256, Ny = 256, 
    Nz = hparams['Nz'] , 
    pxsize = hparams['pxsize'], 
    flux = hparams['flux'],
    device = device,
    show_plots = True
)

kl = KL(back=dataset["back_vec"])
tv=TVLoss()
l1 = l1Loss()

# Definiamo le logiche per ogni algoritmo
CONFIG_REG = {
    "pgd": {
        "prior": (tv.forward_3D, tv.forward),
        "prior_grad": (tv.grad_3D, tv.grad),
        "prox": (None, None) 
    },
    "prox": {
        "prior": (l1.forward_3D, l1.forward),
        "prior_grad": (None, None),
        "prox": (tresholding_3D, tresholding)
    },
    "md": {
        "prior": (tv.forward_3D, tv.forward),
        "prior_grad": (tv.grad_3D, tv.grad),
        "prox": (None, None)
    },
    "rl": {
        "prior": (None, None),
        "prior_grad": (None, None),
        "prox": (None, None)
    }
}

idx = 0 if hparams['IS_3D'] else 1

if hparams['IS_3D']:
    list_methods_tub = [("md", 0.0604)]
else:
    list_methods_tub = [ ("md", 0.0805)]
    
if hparams['IS_3D'] and hparams['real_name'] == 'tubulin':
    list_methods_tub = [("pgd", 0.0134), ("prox", 0.0268), ("md", 0.309)]
elif not hparams['IS_3D'] and hparams['real_name'] == 'tubulin':
    list_methods_tub = [("pgd", 0.00672), ("prox", 0.0805), ("md", 0.0505)]
    
if hparams['IS_REAL'] and hparams['real_name'] == '01_tomm20':
    list_methods_tub = [("pgd", 0.0111), ("prox", 0.0243), ("md", 0.00304)]
elif hparams['IS_REAL'] and hparams['real_name'] == '02_tomm20':
    list_methods_tub = [("pgd", 0.0354), ("prox", 0.0556), ("md", 0.0485)]
elif hparams['IS_REAL'] and hparams['real_name'] == '03_tomm20':
    list_methods_tub = [("pgd", 0.00708), ("prox", 0.0394), ("md", 0.0485)]
elif hparams['IS_REAL'] and hparams['real_name'] == '04_tomm20':
    list_methods_tub = [("pgd", 0.00708), ("prox", 0.0253), ("md", 0.00102)]
    

def wandb_callback(iteration, x_curr, metrics):
    """
    Questa funzione verrà chiamata dal solver ogni 'log_interval' iterazioni.
    """
    # 1. Prepariamo il dizionario base con le metriche
    log_dict = metrics.copy()
    
    # 2. Estraiamo l'immagine per la visualizzazione
    # Gestiamo la differenza tra 2D e 3D: prendiamo sempre la slice centrale se è 3D
    with torch.no_grad():
        img_np = x_curr.cpu().numpy().squeeze()
        caption = "Reconstruction 2D"

    # 3. Creiamo la figura con matplotlib e colorbar
    fig, ax = plt.subplots(figsize=(5, 5))
    cax = ax.imshow(img_np, cmap='hot')
    fig.colorbar(cax, ax=ax)
    ax.axis('off')
    
    # Aggiungiamo l'immagine a wandb
    log_dict["Current_Reconstruction"] = wandb.Image(fig, caption=caption)
    
    # 4. Spediamo tutto a wandb usando lo step corretto
    wandb.log(log_dict, step=iteration)
    
    # Puliamo la memoria
    plt.close(fig)
    
    


ALGORITHM = 'pgd' 
cfg = CONFIG_REG[ALGORITHM]
lam = 0.01
hparams["lam"] = lam

parameters = {
    "max_iter": 10000,
    "tollerance": 1e-5,
    "Lip_reg": dataset["L_th"], 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    "lam": hparams['lam'],
    
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,
    
    "prior": cfg["prior"][idx],
    "prox": cfg["prox"][idx],
    "prior_grad": cfg["prior_grad"][idx],
    "callback": wandb_callback,   # Passiamo la funzione
    "log_interval": 10
}



# ==========================================
# 1. WANDB INITIALIZATION
# ==========================================
run_name = f"{hparams['real_name']}_{ALGORITHM}_lam_{lam}"

wandb.init(
    entity="ism_regularisation",
    project="my-awesome-project",
    reinit=True, # Allows multiple runs in the same script
    config={
        "algorithm": ALGORITHM,
        "lambda": lam,
        "max_iter": parameters["max_iter"],
        "tollerance": parameters["tollerance"],
        **hparams # Logs all your hparams automatically
    }
)
noise_image = dataset["noise_image"].float().cpu().squeeze()
# 3. Logghiamo tutto
wandb.log({"Observation" : wandb.Image(cm.hot((noise_image.sum(0)/torch.max(noise_image.sum(0))).cpu().squeeze(1)), caption="Observation")})

# If Ground Truth exists, log it too
wandb.log({"Ground Truth": wandb.Image(dataset["ground_truth"].squeeze(1), caption="Ground Truth")})

# Crea gli eventi per la sincronizzazione
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

SolverClass = Pgd_Backtracking

solver = SolverClass(parameters, 
                        algorithm = ALGORITHM, 
                        is_3d=hparams['IS_3D'], 
                        is_realdata = hparams['IS_REAL'], 
                        cfg_prior = "l1")

start_time = time.perf_counter() 

results = solver.solve(y=dataset["noise_image"])

end_time = time.perf_counter() 

execution_time = end_time - start_time
print(f"Tempo di esecuzione per {hparams['real_name']} con {ALGORITHM}: {execution_time:.4f} secondi")

reconstructed_img = results['x_result']



## SAVING DATA
real_tag = "real" if hparams['IS_REAL'] else "sim"
sect_tag = "3D" if hparams['IS_3D'] else "2D"

save_path = f"Results/ism_results/ism_{sect_tag}_{real_tag}_{ALGORITHM}_{hparams['real_name']}_lam{parameters['lam']}.pth"

clean_dataset = {
    "noise_image": dataset["noise_image"].cpu() if isinstance(dataset["noise_image"], torch.Tensor) else dataset["noise_image"],
    "ground_truth": dataset["ground_truth"].cpu() if isinstance(dataset["ground_truth"], torch.Tensor) else dataset["ground_truth"],
    "clean_image":dataset["clean_image"].cpu() if isinstance(dataset["clean_image"], torch.Tensor) else dataset["clean_image"],
    'meta': dataset["meta"].cpu() if isinstance(dataset["meta"], torch.Tensor) else dataset["meta"],
}

torch.save({
    'hparams': hparams,
    'results': results,
    'dataset': clean_dataset
}, save_path)

full_path = os.path.abspath(save_path)
print(f"Sto salvando in: {full_path}")

wandb.finish()

