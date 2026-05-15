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
    
    
# ==========================================
# 1. CONFIGURAZIONE DELLO SWEEP
# ==========================================
sweep_config = {
    'method': 'grid', # Usa 'grid' per testare tutti i valori, 'bayes' per cercare in modo intelligente
    'metric': {
      'name': 'PSNR', # Metrica da ottimizzare (deve coincidere con il nome loggato dal solver)
      'goal': 'maximize'   
    },
    'parameters': {
        'lam': {
            # Inserisci qui tutti i valori di lambda che vuoi testare
            'values': torch.linspace(1e-8, 1, steps=150).tolist()
        }
    }
}

# Creiamo lo sweep sul server
sweep_id = wandb.sweep(sweep_config, entity="ism_regularisation", project="my-awesome-project")


# ==========================================
# 2. FUNZIONE DI ESECUZIONE (WRAPPER)
# ==========================================
def run_experiment():
    # Il blocco 'with' chiude automaticamente il run in modo sicuro alla fine
    with wandb.init() as run:
        
        # 1. RECUPERA IL LAMBDA DINAMICO
        lam = wandb.config.lam 
        ALGORITHM = 'prox' 
        cfg = CONFIG_REG[ALGORITHM]
        
        # Aggiorniamo hparams localmente per il salvataggio
        current_hparams = hparams.copy()
        current_hparams["lam"] = lam

        parameters = {
            "max_iter": 10000,
            "tollerance": 1e-5,
            "Lip_reg": dataset["L_th"], 
            "x_init": dataset["x_init"],
            "physics": dataset["physics"],
            "ground_truth": dataset["ground_truth"],
            "back": dataset["back_vec"],
            "lam": lam, # Parametro dinamico!
            
            "data_fid": kl.forward_25_3D if current_hparams['IS_3D'] else kl.forward_25,
            "grad_data_fid": kl.grad_25_3D if current_hparams['IS_3D'] else kl.grad_25,
            "single_data_fid": KL_metric if current_hparams['IS_3D'] else KL_metric,
            
            "prior": cfg["prior"][idx],
            "prox": cfg["prox"][idx],
            "prior_grad": cfg["prior_grad"][idx],
            "callback": wandb_callback,
            "log_interval": 100 # Ti consiglio 100 o 500 per gli sweep, per non sovraccaricare W&B
        }

        # Log delle informazioni aggiuntive sulla dashboard
        wandb.config.update({
            "algorithm": ALGORITHM,
            "max_iter": parameters["max_iter"],
            "tollerance": parameters["tollerance"],
        })

        # 2. LOG DELLE IMMAGINI BASE
        # Nota: Ho aggiunto .numpy() esplicito per evitare problemi con matplotlib cm.hot
        noise_image_np = dataset["noise_image"].float().cpu().squeeze().numpy()
        obs_img = noise_image_np.sum(0) / noise_image_np.sum(0).max()
        wandb.log({"Observation" : wandb.Image(cm.hot(obs_img), caption="Observation")})

        if dataset.get("ground_truth") is not None:
            gt_np = dataset["ground_truth"].float().cpu().squeeze().numpy()
            wandb.log({"Ground Truth": wandb.Image(gt_np, caption="Ground Truth")})

        # 3. AVVIO DEL SOLVER
        # Crea gli eventi per la sincronizzazione GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        SolverClass = Pgd_Backtracking

        solver = SolverClass(parameters, 
                             algorithm = ALGORITHM, 
                             is_3d=current_hparams['IS_3D'], 
                             is_realdata = current_hparams['IS_REAL'], 
                             cfg_prior = "l1")

        print(f"\n=== Avvio run con lam = {lam} ===")
        start_time = time.perf_counter() 
        results = solver.solve(y=dataset["noise_image"])
        execution_time = time.perf_counter() - start_time
        
        print(f"Tempo di esecuzione per {current_hparams['real_name']} (lam={lam}): {execution_time:.4f} secondi")
        wandb.log({"execution_time_seconds": execution_time})

        # 4. SALVATAGGIO DEI DATI
        real_tag = "real" if current_hparams['IS_REAL'] else "sim"
        sect_tag = "3D" if current_hparams['IS_3D'] else "2D"
        save_path = f"Results/ism_results/ism_{sect_tag}_{real_tag}_{ALGORITHM}_{current_hparams['real_name']}_lam{lam}.pth"

        clean_dataset = {
            "noise_image": dataset["noise_image"].cpu() if isinstance(dataset["noise_image"], torch.Tensor) else dataset["noise_image"],
            "ground_truth": dataset["ground_truth"].cpu() if isinstance(dataset["ground_truth"], torch.Tensor) else dataset["ground_truth"],
            "clean_image": dataset["clean_image"].cpu() if isinstance(dataset["clean_image"], torch.Tensor) else dataset["clean_image"],
            'meta': dataset["meta"].cpu() if isinstance(dataset["meta"], torch.Tensor) else dataset["meta"],
        }

        torch.save({
            'hparams': current_hparams,
            'results': results,
            'dataset': clean_dataset
        }, save_path)
        
        print(f"Salvato con successo in: {os.path.abspath(save_path)}")


# ==========================================
# 3. ESECUZIONE DELL'AGENTE
# ==========================================
# Questo comando avvierà la funzione `run_experiment` per ogni valore di lambda
wandb.agent(sweep_id, function=run_experiment)
