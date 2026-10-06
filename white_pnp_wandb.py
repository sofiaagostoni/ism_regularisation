# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import numpy as np
import deepinv as dinv
import wandb
from tqdm.auto import tqdm
import time
from matplotlib import pyplot as plt
from matplotlib import cm

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *
from opt_functions.Solver_functions.projected_gradient import *
from opt_functions.Solver_functions.regularizations import *
from opt_functions.Solver_functions.white_opt_pnp import *
from opt_functions.Solver_functions.Kulback_libler import *
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

torch.manual_seed(0)
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. SETUP E CARICAMENTO DATI GLOBALI
# ==========================================
# Li carichiamo una volta sola per non sprecare tempo durante lo sweep

hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
}

hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'

print("Caricamento dataset...")
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
    show_plots = False,
    normalization_y = False # Manteniamo la normalizzazione manuale
)

noise_image = dataset["noise_image"]
finger_print = dataset["fingerprint"]
physics = dataset["physics"]

# Normalizzazione manuale allineata al tuo codice
x_0 = dataset['x_init']
x_0 = (x_0 / x_0.max())

for i in range(25):
    max_y_i = torch.max(noise_image[i])
    noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]

dataset['noise_image'] = noise_image
dataset['x_init'] = x_0
dataset['back_vec'] = dataset["back_vec"].to(device)

# Caricamento Modello PnP (DRUNet) e metriche
print("Caricamento DRUNet...")
kl = KL(back=dataset["back_vec"])
drunet = dinv.models.DRUNet(in_channels=1, out_channels=1, pretrained="download", device=device)

base_parameters = {
    "tollerance": 1e-12, # Tolleranza stringente
    "Lip_reg": dataset["L_th"]*1e-2, 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,
    "Pnp" : drunet,
}


# ==========================================
# 2. DEFINIZIONE DEL TRAINING LOOP PER LO SWEEP
# ==========================================
def train_sweep():
    # wandb.init() senza argomenti dentro lo sweep: prende i dati dall'agent
    with wandb.init() as run:
        config = wandb.config
        
        # Prepariamo i parametri specifici per questo run
        current_parameters = base_parameters.copy()
        current_parameters["sigma"] = config.sigma
        current_parameters["max_iter"] = config.max_iter
        
        print(f"\n[Run Sweep] -> sigma: {config.sigma:.5e} | max_iter: {config.max_iter}")
        start_time = time.perf_counter()
        
        # Esecuzione Algoritmo PnP
        x_result, funct, iter_err, norm2_drunet, psnr_vec, ssim_vec = pnp_ism(
            dataset["noise_image"], 
            dataset['back_vec'], 
            current_parameters, 
            device
        )
        
        execution_time = time.perf_counter() - start_time
        
        # Calcolo Whiteness Principle (Residual Whiteness)
        lambda_d = physics(x_result) + dataset["back_vec"].view(-1, 1, 1, 1)
        if hparams['IS_3D']:
            lambda_d = lambda_d.sum(1).unsqueeze(1)
            
        Z = standardize_unbiased_masked(dataset["noise_image"], lambda_d)
        wh, M_eff = whiteness_measure(Z, mode="highpass", cutoff_ratio=0.10)
        
        wp_value = (M_eff * wh).item()
        iter_raggiunte = len(iter_err) if len(iter_err) > 0 else config.max_iter
        
        # Preparazione log WANDB
        log_dict = {
            "WP_Final": wp_value,
            "Execution_Time_sec": execution_time,
            "Iterations_Converged": iter_raggiunte
        }
        
        if not hparams['IS_REAL']:
            log_dict["PSNR_Final"] = psnr_vec[-1].item()
            log_dict["SSIM_Final"] = ssim_vec[-1].item()
        
        # Creazione Immagine Ricostruita per la dashboard
        with torch.no_grad():
            img_np = x_result.cpu().numpy().squeeze()
            fig, ax = plt.subplots(figsize=(5, 5))
            cax = ax.imshow(img_np, cmap='hot')
            fig.colorbar(cax, ax=ax)
            ax.axis('off')
            
            caption = f"Sig={config.sigma:.1e}, It={config.max_iter}, WP={wp_value:.2e}"
            log_dict["Reconstruction"] = wandb.Image(fig, caption=caption)
            
        wandb.log(log_dict)
        plt.close(fig)


# ==========================================
# 3. CONFIGURAZIONE E AVVIO DELLO SWEEP
# ==========================================
sweep_config = {
    'method': 'grid',  # Testiamo tutte le combinazioni incrociate
    'metric': {
        'name': 'WP_Final',
        'goal': 'minimize'   
    },
    'parameters': {
        'max_iter': {
            'values': [200, 400, 600, 800, 1000]
        },
        'sigma': {
            # Scala logaritmica da 1e-4 a ~1e-1
            'values': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        }
    }
}

if __name__ == "__main__":
    print("Inizializzazione Sweep su Weights & Biases...")
    
    # Inizializziamo lo sweep passando le tue credenziali e il nome del progetto
    sweep_id = wandb.sweep(
        sweep_config, 
        entity="ism_regularisation", 
        project="my-awesome-project"
    )
    
    # Avvia l'agente che eseguirà `train_sweep` per tutte le combinazioni della grid
    wandb.agent(sweep_id, function=train_sweep)