#%%
import os
import gc
import itertools
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# I tuoi import
from opt_functions import *
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from opt_functions.Data_manager.generate_measurments import *

# Setup globale
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1, steps=150)],
    dim=0
    ).to(device)



MASK = 'masked'

def run_experiment(real_name, nz, algorithm, MASK):
    """
    Esegue un singolo esperimento e salva i risultati.
    """
    # ==========================================
    # 1. CONTROLLO PREVENTIVO ESISTENZA FILE
    # ==========================================
    
    is_3d = (nz > 1)
    opt_sec = '3D' if is_3d else '2D'
    
    real_name = real_name

    
    os.makedirs("Results/WP", exist_ok=True) # Assicura che la cartella esista
    save_path = f"Results/WP/wp_newgrid_{opt_sec}_{algorithm}_{MASK}_{real_name}.pth"

    # if os.path.exists(save_path):
    #     print(f"\n[SKIP] L'esperimento {real_name} | Nz={nz} | Algo={algorithm} è già completato. File: {save_path}")
    #     return # Esce immediatamente dalla funzione e passa al prossimo

    # ==========================================
    # 2. SE IL FILE NON ESISTE, AVVIA I CALCOLI
    # ==========================================
    print(f"\n--- Avvio exp: Dataset={real_name} | Nz={nz} | Algo={algorithm} ---")
    
    ## HYPER PARAM SETTING
    hparams = {
        'Nz': nz,
        'pxsize': 40,
        'IS_REAL': True,
        'LOAD_FROM_FILE': True,
        'flux': 20,
        'lam': 0.1,
        'mu_grid': mu_values_grid,
        'real_name': real_name,
        'IS_3D': is_3d
    }

    # NOTA: Controlla che questa logica del path vada bene per tutti i tuoi 8 dataset
    hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'

    ## DATA LOAD
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
        show_plots = False # IMPORTANTE: impostato a False per i run notturni
    )

    ## ALGORITHM E LOSSES
    kl = KL(back=dataset["back_vec"])
    tv = TVLoss()
    l1 = l1Loss()

    CONFIG_REG = {
        "pgd": {
            "prior": (tv.forward_3D, tv.forward),
            "prior_grad": (tv.grad_3D, tv.grad),
            "prox": (None, None) # PGD non usa prox solitamente
        },
        "prox": {
            "prior": (l1.forward_3D, l1.forward),
            "prior_grad": (None, None),
            "prox": (tresholding_3D, tresholding)
        },
        "md": {
            # "prior": (l1.forward_3D, l1.forward),
            "prior": (tv.forward_3D, tv.forward),
            "prior_grad": (tv.grad_3D, tv.grad),
            # "prior_grad": (l1.grad, l1.grad),
            "prox": (None, None)
        },
    }

    idx = 0 if hparams['IS_3D'] else 1
    cfg = CONFIG_REG[algorithm]
        
    parameters = {
        "max_iter": 10000,
        "tollerance": 1e-6,
        "Lip_reg": dataset["L_th"], 
        "x_init": dataset["x_init"],
        "physics": dataset["physics"],
        "ground_truth": dataset["ground_truth"],
        "back": dataset["back_vec"],
        "lam": hparams['lam'],
        
        "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
        "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
        "single_data_fid": KL_metric,
        
        "prior": cfg["prior"][idx],
        "prox": cfg["prox"][idx],
        "prior_grad": cfg["prior_grad"][idx]
    }

    # RUN OTTIMIZZAZIONE
    W_sum, psnr_vecs, ssim_vecs, results_best, wh_true = RWP(
        dataset, parameters, hparams, 
        optim=Pgd_Backtracking, 
        algorithm=algorithm, 
        mask_type = MASK, 
        eps_f=0
    )

    results = { "W_sum": W_sum,
                "psnr_vecs": psnr_vecs,
                "ssim_vecs": ssim_vecs,
                "results_best": results_best,
                "wh_true": wh_true,
                "ground_truth": dataset["ground_truth"]}

    ## SAVE RESULTS
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
    
    print(f"Salvato con successo in: {save_path}")


# ==========================================
# CONFIGURAZIONE DEL GRID SEARCH NOTTURNO
# ==========================================

if __name__ == "__main__":
    # I tuoi 8 dataset
    datasets_list = [
        '01_tomm20', '02_tomm20', '03_tomm20', '04_tomm20',
        '05_convallaria', '06_convallaria', '07_tubulin', '08_tubulin'
    ] 
    
    
    nz_list = [2]
    algorithms_list = ["pgd", "prox", "md"]

    # Genera tutte le combinazioni (8 * 2 * 2 = 32 esperimenti)
    experiments = list(itertools.product(datasets_list, nz_list, algorithms_list))

    print(f"Inizio sessione: trovati {len(experiments)} esperimenti da lanciare.")
    
    MASK = "masked"

    # Ciclo su tutte le combinazioni con barra di progresso
    for real_name, nz, algo in tqdm(experiments, desc="Progresso Globale"):
        try:
            run_experiment(real_name, nz, algo, MASK)
        except Exception as e:
            print(f"\n[ERRORE] L'esperimento con {real_name}, Nz={nz}, algo={algo} è fallito!")
            print(f"Dettaglio errore: {e}")
        
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            
    print("\nTutti gli esperimenti sono terminati!")


# if __name__ == "__main__":
#     # I tuoi 8 dataset
    
#     nz_list = [1, 2]
#     algorithms_list = ["prox", "pgd"]

#     # Genera tutte le combinazioni (8 * 2 * 2 = 32 esperimenti)
#     experiments = list(itertools.product( nz_list, algorithms_list))

#     print(f"Inizio sessione: trovati {len(experiments)} esperimenti da lanciare.")

#     # Ciclo su tutte le combinazioni con barra di progresso
#     for nz, algo in tqdm(experiments, desc="Progresso Globale"):
#         try:
#             run_experiment(nz, algo)
#         except Exception as e:
#             print(f"\n[ERRORE] L'esperimento, Nz={nz}, algo={algo} è fallito!")
#             print(f"Dettaglio errore: {e}")
        
#         finally:
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             gc.collect()
            
#     print("\nTutti gli esperimenti sono terminati!")