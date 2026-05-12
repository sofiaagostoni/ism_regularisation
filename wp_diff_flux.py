import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# Import dalle tue librerie custom
from opt_functions import *
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from opt_functions.Data_manager.generate_measurments import *

def run_flux_experiment(algo='pgd', n_realizations=10):
    # Setup base
    mu_values_grid = torch.linspace(1e-6, 1, steps=100) # Ripristinato steps a 100 per RWP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_base = Path("Results/Experiment_Flux_MultiRealization")
    output_base.mkdir(parents=True, exist_ok=True)
    mu_values_grid = mu_values_grid.to(device)
    
    history = {
        'flux': [],
        'psnr_mean': [],
        'psnr_std': [],
        'ssim_mean': [],
        'ssim_std': [],
        'best_mu_mean': [],
        'raw_data': [] 
    }
    
    flux_levels = [10, 15, 20, 30, 40, 50, 80]
    
    for f in flux_levels:
        print(f"\n--- Analisi Flux: {f} ---")
        
        # Pre-allocazione dei vettori (tensori) per le realizzazioni
        run_psnr = torch.zeros(n_realizations, device=device)
        run_ssim = torch.zeros(n_realizations, device=device)
        run_mu   = torch.zeros(n_realizations, device=device)
       
        for r in tqdm(range(n_realizations), desc=f"Realizzazioni Flux {f}"):
            
            hparams = {
                'Nz': 1, 'pxsize': 40, 'IS_REAL': False, 'LOAD_FROM_FILE': True,
                'flux': f, 'lam': 0.001, 'mu_grid': mu_values_grid,
                'IS_3D': False, 'real_name': 'tubulin',
                'path': 'Data/Simul_data/tub_level.pth'
            }

            # Caricamento dati (genera rumore nuovo ogni volta)
            dataset = prepare_ism_data(
                is_real=hparams['IS_REAL'], real_name=hparams['real_name'],
                load_path=hparams['path'], phantom_type=hparams['real_name'],
                Nx=256, Ny=256, Nz=hparams['Nz'], pxsize=hparams['pxsize'], 
                flux=hparams['flux'], device=device, show_plots=False
            )

            # Configurazione Algoritmo
            ALGORITHM = algo
            MASK = 'masked'
            kl = KL(back=dataset["back_vec"])
            tv, l1 = TVLoss(), l1Loss()

            CONFIG_REG = {
                "pgd": {"prior": (tv.forward_3D, tv.forward), "prior_grad": (tv.grad_3D, tv.grad), "prox": (None, None)},
                "prox": {"prior": (l1.forward_3D, l1.forward), "prior_grad": (None, None), "prox": (tresholding_3D, tresholding)},
            }

            idx = 1 if not hparams['IS_3D'] else 0
            cfg = CONFIG_REG[ALGORITHM]
            
            parameters = {
                "max_iter": 10000, "tollerance": 1e-8, "Lip_reg": dataset["L_th"], 
                "x_init": dataset["x_init"], "physics": dataset["physics"],
                "ground_truth": dataset["ground_truth"], "back": dataset["back_vec"],
                "lam": hparams['lam'],
                "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
                "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
                "prior": cfg["prior"][idx], "prox": cfg["prox"][idx], "prior_grad": cfg["prior_grad"][idx],
                "single_data_fid": KL_metric,
            }

            # Esecuzione RWP
            _, _, _, mu_best, results_best, _ = RWP(
                dataset, parameters, hparams, optim=Pgd_Backtracking, 
                algorithm=ALGORITHM, mask_type=MASK, eps_f=1
            )

            # Riempimento dei vettori per indice
            run_psnr[r] = results_best['psnr'][-1]
            run_ssim[r] = results_best['ssim'][-1]
            run_mu[r]   = mu_best if isinstance(mu_best, (float, int)) else mu_best.item()

            if r == 0:
                fig, _ = gr.ShowImg(results_best['x_result'].cpu(), 40*1e-3)
                fig.savefig(output_base / f"recon_{ALGORITHM}_flux_{f}_ref.pdf", format='pdf', bbox_inches='tight', dpi=300)
                plt.close(fig)

        # Ora puoi usare torch.mean e torch.std senza problemi
        history['flux'].append(f)
        history['psnr_mean'].append(run_psnr.mean().item())
        history['psnr_std'].append(run_psnr.std().item())
        history['ssim_mean'].append(run_ssim.mean().item())
        history['ssim_std'].append(run_ssim.std().item())
        history['best_mu_mean'].append(run_mu.mean().item())
        
        # Salviamo i tensori nel dizionario raw per sicurezza
        history['raw_data'].append({
            'flux': f,
            'psnr_all': run_psnr.cpu(),
            'ssim_all': run_ssim.cpu(),
            'mu_all': run_mu.cpu()
        })

    torch.save(history, output_base / f"experiment_avg_{ALGORITHM}.pth")
    print(f"\nEsperimento completato! Risultati in: {output_base}")

if __name__ == "__main__":
    # Puoi cambiare n_realizations qui
    run_flux_experiment(algo='pgd', n_realizations=10)