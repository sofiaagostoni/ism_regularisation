# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

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


# path = r'Data/Real_data'       #es: \\iitfsvge101.iit.local\mms\Data MMS server\STED-ISM\AxialDeconvolution\Convallaria\C

# file = r'01_TOMM20_AF488-17-03-2026-17-34-10.h5'

# Nz = 2

# name = "01_tomm20"

# exwl = 493
# emwl = 518

# save_fromh5_totorch(path, file, Nz, name, exwl, emwl) 
#%%
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv= TVLoss()


hparams = {
    'Nz': 2,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
    'lam': 9.09e-2
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '02_tomm20' if hparams['IS_REAL'] else 'tubulin'
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

#%%

ALGORITHM = "md"       # "prox" o "pgd"

kl = KL(back=dataset["back_vec"])
tv=TVLoss()
l1 = l1Loss()

# Definiamo le logiche per ogni algoritmo
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
        "prior": (l1.forward_3D, l1.forward),
        "prior_grad": (l1.grad, l1.grad),
        # "prior": (tv.forward_3D, tv.forward),
        # "prior_grad": (tv.grad, tv.grad),
        "prox": (None, None)
    },
    "rl": {
        "prior": (None, None),
        "prior_grad": (None, None),
        "prox": (None, None)
    }
}

# Estraiamo le funzioni in base a IS_3D (0 per 3D, 1 per 2D)
idx = 0 if hparams['IS_3D'] else 1
cfg = CONFIG_REG[ALGORITHM]
    
    
parameters = {
    "max_iter": 10000,
    "tollerance": 1e-12,
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
    "prior_grad": cfg["prior_grad"][idx]
}

#%%

# Choose betwen Pgd, Pgd_Fast, Pgd_Fast_Backtracking, Pgd_Bakctracking, RichLucy
SolverClass = Pgd_Backtracking

solver = SolverClass(parameters, algorithm = ALGORITHM, is_3d=hparams['IS_3D'], is_realdata = hparams['IS_REAL'])

results = solver.solve(y=dataset["noise_image"])

real_tag = "real" if hparams['IS_REAL'] else "sim"
sect_tag = "3D" if hparams['IS_3D'] else "2D"

save_path = f"Results/ism_results/ism_l1_{sect_tag}_{real_tag}_{ALGORITHM}_{hparams['real_name']}_lam{parameters['lam']}.pth"

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



# M = dataset['noise_image'].numel() 
# mask_type = "masked"
# lambda_d = dataset['physics'](results['x_result']) + dataset['back_vec'].view(-1, 1, 1, 1)
# if hparams['IS_3D']:
#     lambda_d = lambda_d.sum(1).unsqueeze(1)

# # Calcolo di Z per il risultato corrente
# if mask_type == "masked":
#     Z = standardize_unbiased_masked(dataset['noise_image'], lambda_d) 
# elif mask_type == "masked_eps":
#     Z = standardize_unbiased_masked_eps(dataset['noise_image'], lambda_d, eps = 1)
# elif mask_type == "whole":
#     Z = standardize(dataset['noise_image'], lambda_d)
# else:
#     raise ValueError(f"Metodo di masking non riconosciuto: {mask_type}")

# # Metriche Whiteness
# wh = whiteness_measure(Z)
# W_sum = M * wh

# print(f"RWP value with lam = {hparams['lam']} is {W_sum}")



#%%

# RUN different RL

# SolverClass = Pgd_Backtracking


# for max_iter in [100000]:
    
#     parameters['max_iter'] = max_iter

#     solver = SolverClass(parameters, algorithm = ALGORITHM, is_3d=hparams['IS_3D'], is_realdata = hparams['IS_REAL'])

#     results = solver.solve(y=dataset["noise_image"])

#     real_tag = "real" if hparams['IS_REAL'] else "sim"
#     sect_tag = "3D" if hparams['IS_3D'] else "2D"

#     save_path = f"Results/ism_results/ism_{sect_tag}_{real_tag}_{ALGORITHM}_{hparams['real_name']}_lam{parameters['lam']}_iter{max_iter}.pth"

#     clean_dataset = {
#         "noise_image": dataset["noise_image"].cpu() if isinstance(dataset["noise_image"], torch.Tensor) else dataset["noise_image"],
#         "ground_truth": dataset["ground_truth"].cpu() if isinstance(dataset["ground_truth"], torch.Tensor) else dataset["ground_truth"],
#         "clean_image":dataset["clean_image"].cpu() if isinstance(dataset["clean_image"], torch.Tensor) else dataset["clean_image"],
#         'meta': dataset["meta"].cpu() if isinstance(dataset["meta"], torch.Tensor) else dataset["meta"],
#     }

#     torch.save({
#         'hparams': hparams,
#         'results': results,
#         'dataset': clean_dataset
#     }, save_path)
    
    
# #%%
    

# Nz = 2 if hparams['Nz'] else 1

# grid = ism.GridParameters()

# grid.N = 5              # number of detector elements in each dimension
# grid.pxsizex = hparams['pxsize']      # pixel size of the simulation space (nm)
# grid.pxdim = 50e3       # detector element size in real space (nm)
# grid.pxpitch = 75e3     # detector element pitch in real space (nm)
# grid.M = 500            # total magnification of the optical system (e.g. 100x objective followed by 5x telescope)
# grid.Nz = Nz
# grid.pxsizez = 700
# exPar = ism.simSettings()
# exPar.wl = 640 # excitation wavelength (nm)
# exPar.mask_sampl = 31
# emPar = exPar.copy()
# emPar.wl = 660 # emission wavelength (nm)
# z_shift = 0 #nm

# # Parametri di base
# IS_3D = hparams.get('IS_3D', False)
# IS_REAL = hparams.get('IS_REAL', False)
# ALGORITHM = "pgd"
# real_name = hparams.get('real_name', 'tubulin')
# lam = hparams.get('lam', 0)
# iter_list = [100000]

# sect_tag = "3D" if IS_3D else "2D"
# real_tag = "real" if IS_REAL else "sim"

# # Liste per salvare i valori finali
# valid_iters = [18169]
# final_psnr = []
# final_ssim = []
# final_funct_diff = []
# final_funct = []


# for max_iter in iter_list:
#     file_path = f"Results/ism_results/ism_{sect_tag}_{real_tag}_{ALGORITHM}_{real_name}_lam{lam}_iter{max_iter}.pth"
    
#     try:
#         # Carichiamo i dati salvati
#         saved_data = torch.load(file_path, map_location='cpu', weights_only = False)
        
#         # Estraiamo l'ultimo valore dei vettori salvati in results
#         # Assumiamo che diff_fid sia salvato come lista o tensore 1D
#         x_result = saved_data['results']['x_result']
#         psnr_vec = saved_data['results']['psnr']
#         ssim_vec = saved_data['results']['ssim']
#         funct = saved_data['results']['funct']
#         funct_diff_vec = saved_data['results']['diff_fid'] # o 'funct_metric' a seconda di come è stato salvato
        
#         # Se sono tensori di PyTorch, estraiamo l'ultimo elemento come float
#         if isinstance(psnr_vec, torch.Tensor):
#             psnr_val = psnr_vec[-1].item() if not IS_REAL else None
#         else:
#             psnr_val = psnr_vec[-1] if not IS_REAL else None
            
#         if isinstance(ssim_vec, torch.Tensor):
#             ssim_val = ssim_vec[-1].item() if not IS_REAL else None
#         else:
#             ssim_val = ssim_vec[-1] if not IS_REAL else None
            
#         if isinstance(funct_diff_vec, torch.Tensor):
#             funct_diff_val = funct_diff_vec[-1].item() if not IS_REAL else None
#         else:
#             funct_diff_val = funct_diff_vec[-1] if not IS_REAL else None
            
            
#         if isinstance(funct, torch.Tensor):
#             funct_val = funct[-1].item()
#         else:
#             funct_val = funct_val[-1]
        
#         valid_iters.append(max_iter)
#         final_psnr.append(psnr_val)
#         final_ssim.append(ssim_val)
#         final_funct.append(funct_val)
#         final_funct_diff.append(funct_diff_val)
        
#         fig, ax = gr.ShowImg(x_result[:,1:2], grid.pxsizex*1e-3)
#         # fig.suptitle(f"max iter = {max_iter}")
#     except FileNotFoundError:
#         print(f"File non trovato: {file_path}, salto questo punto.")

# # Plot dei risultati
# if len(valid_iters) > 0:
#     # Se stiamo usando dati reali, PSNR e SSIM potrebbero non essere validi (non c'è ground truth)
#     if not IS_REAL:
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
#         # Plot PSNR
#         ax1.plot(valid_iters, final_psnr, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
#         ax1.set_title('Final PSNR vs Max Iter', fontsize=14)
#         ax1.set_xlabel('Max Iterations', fontsize=12)
#         ax1.set_ylabel('Final PSNR (dB)', fontsize=12)
#         ax1.grid(True, linestyle='--', alpha=0.7)
#         ax1.set_xticks(valid_iters) # Mostra esattamente i tick delle iterazioni
        
#         # Plot SSIM
#         ax2.plot(valid_iters, final_funct, marker='s', linestyle='-', color='g', linewidth=2, markersize=8)
#         ax2.set_title('Functional vs Max Iter', fontsize=14)
#         ax2.set_xlabel('Max Iterations', fontsize=12)
#         ax2.set_ylabel('Functional', fontsize=12)
#         ax2.grid(True, linestyle='--', alpha=0.7)
#         ax2.set_xticks(valid_iters)
        
#         # Plot Funct Diff
#         ax3.plot(valid_iters, final_funct_diff, marker='^', linestyle='-', color='r', linewidth=2, markersize=8)
#         ax3.set_title('Final Data Fidelity (KL) vs Max Iter', fontsize=14)
#         ax3.set_xlabel('Max Iterations', fontsize=12)
#         ax3.set_ylabel('Fidelity Diff', fontsize=12)
#         ax3.grid(True, linestyle='--', alpha=0.7)
#         ax3.set_xticks(valid_iters)
        
#         plt.tight_layout()
#         plt.show()
    
#     else:
#         # Per dati reali plottiamo tutto l'andamento del funzionale
#         plt.figure(figsize=(8, 5))
        
#         # 1. Tracciamo la linea continua di tutto il funzionale (senza marker)
#         # Se l'array funct parte dall'iterazione 1, l'asse x di default partirà da 0. 
#         # Possiamo forzare l'asse x a partire da 1 creando un array per le x:
#         x_all = range(1, len(funct) + 1)
#         plt.semilogx(x_all, funct, linestyle='-', color='r', linewidth=2)
        
#         # 2. Estraiamo le coordinate (x, y) per i marker
#         # Nota: usiamo (i-1) perché gli array in Python sono 0-indexed
#         # Assicuriamoci che l'iterazione richiesta non superi la lunghezza di funct
#         marker_x = [i for i in valid_iters if i <= len(funct)]
#         marker_y = [funct[i - 1] for i in marker_x]
        
#         # 3. Sovrapponiamo i marker sui punti specifici
#         plt.plot(marker_x, marker_y, marker='^', linestyle='', color='b', markersize=10, label='Solution')
        
#         plt.title('Data Fidelity (KL)', fontsize=14)
#         plt.xlabel('Iterations', fontsize=12)
#         plt.ylabel(r"$ \sum_d \mathrm{KL}(A_d x_k,\ y_d) + \lambda TV$", fontsize=12)
#         plt.grid(True, linestyle='--', alpha=0.7)
        
#         # Mostriamo sull'asse x esattamente i punti di early stopping
#         # plt.xticks(marker_x) 
        
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
# else:
#     print("Nessun dato valido trovato per generare i grafici.")
# %%
