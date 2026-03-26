#%%
import os
from opt_functions import *
import numpy as np
import torch
from tqdm.auto import tqdm
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
from opt_functions import *
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from microssim import MicroSSIM, micro_structural_similarity
from opt_functions import * 
from opt_functions.Data_manager.generate_measurments import *


dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100)],
    dim=0
    )

mu_values_grid = mu_values_grid.to(device)

## HYPER PARAM SETTING

hparams = {
    'Nz': 2,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
    'lam': 0.001,
    'mu_grid': mu_values_grid
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
opt_sec = '3D' if hparams['IS_3D'] else '2D'
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'                                # '06_convallaria' '05_convallaria' '07_tubulin' '08_tubulin'
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
    show_plots = True
)


## ALGORITHM

ALGORITHM = "pgd"       # "prox" o "pgd"
MASK = 'masked_eps'          # 'whole' 'masked' 'masked_eps'

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
        # "prior": (tv.forward_3D, tv.forward),
        # "prior_grad": (tv.grad_3D, tv.grad),
        "prior_grad": (l1.grad, l1.grad),
        "prox": (None, None)
    },
}

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
    "single_data_fid": KL_metric,
    
    "prior": cfg["prior"][idx],
    "prox": cfg["prox"][idx],
    "prior_grad": cfg["prior_grad"][idx]
}

save_path = f"Results/WP/wp_{opt_sec}_{ALGORITHM}_{MASK}_{hparams['real_name']}.pth"

print(f"White princ per risultati in: {save_path}")


W_sum, psnr_vecs, ssim_vecs, results_best, wh_true = RWP (dataset, parameters, hparams, optim = Pgd_Backtracking ,algorithm= ALGORITHM, mask_type=MASK, eps_f=5)

results = { "W_sum": W_sum,
            "psnr_vecs": psnr_vecs,
            "ssim_vecs": ssim_vecs,
            "results_best": results_best,
            "wh_true": wh_true,
            "ground_truth": dataset["ground_truth"]}


## SAVE RESULTS

save_path = f"Results/WP/wp_{opt_sec}_{ALGORITHM}_{MASK}_{hparams['real_name']}.pth"

print(f"Salvataggio risultati in: {save_path}")

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





#%%


# best_mu, best_rwp, RWP_vec, best_results = RWP_Adam_1Step (dataset, parameters, hparams, optim = Pgd_Backtracking ,algorithm= ALGORITHM, mask_type=MASK, eps=2, max_outer_iter=300, lrate = 1e-3, stepsize = 50, gamma=0.9)


# save_path = f"Results/WP/wp_optim_{opt_sec}_{ALGORITHM}_{MASK}_{hparams['real_name']}.pth"

# print(f"Salvataggio risultati in: {save_path}")

# results = { "best_mu": best_mu, 
#            "W_sum": best_rwp,
#             "results_best": best_results,
#             'rwp_vec': RWP_vec,
#             "ground_truth": dataset["ground_truth"]}


# clean_dataset = {
#     "noise_image": dataset["noise_image"].cpu() if isinstance(dataset["noise_image"], torch.Tensor) else dataset["noise_image"],
#     "ground_truth": dataset["ground_truth"].cpu() if isinstance(dataset["ground_truth"], torch.Tensor) else dataset["ground_truth"],
#     "clean_image":dataset["clean_image"].cpu() if isinstance(dataset["clean_image"], torch.Tensor) else dataset["clean_image"],
#     'meta': dataset["meta"].cpu() if isinstance(dataset["meta"], torch.Tensor) else dataset["meta"],
# }

# torch.save({
#     'hparams': hparams,
#     'results_optim': results,
#     'dataset': clean_dataset
# }, save_path)




