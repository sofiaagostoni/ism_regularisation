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
import torchmin
from opt_functions import *
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import math
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from microssim import MicroSSIM, micro_structural_similarity
from opt_functions import * 
from opt_functions.Data_manager.generate_measurments import *

import torchmin
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
import time
from scipy.optimize import least_squares

import os
import numpy as np
import torch
from tqdm.auto import tqdm
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from skimage.metrics import structural_similarity
import torchmin
import torch
import math
from opt_functions.Solver_functions.projected_gradient import *


dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv=TVLoss()
# device = torch.device("cpu")
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)

mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100)],
    dim=0
    )

mu_values_grid = mu_values_grid.to(device)


hparams = {
    'Nz': 2,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 30,
    'lam': 0.1,
    'mu_grid': mu_values_grid
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
opt_sec = '3D' if hparams['IS_3D'] else '2D'
hparams['real_name'] = '06_convallaria' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'


# real data names: 

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

tv=TVLoss()

ALGORITHM = "pgd"       # "prox" o "pgd"
MASK = 'whole'
kl = KL(back=dataset["back_vec"])

parameters = {
    "max_iter": 10000,
    "tollerance": 1e-10,
    "Lip_reg": dataset["L_th"], 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,
    
    
    # "prior": l1.forward_3D if IS_3D else l1.forward,
    "lam": hparams['lam'],
    "prior": tv.forward_3D if hparams['IS_3D'] else tv.forward,
    "prox": tresholding_3D if hparams['IS_3D'] else tresholding,           # Servirà se ALGORITHM="prox"
    "prior_grad": tv.grad_3D if hparams['IS_3D'] else tv.grad          # Servirà se ALGORITHM="pgd"
}


W_sum, psnr_vecs, ssim_vecs, x_best, wh_true = RWP (hparams['mu_grid'], dataset["noise_image"], dataset["back_vec"], parameters, 
        algorithm= ALGORITHM, is_3d= hparams['IS_3D'], is_realdata= hparams['IS_REAL'],
        mask_type=MASK, eps=1)

results = { "W_sum": W_sum,
            "psnr_vecs": psnr_vecs,
            "ssim_vecs": ssim_vecs,
            "x_result": x_best,
            "wh_true": wh_true,
            "ground_truth": dataset["ground_truth"]}

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














