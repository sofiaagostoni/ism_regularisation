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


Nz = 2
pxsize = 40
IS_3D = (Nz > 1)
IS_REAL = True
LOAD_FROM_FILE = True
path = 'Data_results/prel_data/tub_3D.pth'
flux = 30
lam = 0.01


# real data names: 

dataset = prepare_ism_data(
    is_real = IS_REAL,
    real_name='07_tubulin',
    load_path = None if not LOAD_FROM_FILE else path,
    phantom_type='tubulin',
    Nx = 256, Ny = 256, 
    Nz = Nz , 
    pxsize = pxsize, 
    flux = flux,
    device = device,
    show_plots = True
)
#%%

tv=TVLoss()

mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100)],
    dim=0
    )

mu_values_grid = mu_values_grid.to(device)

ALGORITHM = "pgd"       # "prox" o "pgd"
MASK = 'masked'
kl = KL(back=dataset["back_vec"])

parameters = {
    "max_iter": 10000,
    "tollerance": 1e-8,
    "Lip_reg": dataset["L_th"], 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    
    "data_fid": kl.forward_25_3D if IS_3D else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if IS_3D else kl.grad_25,
    "single_data_fid": KL_metric if IS_3D else KL_metric,
    
    
    # "prior": l1.forward_3D if IS_3D else l1.forward,
    "lam": lam,
    "prior": tv.forward_3D if IS_3D else tv.forward,
    "prox": tresholding_3D if IS_3D else tresholding,           # Servirà se ALGORITHM="prox"
    "prior_grad": tv.grad_3D if IS_3D else tv.grad          # Servirà se ALGORITHM="pgd"
}


W_sum, psnr_vecs, ssim_vecs, x_best, wh_true = RWP (mu_values_grid, dataset["noise_image"], dataset["back_vec"], parameters, 
        algorithm= ALGORITHM, is_3d= IS_3D, is_realdata= IS_REAL,
        mask_type=MASK, eps=1)

save_path = f"Data_results/WP/wp_3D_{ALGORITHM}_{MASK}_{dataset['real_name']}.pth"

print(f"Salvataggio risultati in: {save_path}")

torch.save({"W_sum": W_sum,
            "psnr_vecs": psnr_vecs,
            "ssim_vecs": ssim_vecs,
            "x_best": x_best,
            "wh_true": wh_true,
            "ground_truth": dataset["ground_truth"]
        },save_path) 





#%%

# mu_values_grid = torch.concat(
#     [torch.tensor([0, 1e-8]), torch.linspace(1e-4, 10e-1, steps=200)],
#     dim=0
# )

# mu_values_grid = torch.concat(
#     [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100), torch.linspace(1e-1, 10e-1, steps=50)],
#     dim=0
# )
mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100)],
    dim=0
)

results_tv = torch.load("Data_results/WP/wp_l1_maskedeps_3d_tub")
W_sum = results_tv["W_sum"]
psnr_vecs_tv = results_tv["psnr_vecs"]  
ssim_vecs_tv = results_tv["ssim_vecs"]
x_best = results_tv["x_best"]

gr.ShowImg(x_best[:,1:2].cpu(), pxsize_x = pxsize*1e-3)
gr.ShowImg(x_best[:,0:1].cpu(), pxsize_x = pxsize*1e-3)


plot_wp_results(mu_values_grid, dataset, W_sum, psnr_vecs_tv, ssim_vecs_tv, is_real= IS_REAL, title="WP", layout="stacked")



