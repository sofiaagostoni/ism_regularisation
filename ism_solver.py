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

from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares


# path = r'Data_results/Real_data'       #es: \\iitfsvge101.iit.local\mms\Data MMS server\STED-ISM\AxialDeconvolution\Convallaria\C

# file = r'05_Convallaria-03-03-2026-17-42-30.h5'

# Nz = 2

# name = "05_convallaria"

# exwl = 488
# emwl = 510

# save_fromh5_totorch(path, file, Nz, name, exwl, emwl) 
#%%
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv= TVLoss()

Nz = 2
pxsize = 40
IS_3D = (Nz > 1)
IS_REAL = True
LOAD_FROM_FILE = True
path = 'Data/Simul_data/tub_level.pth'
flux = 30
lam = 0.001

dataset = prepare_ism_data(
    is_real = IS_REAL,
    real_name='05_convallaria',
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

ALGORITHM = "pgd"       # "prox" o "pgd"
kl = KL(back=dataset["back_vec"])
tv=TVLoss()
l1 = l1Loss()

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


# Choose betwen Pgd, Pgd_Fast, Pgd_Fast_Backtracking, Pgd_Bakctracking
SolverClass = Pgd_Backtracking

solver = SolverClass(parameters, algorithm = ALGORITHM, is_3d=IS_3D, is_realdata = IS_REAL)

results = solver.solve(y=dataset["noise_image"])

plot_results(results, dataset, IS_REAL, IS_3D, pxsize, Nz, x0_sec = 100, y0_sec = 100)


