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
    'IS_REAL': True,
    'LOAD_FROM_FILE': True,
    'flux': 30,
    'lam': 0.001
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '04_tomm20' if hparams['IS_REAL'] else 'tubulin'
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
        "prior": (tv.forward_3D, tv.forward),
        "prior_grad": (tv.grad_3D, tv.grad),
        "prox": (None, None)
    }
}

# Estraiamo le funzioni in base a IS_3D (0 per 3D, 1 per 2D)
idx = 0 if hparams['IS_3D'] else 1
cfg = CONFIG_REG[ALGORITHM]
    
    
parameters = {
    "max_iter": 100000,
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


# Choose betwen Pgd, Pgd_Fast, Pgd_Fast_Backtracking, Pgd_Bakctracking
SolverClass = Pgd_Backtracking

solver = SolverClass(parameters, algorithm = ALGORITHM, is_3d=hparams['IS_3D'], is_realdata = hparams['IS_REAL'])

results = solver.solve(y=dataset["noise_image"])

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



