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


hparams = {
    'Nz': 2,
    'pxsize': 40,
    'IS_REAL': True,
    'LOAD_FROM_FILE': True,
    'flux': 30,
    'lam': 0.1
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '05_convallaria' if hparams['IS_REAL'] else 'tubulin'
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

ALGORITHM = "prox"       # "prox" o "pgd"
kl = KL(back=dataset["back_vec"])
tv=TVLoss()
l1 = l1Loss()

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
    # Estrai solo ciò che serve per la visualizzazione o l'analisi futura.
    # Evita di inserire dataset["physics"] o funzioni!
}

torch.save({
    'hparams': hparams,
    'results': results,
    'dataset': clean_dataset
}, save_path)



