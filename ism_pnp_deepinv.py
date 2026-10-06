#%%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
from deepinv.optim.optimizers import optim_builder
from huggingface_hub import hf_hub_download
from deepinv.utils.parameters import get_GSPnP_params
from deepinv.optim.data_fidelity import PoissonLikelihood
from deepinv.optim.optimizers import optim_builder
from deepinv.utils.demo import load_dataset, load_degradation
from deepinv.utils.plotting import plot, plot_curves, plot_inset
from deepinv.optim import L2, TVPrior, BacktrackingConfig


import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.data_preparation import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares

#%%
prior_name = 'Drunet'

torch.manual_seed(0)
pxsizex = 40
flux = 50
Nx = 256
Nz = 1

x = torch.load('Data/Simul_data/tub_level.pth', device)
# x = x/x.max()
# x = x + 1e-5

PSF, y, avg_y, sum_y, finger_print, physics, data_fidelity_poisson, data_fidelity_l2, L_kl, L_l2 = generate_meas_ism(x, Nx, Nz, pxsizex, flux, device)


gr.ShowImg(x.cpu(), pxsize_x = pxsizex)
gr.ShowImg(sum_y[0].cpu(), pxsize_x = pxsizex)
# gr.ShowDataset(y.cpu())

x = x/x.max()

print(f"Lip constat for KL = {L_kl.item()}")
print(f"Lip constat for l2 = {L_l2.item()}")


#%%
lamb = 0.001
# stepsize = (1 / L_kl.item()) 
stepsize = 1e-4
# stepsize = 1 / L_l2.item()
print(f"Stepsize = {stepsize}")
sigma_denoiser = 0.08
max_iter = 200
max_iter = int(max_iter)

if prior_name == 'TV':
    # 1. Inizializza il TV Denoiser
    tv_denoiser = dinv.models.TVDenoiser(n_it_max=20)
    # 2. Usa ScorePrior invece di PnP
    prior = dinv.optim.PnP(denoiser=tv_denoiser)
elif prior_name == 'DnCNN':
    # prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=1, out_channels=1,act_mode='s').to(device))
    # checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    # state_dict = checkpoint["state_dict"]  # Extract only the model weights
    dncnn = dinv.models.DnCNN(in_channels=1, out_channels=1, pretrained="download_lipschitz").to(device)
    prior = dinv.optim.PnP(denoiser= dncnn)

    # prior.denoiser.load_state_dict(checkpoint, strict=False)    
elif prior_name == 'Drunet':
    # prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=1, out_channels=1,act_mode='s').to(device))
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,  pretrained="download", device = device)
    prior = dinv.optim.PnP(denoiser= drunet)    
elif prior_name == "Drunet_finetune":
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                pretrained=None, device=device)
    checkpoint = torch.load("training_drunet/best_model_checkpoint_drunet.pth",
                            map_location=device)
    drunet.load_state_dict(checkpoint["model_state_dict"])
    drunet.eval()
    for p in drunet.parameters():
        p.requires_grad_(False)      # il denoiser e' fisso, ottimizzo l'immagine
    prior = dinv.optim.PnP(denoiser=drunet)
    print(f"Checkpoint caricato: epoca {checkpoint.get('epoch', '?')}")    
else:
    print('Unknown Prior')


def custom_output(X):
    return X["est"][1]


# backtrck = BacktrackingConfig(gamma=0.1, eta=0.9, max_iter=20)

params_algo = {
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "sigma": sigma_denoiser,
    }

# instantiate the algorithm class to solve the IP problem.
model = optim_builder(
    iteration="PGD",
    prior=prior,
    g_first=False,
    data_fidelity=data_fidelity_poisson,
    params_algo=params_algo,
    early_stop=False,
    max_iter=max_iter,
    crit_conv="residual",
    thres_conv=1e-5,
    get_output=custom_output,
    verbose=True,
    custom_init=lambda observation, physics: {
    "est": ((physics.A_adjoint(observation)/physics.A_adjoint(observation).max()), 
            (physics.A_adjoint(observation)/physics.A_adjoint(observation).max()))
        },  # initialize the optimization with scaled adjoint
)

with torch.no_grad():
    x_model, metrics = model(
        y, physics, x_gt=x, compute_metrics=True
    )  # reconstruction with PGD algorithm
print(f"Reconstruction PSNR: {dinv.metric.PSNR()(x/x.max(), x_model/x_model.max()).item():.2f} dB")



# plot images. Images are saved in RESULTS_DIR.
# imgs = [avg_y[0], x,  x_model]
imgs = [sum_y[0], x,  x_model]



gr.ShowImg(x.cpu(), pxsize_x = pxsizex)
gr.ShowImg(sum_y[0].cpu(), pxsize_x = pxsizex)
gr.ShowImg(x_model.cpu(), pxsize_x = pxsizex)

psnr_per_iter = metrics["psnr"][0]       # [0] = first image in the batch
res_per_iter  = metrics["residual"][0]   # ||x_k - x_{k-1}|| / ||x_k||

dinv.utils.plotting.plot_curves(metrics)  # plots all of them

# %%
