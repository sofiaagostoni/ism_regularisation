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
prior_name = 'Drunet_finetune'

torch.manual_seed(0)
pxsizex = 40
flux = 30
Nx = 256
Nz = 1

x = torch.load('Data/Simul_data/tub_level.pth', device)
x = x + 1e-5
x = x/x.max()

PSF, y, avg_y, sum_y, finger_print, physics, data_fidelity = generate_meas_ism(x, Nx, Nz, pxsizex, flux, device)

filepath = hf_hub_download(
    repo_id="deepinv/gradientstep",     
    filename="GSDRUNet_grayscale_torch.ckpt"
)

lamb = 1
stepsize = 3e-2
# sigma_denoiser = (20/255.0)  
sigma_denoiser = 0.08
max_iter = 10000
max_iter = int(max_iter)

class GSPnP(dinv.optim.prior.RED):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.explicit_prior = True

    def forward(self, x, *args, **kwargs):
        out = self.denoiser.potential(x, *args, **kwargs)
        return out
    
class RED_reg(dinv.optim.prior.RED):
    def __init__(self, net):
        """
        Inizializza la rete neurale.

        Args:
            net (nn.Module): Una rete neurale che accetta un input vettoriale e restituisce un output della stessa dimensione.
        """
        super(RED_reg, self).__init__()
        self.net = net
        self.explicit_prior = True

    def forward(self, x,sigma=None,kernel=None):
        
        if sigma is not None:
            diff = x - self.net(x,sigma=sigma) 
        elif kernel is not None:
            diff = x - self.net(x,kernel=kernel)
        else:
            diff = x - self.net(x)  # (x - N(x))
        reg = 0.5 * torch.linalg.norm(diff.flatten(start_dim=1), 2, dim=1)**2  # 0.5*||(x - N(x))||^2
        return reg
    
    def denoising(self,x,training=False,param=1,sigma=None,kernel=None):
        """
        Calcola il valore scalare f(x) = 0.5*||x - net(x)||^2.

        Args:
            x (torch.Tensor): Input tensore.

        Returns:
            torch.Tensor: Valore scalare risultante.
        """
        # Calcolo del residuo: x - net(x)
        x.requires_grad_()
        if sigma is not None:
            aux=self.forward(x,sigma=sigma)
        elif kernel is not None:
            aux=self.forward(x,kernel=kernel)
        else:
            aux=self.forward(x)
        grad_g=torch.autograd.grad(outputs=aux, inputs=x, grad_outputs=torch.ones_like(aux), create_graph=training)[0]
        return x-param*grad_g


if prior_name == 'TV':
    # 1. Inizializza il TV Denoiser
    tv_denoiser = dinv.models.TVDenoiser(n_it_max=20)
    # 2. Usa ScorePrior invece di PnP
    prior = dinv.optim.ScorePrior(denoiser=tv_denoiser)
elif prior_name == 'DnCNN':
    # prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=1, out_channels=1,act_mode='s').to(device))
    # checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    # state_dict = checkpoint["state_dict"]  # Extract only the model weights
    dncnn = dinv.models.DnCNN(in_channels=1, out_channels=1, pretrained="download_lipschitz").to(device)
    prior = dinv.optim.ScorePrior(denoiser= dncnn)

    # prior.denoiser.load_state_dict(checkpoint, strict=False)
elif prior_name == 'DnCNN_chris':
    net=dinv.models.DnCNN(depth=5,in_channels=1,out_channels=1, pretrained = None).to(device)
    checkpoint = torch.load('best_model_checkpoint_IIT_flux_40.pth',weights_only=False,map_location=device)
    state_dict = checkpoint['model_state_dict']
    # nuovo dict filtrato e con chiavi rinominate
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("physics."):
            # salta le chiavi di physics
            continue
        if k.startswith("network.net."):
            # rimuovi il prefisso "network."
            new_k = k[len("network.net."):]
        else:
            new_k = k
        new_state_dict[new_k] = v
    # carica lo state_dict sistemato
    net.load_state_dict(new_state_dict, strict=True)
    # prior = dinv.optim.ScorePrior(denoiser= net)
    prior = RED_reg(net = net)
    
elif prior_name == 'Drunet':
    # prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=1, out_channels=1,act_mode='s').to(device))
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,  pretrained="download", device = device)
    prior = dinv.optim.ScorePrior(denoiser= drunet)
    
elif prior_name == "Drunet_finetune":
    drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                pretrained=None, device=device)
    checkpoint = torch.load("training_drunet/best_model_checkpoint_drunet.pth",
                            map_location=device)
    drunet.load_state_dict(checkpoint["model_state_dict"])
    drunet.eval()
    for p in drunet.parameters():
        p.requires_grad_(False)      # il denoiser e' fisso, ottimizzo l'immagine
    prior = dinv.optim.ScorePrior(denoiser=drunet)
    print(f"Checkpoint caricato: epoca {checkpoint.get('epoch', '?')}")
    
elif prior_name == "GSDrunet":
    prior = GSPnP(denoiser=dinv.models.GSDRUNet(in_channels=1, out_channels=1, act_mode='s').to(device))
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    prior.denoiser.load_state_dict(checkpoint["state_dict"], strict=False)


else:
    print('Unknown Prior')


def custom_output(X):
    return X["est"][1]


backtrck = BacktrackingConfig(gamma=0.1, eta=0.9, max_iter=20)

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
    data_fidelity=data_fidelity,
    params_algo=params_algo,
    early_stop=False,
    max_iter=max_iter,
    crit_conv="cost",
    thres_conv=1e-5,
    backtracking= backtrck,
    get_output=custom_output,
    verbose=True,
    custom_init=lambda observation, physics: {
    "est": ((physics.A_adjoint(observation)/physics.A_adjoint(observation).max()), 
            (physics.A_adjoint(observation)/physics.A_adjoint(observation).max()))
        },  # initialize the optimization with scaled adjoint
)

    #run the model on the problem.
with torch.no_grad():
    x_model, metrics = model(
        y, physics, x_gt=x, compute_metrics=True
    )  # reconstruction with PGD algorithm
print(f"Reconstruction PSNR: {dinv.metric.PSNR()(x, x_model).item():.2f} dB")

print(f"Reconstruction lpips: {dinv.metric.LPIPS(device=device)(x, x_model).item():.3f}")


# plot images. Images are saved in RESULTS_DIR.
# imgs = [avg_y[0], x,  x_model]
imgs = [sum_y[0], x,  x_model]

# plot(
#     imgs,
#     titles=["Input", "GT", "Recon"],
#     save_dir=RESULTS_DIR,
# )

# # plot convergence curves
# if plot_convergence_metrics:
#     plot_curves(metrics)
# %%
