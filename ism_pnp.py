# %%
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *
        
import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares

dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.cuda.manual_seed(0)

## GENERATE DATA ---------
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
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
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

ALGORITHM = "pgd"       # "prox" o "pgd"

kl = KL(back=dataset["back_vec"])

drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,  pretrained="download", device = device)
    
    


noise_image = dataset["noise_image"]
finger_print = dataset["fingerprint"]
physics = dataset["physics"]

# Initial vector
x_0 = dataset['x_init']
x_0 = (x_0 / x_0.max())

    
for i in range(25):
    max_y_i = torch.max(noise_image[i])
    print(f"before normalization {noise_image[i].max()}")
    noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]
    print(f"after normalization {noise_image[i].max()}")

sigma = 1e-3
parameters = {
    "max_iter": 600,
    "tollerance": 1e-12,
    "Lip_reg": dataset["L_th"]*1e-3, 
    "x_init": x_0,
    "physics": dataset["physics"],
    "back": dataset["back_vec"],
    "sigma": sigma,
    
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,
    
    "Pnp" : drunet,

}


# %%
def pnp_ism(y, back, parameters_pnp, device):
    
    data_fid = parameters_pnp["data_fid"]
    grad_data_fid  = parameters_pnp["grad_data_fid"] 
    single_fid  = parameters_pnp["single_data_fid"] 
    tollerance  = parameters_pnp["tollerance"]      
    max_iter  = parameters_pnp["max_iter"]     
    x_init  = parameters_pnp["x_init"]     
    sigma = parameters_pnp["sigma"]  
    L_max  = parameters_pnp["Lip_reg"]      
    pnp   = parameters_pnp["Pnp"]  
    physics   = parameters_pnp["physics"]

    # Sposto tutti i tensori sul device
    funct = torch.zeros(max_iter, device=device)
    iter_err = torch.zeros(max_iter, device=device)
    
    min_distance = - float('inf')

    x_k_prec = x_init.to(device)  # Assicurati che x_init sia sul device
    y = y.to(device)
    back = back.to(device)

    tau = 1/L_max
    print(f"x_0 max {x_init.max()}")
    # for k in tqdm(range(max_iter), desc="Iterations") :
    for k in range(max_iter):
        with torch.no_grad():

            x_k_succ = torch.max(x_k_prec -  tau* grad_data_fid(y, x_k_prec,  physics), torch.tensor(0))
            x_k_succ_prepnp = x_k_succ
            x_k_succ = x_k_succ/ x_k_succ.max()
            x_k_succ[:,1:2] = pnp(x_k_succ[:,1:2], sigma)
            x_k_succ = torch.clamp(x_k_succ, 0, 1)
                
        funct[k] = data_fid(y, x_k_succ, physics)
        iter_err[k] = torch.norm(x_k_prec - x_k_succ, 'fro') / torch.norm(x_k_prec, 'fro')

        
        if (k + 1) % 100 == 0:
            print()
            print(f'Iter {k+1}/{max_iter}')
            print()
            print(f"sigma = {sigma}")
            print(f"x_kprec.max = {x_k_prec.max()}")
            print(f"physiscs(x_kprec).max = {physics(x_k_prec.repeat(25,1,1,1)).max()}")
            print(f"Max y: {y.max()}")
            print(f"Max xksucc pre norm : {x_k_succ_prepnp.max()}")
            print(f"Max xksucc post pnp: {x_k_succ.max()}")
            plot([y.sum(0), x_init, x_k_succ[:,1:2]],
                    cmap = 'hot',
                    rescale_mode = 'clip',
                    suptitle = f"iteration {k}")
        
        if iter_err[k] < tollerance:
            print(f"Convergence reached at iteration = {k}")
            funct = funct[0:k]
            iter_err = iter_err[0:k]
            
            # lpips_vec = lpips_vec[0:k] 
            break

        x_k_prec = x_k_succ
        

    return x_k_succ, funct.detach(), iter_err.detach()



# ALGORITHM
# bm3d = dinv.models.BM3D() 
# tgv = dinv.models.TGVDenoiser()
# tv = dinv.models.TVDenoiser()
# med_filter = dinv.models.MedianFilter()
# dncnn = dinv.models.DnCNN(in_channels=1, out_channels=1, depth=20, pretrained="download") # sigma not used
# weights = torch.load("weights_drunet/best_model_checkpoint_drunet.pth", weights_only= False)

# drunet.load_state_dict(weights['model_state_dict'], strict = True)



# MULTIPLE SIGMA -------------------
sigma_list = [ 5e-4]
# sigma_list = [3e-1]

for sigma in sigma_list:
    parameters['sigma'] = sigma
    x_result_drunet, KL_vec_drunet, iter_drunet = pnp_ism(noise_image, dataset['back_vec'], parameters, device)

results = {'x_result': x_result_drunet, 'funct': KL_vec_drunet, 'iter_err': iter_drunet,
                'diff_fid': None if hparams['IS_REAL'] else diff_fid,
                'psnr': None if hparams['IS_REAL'] else psnr_vec,
                'ssim': None if hparams['IS_REAL'] else ssim_vec}

plot_results(results, dataset, hparams['IS_REAL'], hparams['IS_3D'], hparams['pxsize'], x0_sec = 100, y0_sec = 100)


plot([ dataset['noise_image'].sum(0), x_result_drunet[:,1:2]],
      cmap = 'hot',
      rescale_mode = 'clip')
gr.ShowImg(x_result_drunet[:,1:2].to("cpu"), hparams['pxsize']*1e-3)  

plot_met(KL_vec_drunet.cpu(), iter_drunet.cpu())
# %%
