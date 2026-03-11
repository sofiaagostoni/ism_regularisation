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
from ism_regularisation.opt_functions.Data_manager.generate_measurments import *

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
from opt_functions.projected_gradient import *


dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tv=TVLoss()
# device = torch.device("cpu")
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)


Nz = 2
pxsize = 40
IS_3D = (Nz > 1)
IS_REAL = False
LOAD_FROM_FILE = True
path = 'Data_results/prel_data/tub_3D.pth'
flux = 30
lam = 0.01


# real data names: 

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

tv=TVLoss()

mu_values_grid = torch.concat(
    [torch.tensor([0, 1e-8]), torch.linspace(1e-5, 1e-1, steps=100)],
    dim=0
    )

mu_values_grid = mu_values_grid.to(device)

ALGORITHM = "pgd"       # "prox" o "pgd"
kl = KL(back=dataset["back_vec"])

parameters = {
    "max_iter": 10000,
    "tollerance": 1e-12,
    "Lip_reg": dataset["L_th"], 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    
    "data_fid": kl.forward_25_3D if IS_3D else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if IS_3D else kl.grad_25,
    "single_data_fid": KL_metric if IS_3D else KL_metric,
    
    
    # "prior": l1_energy,
    "lam": 0.0,
    "prior": tv.forward,
    "prox": tresholding,           # Servirà se ALGORITHM="prox"
    "prior_grad": tv.grad          # Servirà se ALGORITHM="pgd"
}


W_sum, psnr_vecs, ssim_vecs, x_best, wh_true = RWP (mu_values_grid, dataset["noise_image"], dataset["back_vec"], parameters, 
        algorithm= ALGORITHM, is_3d= IS_3D, is_realdata= IS_REAL,
        mask_type="masked", eps=1)


torch.save({"W_sum": W_sum,
            "psnr_vecs": psnr_vecs,
            "ssim_vecs": ssim_vecs,
            "x_best": x_best,
            "wh_true": wh_true,
            "ground_truth": dataset["ground_truth"]
        },'Data_results/WP/wp_tv_maskedeps_3d_convallaria') 





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




#%%
# # ground_truth = results_tv[""]

# x_best_tv = x_best[:,1:2,:,:]
# # gt = ground_truth[:,1:2,:,:]
# # wh_true = results_tv["wh_true"]

# # Z_rec = standardize(noise_image, physics(x_best_tv) + back_vec.view(25,1,1,1))
# # print(torch.mean(Z_rec))

# max_idx_psnr_tv = torch.argmax(psnr_vecs_tv)
# max_idx_ssim_tv = torch.argmax(ssim_vecs_tv)
# min_idx_tv = torch.argmin(W_sum_tv)
# mu_best_tv = mu_values_grid[min_idx_tv]

# results_sob = torch.load("Data_results/Results/wp10_sob_masked")
# W_sum_sob = results_sob["W_sum"]
# psnr_vecs_sob = results_sob["psnr_vecs"]
# ssim_vecs_sob = results_sob["ssim_vecs"]
# x_best_sob = results_sob["x_best"]
# # wh_true = results_sob["wh_true"]

# max_idx_psnr_sob = torch.argmax(psnr_vecs_sob)
# max_idx_ssim_sob = torch.argmax(ssim_vecs_sob)
# min_idx_sob = torch.argmin(W_sum_sob)
# mu_best_sob = mu_values_grid[min_idx_sob]

# results_l1 = torch.load("Data_results/Results/wp10_l1_masked")
# W_sum_l1 = results_l1["W_sum"]
# psnr_vecs_l1 = results_l1["psnr_vecs"]
# ssim_vecs_l1 = results_l1["ssim_vecs"]
# x_best_l1 = results_l1["x_best"]
# # wh_true = results_l1["wh_true"]

# max_idx_psnr_l1 = torch.argmax(psnr_vecs_l1)
# max_idx_ssim_l1 = torch.argmax(ssim_vecs_l1)
# min_idx_l1 = torch.argmin(W_sum_l1)
# mu_best_l1 = mu_values_grid[min_idx_l1]

# print("MICROSSIM")
# # print(micro_structural_similarity(ground_truth.squeeze().detach().cpu().numpy().astype(np.float32), x_best_l1.squeeze().detach().cpu().numpy().astype(np.float32)))
# # print(micro_structural_similarity(gt.squeeze().detach().cpu().numpy().astype(np.float32), x_best_tv.squeeze().detach().cpu().numpy().astype(np.float32)))


# # print(f"Wp teorica {wh_true*M}")
# print(f"Mu ottimale TV {mu_best_tv} con WP {W_sum_tv[min_idx_tv]}, PSNR {psnr_vecs_tv[min_idx_tv].item():.4f} e SSIM {ssim_vecs_tv[min_idx_tv].item():.4f}")
# # print(f"Mu ottimale SOB {mu_best_sob} con WP {W_sum_sob[min_idx_sob]}, PSNR {psnr_vecs_sob[min_idx_sob].item():.4f} e SSIM {ssim_vecs_sob[min_idx_sob].item():.4f}")
# # print(f"Mu ottimale L1 {mu_best_l1} con WP {W_sum_l1[min_idx_l1]}, PSNR {psnr_vecs_l1[min_idx_l1].item():.4f} e SSIM {ssim_vecs_l1[min_idx_l1].item():.4f}")

# # plot([gt, noise_image.sum(0), x_best_tv, x_best[:,0:1,:,:], x_best_sob, x_best_l1], 
# #      titles= [ "GT", "Noise sum", "TV", 'best', 'SOB', 'L1'],
# #         cmap = 'hot',
# #         figsize= (15,5))

# gr.ShowImg(x_best_tv.to("cpu"), pxsize*1e-3)  

# fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# # --- DPNE ---
# axs[0].plot(mu_values_grid, W_sum_tv.to("cpu"), color="tab:blue")
# axs[0].plot(mu_values_grid[min_idx_tv], W_sum_tv.to("cpu")[min_idx_tv], 'go')
# axs[0].axvline(mu_values_grid[min_idx_tv], color="green", linestyle="--")
# axs[0].set_ylabel("WP")
# axs[0].grid(True)

# # --- PSNR ---
# axs[1].plot(mu_values_grid, psnr_vecs_tv.to("cpu"), color="tab:orange")
# axs[1].plot(mu_values_grid[max_idx_psnr_tv], psnr_vecs_tv.to("cpu")[max_idx_psnr_tv], 'ro')
# axs[1].axvline(mu_values_grid[max_idx_psnr_tv], color="red", linestyle="--")
# axs[1].set_ylabel("PSNR (dB)")
# axs[1].grid(True)

# # --- SSIM ---
# axs[2].plot(mu_values_grid, ssim_vecs_tv.to("cpu"), color="tab:green")
# axs[2].plot(mu_values_grid[max_idx_ssim_tv], ssim_vecs_tv.to("cpu")[max_idx_ssim_tv], 'go')
# axs[2].axvline(mu_values_grid[max_idx_ssim_tv], color="green", linestyle="--")
# axs[2].set_ylabel("SSIM")
# axs[2].set_xlabel("$\mu$")
# axs[2].grid(True)

# plt.suptitle("TV")
# plt.tight_layout()
# plt.show()

# # fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# # # --- DPNE ---
# # axs[0].plot(mu_values_grid, W_sum_sob.to("cpu"), color="tab:blue")
# # axs[0].plot(mu_values_grid[min_idx_sob], W_sum_sob.to("cpu")[min_idx_sob], 'go')
# # axs[0].axvline(mu_values_grid[min_idx_sob], color="green", linestyle="--")
# # axs[0].set_ylabel("WP")
# # axs[0].grid(True)

# # # --- PSNR ---
# # axs[1].plot(mu_values_grid, psnr_vecs_sob.to("cpu"), color="tab:orange")
# # axs[1].plot(mu_values_grid[max_idx_psnr_sob], psnr_vecs_sob.to("cpu")[max_idx_psnr_sob], 'ro')
# # axs[1].axvline(mu_values_grid[max_idx_psnr_sob], color="red", linestyle="--")
# # axs[1].set_ylabel("PSNR (dB)")
# # axs[1].grid(True)

# # # --- SSIM ---
# # axs[2].plot(mu_values_grid, ssim_vecs_sob.to("cpu"), color="tab:green")
# # axs[2].plot(mu_values_grid[max_idx_ssim_sob], ssim_vecs_sob.to("cpu")[max_idx_ssim_sob], 'go')
# # axs[2].axvline(mu_values_grid[max_idx_ssim_sob], color="green", linestyle="--")
# # axs[2].set_ylabel("SSIM")
# # axs[2].set_xlabel("$\mu$")
# # axs[2].grid(True)

# # plt.suptitle("SOB")
# # plt.tight_layout()
# # plt.show()


# # fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

# # # --- DPNE ---
# # axs[0].plot(mu_values_grid, W_sum_l1.to("cpu"), color="tab:blue")
# # axs[0].plot(mu_values_grid[min_idx_l1], W_sum_l1.to("cpu")[min_idx_l1], 'go')
# # axs[0].axvline(mu_values_grid[min_idx_l1], color="green", linestyle="--")
# # axs[0].set_ylabel("WP")
# # axs[0].grid(True)

# # # --- PSNR ---
# # axs[1].plot(mu_values_grid, psnr_vecs_l1.to("cpu"), color="tab:orange")
# # axs[1].plot(mu_values_grid[max_idx_psnr_l1], psnr_vecs_l1.to("cpu")[max_idx_psnr_l1], 'ro')
# # axs[1].axvline(mu_values_grid[max_idx_psnr_l1], color="red", linestyle="--")
# # axs[1].set_ylabel("PSNR (dB)")
# # axs[1].grid(True)

# # # --- SSIM ---
# # axs[2].plot(mu_values_grid, ssim_vecs_l1.to("cpu"), color="tab:green")
# # axs[2].plot(mu_values_grid[max_idx_ssim_l1], ssim_vecs_l1.to("cpu")[max_idx_ssim_l1], 'go')
# # axs[2].axvline(mu_values_grid[max_idx_ssim_l1], color="green", linestyle="--")
# # axs[2].set_ylabel("SSIM")
# # axs[2].set_xlabel("$\mu$")
# # axs[2].grid(True)

# # plt.suptitle("l1")
# # plt.tight_layout()
# # plt.show()
# # %%


# # fig, ax1 = plt.subplots(figsize=(10, 5))

# # # Left y-axis: W_sum_l1
# # ax1.plot(mu_values_grid, torch.abs(W_sum_l1), label="RWP", color="tab:blue")
# # ax1.plot(mu_values_grid[min_idx_l1], torch.abs(W_sum_l1)[min_idx_l1], 'go')
# # ax1.set_xlabel("$\mu$")
# # ax1.set_ylabel("$W(\mu)$", color="tab:blue")
# # ax1.tick_params(axis='y', labelcolor="tab:blue")
# # ax1.axvline(mu_values_grid[min_idx_l1].item(), color="green", linestyle="--", alpha=0.7)
# # # ax1.axis([0, 0.2, 0, 20])


# # # Right y-axis: PSNR
# # ax2 = ax1.twinx()
# # ax2.plot(mu_values_grid, psnr_vecs_l1, label="PSNR", color="tab:orange")
# # ax2.plot(mu_values_grid[max_idx_psnr_l1], psnr_vecs_l1[max_idx_psnr_l1], 'ro')
# # ax2.set_ylabel("PSNR (dB)", color="tab:orange")
# # ax2.tick_params(axis='y', labelcolor="tab:orange")

# # ax2.axvline(mu_values_grid[max_idx_psnr_l1].item(), color="red", linestyle="--", alpha=0.7)

# # # Right y-axis: SSIM
# # # ax3 = ax1.twinx()
# # # ax3.plot(mu_values_grid, ssim_vecs_l1, label="SSIM", color="tab:green")
# # # ax3.plot(mu_values_grid[max_idx_ssim_l1], ssim_vecs_l1[max_idx_ssim_l1], 'ro')
# # # ax3.set_ylabel("SSIM (dB)", color="tab:green")
# # # ax3.tick_params(axis='y', labelcolor="tab:green")
# # # ax3.axvline(mu_values_grid[max_idx_ssim_l1].item(), color="red", linestyle="--", alpha=0.7)

# # # Combine legends from both axes
# # lines1, labels1 = ax1.get_legend_handles_labels()
# # lines2, labels2 = ax2.get_legend_handles_labels()
# # # lines3, labels3 = ax3.get_legend_handles_labels()

# # ax1.legend(lines1 + lines2 , labels1 + labels2)

# # plt.title("L1")
# # plt.tight_layout()
# # # plt.axis([0, 0.2, 0, 20000])
# # plt.grid(True)
# # plt.show()
# from matplotlib.ticker import ScalarFormatter

# fig, ax1 = plt.subplots(figsize=(8, 5))

# # Left y-axis: W_sum_l1
# ax1.plot(mu_values_grid, torch.abs(W_sum_tv), label="RWP", color="tab:blue")
# ax1.plot(mu_values_grid[min_idx_tv], torch.abs(W_sum_tv)[min_idx_tv], 'go')
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-1, 1)) # Forza la notazione scientifica
# ax1.yaxis.set_major_formatter(formatter)
# ax1.set_xlabel("$\mu$")
# ax1.set_ylabel("$W(\mu)$", color="tab:blue")
# ax1.tick_params(axis='y', labelcolor="tab:blue")
# ax1.axvline(mu_values_grid[min_idx_tv].item(), color="green", linestyle="--", alpha=0.7)

# # ax1.axis([0, 0.2, 0, 20])


# # Right y-axis: PSNR
# ax2 = ax1.twinx()
# ax2.plot(mu_values_grid, psnr_vecs_tv, label="PSNR", color="tab:orange")
# ax2.plot(mu_values_grid[max_idx_psnr_tv], psnr_vecs_tv[max_idx_psnr_tv], 'ro')
# ax2.set_ylabel("PSNR (dB)", color="tab:orange")
# # ax2.tick_params(axis='y', labelcolor="tab:orange")

# ax2.axvline(mu_values_grid[max_idx_psnr_tv].item(), color="red", linestyle="--", alpha=0.7)

# # Right y-axis: SSIM
# # ax3 = ax1.twinx()
# # ax3.plot(mu_values_grid, ssim_vecs_tv, label="SSIM", color="tab:green")
# # ax3.plot(mu_values_grid[max_idx_ssim_tv], ssim_vecs_tv[max_idx_ssim_tv], 'ro')
# # ax3.set_ylabel("SSIM (dB)", color="tab:green")
# # ax3.tick_params(axis='y', labelcolor="tab:green")
# # ax3.axvline(mu_values_grid[max_idx_ssim_tv].item(), color="red", linestyle="--", alpha=0.7)

# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# # lines3, labels3 = ax3.get_legend_handles_labels()

# ax1.legend(lines1 + lines2 , labels1 + labels2)

# # plt.title("TV")
# plt.tight_layout()
# # plt.legend(loc = 'best')
# # plt.axis([0, 0.2, 0, 20000])
# plt.grid(True)
# plt.show()



# # fig, ax1 = plt.subplots(figsize=(10, 5))

# # # Left y-axis: W_sum_tv
# # ax1.plot(mu_values_grid[0:50], torch.abs(W_sum_sob[0:50]), label="RWP", color="tab:blue")
# # ax1.plot(mu_values_grid[0:50][min_idx_sob], torch.abs(W_sum_sob[0:50])[min_idx_sob], 'go')
# # ax1.set_xlabel("$\mu$")
# # ax1.set_ylabel("$W(\mu)$", color="tab:blue")
# # ax1.tick_params(axis='y', labelcolor="tab:blue")
# # ax1.axvline(mu_values_grid[min_idx_sob].item(), color="green", linestyle="--", alpha=0.7)
# # # ax1.axis([0, 0.2, 0, 20])


# # # Right y-axis: PSNR
# # ax2 = ax1.twinx()
# # ax2.plot(mu_values_grid[0:50], psnr_vecs_sob[0:50], label="PSNR", color="tab:orange")
# # ax2.plot(mu_values_grid[0:50][max_idx_psnr_sob], psnr_vecs_sob[0:50][max_idx_psnr_sob], 'ro')
# # ax2.set_ylabel("PSNR (dB)", color="tab:orange")
# # ax2.tick_params(axis='y', labelcolor="tab:orange")

# # ax2.axvline(mu_values_grid[0:50][max_idx_psnr_sob].item(), color="red", linestyle="--", alpha=0.7)

# # # Right y-axis: SSIM
# # # ax3 = ax1.twinx()
# # # ax3.plot(mu_values_grid[0:50], ssim_vecs_sob[0:50], label="SSIM", color="tab:green")
# # # ax3.plot(mu_values_grid[0:50][max_idx_ssim_sob], ssim_vecs_sob[0:50][max_idx_ssim_sob], 'ro')
# # # ax3.set_ylabel("SSIM (dB)", color="tab:green")
# # # ax3.tick_params(axis='y', labelcolor="tab:green")
# # # ax3.axvline(mu_values_grid[0:50][max_idx_ssim_sob].item(), color="red", linestyle="--", alpha=0.7)

# # # Combine legends from both axes
# # lines1, labels1 = ax1.get_legend_handles_labels()
# # lines2, labels2 = ax2.get_legend_handles_labels()
# # # lines3, labels3 = ax3.get_legend_handles_labels()

# # ax1.legend(lines1 + lines2 , labels1 + labels2)

# # plt.title("SOB")
# # plt.tight_layout()
# # # plt.axis([0, 0.2, 0, 20000])
# # plt.grid(True)
# # plt.show()