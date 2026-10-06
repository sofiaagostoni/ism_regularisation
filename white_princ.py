#%%
from opt_functions import *
import torch
from opt_functions import * 
from opt_functions.Data_manager.generate_measurments import *

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Genera valori da 1e-4 fino a 1e-0 (1.0)
sigma_values_grid = torch.linspace(1e-6, 1e-2, steps=60)

sigma_values_grid = sigma_values_grid.to(device)

## HYPER PARAM SETTING

hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
    'sigma_grid': sigma_values_grid,
    's': 2e3
}

# Aggiunta dei parametri dipendenti
hparams['IS_3D'] = (hparams['Nz'] > 1)
opt_sec = '3D' if hparams['IS_3D'] else '2D'
hparams['real_name'] = '04_tomm20' if hparams['IS_REAL'] else 'tubulin'                                # '06_convallaria' '05_convallaria' '07_tubulin' '08_tubulin'
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
    show_plots = True,
    normalization_y = True,
    s = hparams['s']
)

# def radial_otf_cutoff(psf, rel_thresh=0.05):
#     """
#     psf: tensore 2D (H,W) — es. PSF out-of-focus mediata sugli elementi
#     Ritorna il raggio di cutoff in frequenza (in pixel-frequency)
#     dove l'OTF radiale scende sotto rel_thresh * picco.
#     """
#     otf = torch.fft.fftshift(torch.fft.fft2(psf)).abs()
#     otf = otf / otf.max()
#     H, W = otf.shape
#     cy, cx = H // 2, W // 2
#     yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
#     r = torch.sqrt((yy - cy).float()**2 + (xx - cx).float()**2)
#     r_int = r.round().long()

#     # profilo radiale medio
#     nbins = r_int.max().item() + 1
#     prof = torch.zeros(nbins)
#     cnt  = torch.zeros(nbins)
#     prof.index_add_(0, r_int.flatten(), otf.flatten())
#     cnt.index_add_(0, r_int.flatten(), torch.ones_like(otf.flatten()))
#     prof = prof / cnt.clamp(min=1)

#     # primo raggio sotto soglia
#     below = (prof < rel_thresh).nonzero()
#     return below[0].item() if len(below) else nbins - 1


noise_image = dataset["noise_image"]
finger_print = dataset["fingerprint"]
physics = dataset["physics"]

# Initial vector
x_0 = dataset['x_init']
# x_0 = (x_0 / x_0.max())

    
# for i in range(25):
#     max_y_i = torch.max(noise_image[i])
#     print(f"before normalization {noise_image[i].max()}")
#     noise_image[i] = (noise_image[i] / max_y_i) * finger_print[i]
#     print(f"after normalization {noise_image[i].max()}")

dataset['noise_image'] = noise_image
dataset['x_init'] = x_0

## ALGORITHM
MASK = 'masked'          # 'whole' 'masked' 'masked_eps'

kl = KL(back=dataset["back_vec"])
drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,  pretrained="download", device = device)

chris_net=dinv.models.DnCNN(depth=5,in_channels=1,out_channels=1, pretrained = None).to(device)
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
chris_net.load_state_dict(new_state_dict, strict=True)


parameters = {
    "max_iter": 400,
    "tollerance": 1e-8,
    "Lip_reg": dataset["L_th"]*1e-2, 
    "x_init": dataset["x_init"],
    "physics": dataset["physics"],
    "ground_truth": dataset["ground_truth"],
    "back": dataset["back_vec"],
    "sigma": 0.01,              # overwritten
    "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
    "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
    "single_data_fid": KL_metric,
    "Pnp" : chris_net,
}

# save_path = f"Results/WP/wp_l1_{opt_sec}_{ALGORITHM}_{MASK}_{hparams['real_name']}.pth"

# Save the grid in hparams so we know what was tested when loading
iter_values_grid = [200, 400, 600, 800, 1000]
hparams['iter_values_grid'] = iter_values_grid

# Dictionary to hold the results for EVERY iteration number
results_all_iters = {}

for i, iter_num in enumerate(tqdm(iter_values_grid, desc="Searching iter grid (RWP)")):
    print(f"\n{'='*40}")
    print(f"   Running max_iter = {iter_num}")
    print(f"{'='*40}")

    # CRITICAL FIX: Update the max_iter parameter for this loop!
    parameters["max_iter"] = iter_num

    W_sum, psnr_vecs, ssim_vecs, sigma_best, results_best_dic, wh_true = RWP_PNP(
        dataset, parameters, hparams, mask_type=MASK, eps_f=1
    )

    # Save the specific results into the dictionary under the key `iter_num`
    results_all_iters[iter_num] = { 
        "W_sum": W_sum,
        "psnr_vecs": psnr_vecs,
        "ssim_vecs": ssim_vecs,
        "sigma_best": sigma_best,
        "results_best": results_best_dic,
        "wh_true": wh_true
    }

# Pack everything up for saving. We only need to save the ground truth once.
results_to_save = {
    "iterations_data": results_all_iters,
    "ground_truth": dataset["ground_truth"]
}

## SAVE RESULTS
save_path = f"Results/WP/wp_pnp_{parameters['Pnp']}_{opt_sec}_{MASK}_{hparams['real_name']}.pth"

print(f"\nSalvataggio risultati in: {save_path}")

clean_dataset = {
    "noise_image": dataset["noise_image"].cpu() if isinstance(dataset["noise_image"], torch.Tensor) else dataset["noise_image"],
    "ground_truth": dataset["ground_truth"].cpu() if isinstance(dataset["ground_truth"], torch.Tensor) else dataset["ground_truth"],
    "clean_image": dataset["clean_image"].cpu() if isinstance(dataset["clean_image"], torch.Tensor) else dataset["clean_image"],
    'meta': dataset["meta"].cpu() if isinstance(dataset["meta"], torch.Tensor) else dataset["meta"],
}

torch.save({
    'hparams': hparams,
    'results': results_to_save, # Using our newly structured dict
    'dataset': clean_dataset
}, save_path)

