# %%
import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import deepinv as dinv
from deepinv.physics import Denoising, GaussianNoise, PoissonNoise
from deepinv.utils.demo import load_url_image, get_image_url
from deepinv.utils.plotting import plot
from deepinv.loss.metric import SSIM, MSE, PSNR, LPIPS

import wandb

from opt_functions.Data_manager.generate_measurments import *
from opt_functions.plot_results import *
from opt_functions.Solver_functions import *
from opt_functions.Data_manager.real_data_load import *
from opt_functions.Solver_functions.projected_gradient import *
from opt_functions.Solver_functions.regularizations import *
from opt_functions.Solver_functions.white_opt_princ import *
from opt_functions.Solver_functions.Kulback_libler import *

from microssim import MicroSSIM, micro_structural_similarity
from skimage.metrics import structural_similarity

import ISM.simulation.PSF_sim as ism
import ISM.analysis.Graph_lib as gr
from scipy.optimize import least_squares
import time
from matplotlib import cm


torch.manual_seed(0)

# %%
dtype = torch.float32
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tv = TVLoss()

hparams = {
    'Nz': 1,
    'pxsize': 40,
    'IS_REAL': False,
    'LOAD_FROM_FILE': True,
    'flux': 20,
    'lam': 0
}

hparams['IS_3D'] = (hparams['Nz'] > 1)
hparams['real_name'] = '01_tomm20' if hparams['IS_REAL'] else 'tubulin'
hparams['path'] = 'Data/Simul_data/tub_3D.pth' if hparams['IS_3D'] else 'Data/Simul_data/tub_level.pth'

dataset = prepare_ism_data(
    is_real=hparams['IS_REAL'],
    real_name=hparams['real_name'],
    load_path=None if not hparams['LOAD_FROM_FILE'] else hparams['path'],
    phantom_type=hparams['real_name'],
    Nx=256, Ny=256,
    Nz=hparams['Nz'],
    pxsize=hparams['pxsize'],
    flux=hparams['flux'],
    device=device,
    show_plots=True
)

kl = KL(back=dataset["back_vec"])
tv = TVLoss()
l1 = l1Loss()

# ==========================================
# CONFIGURAZIONE REGOLARIZZATORI
# ==========================================
CONFIG_REG = {
    "pgd": {
        "prior": (tv.forward_3D, tv.forward),
        "prior_grad": (tv.grad_3D, tv.grad),
        "prox": (None, None)
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
    },
    "rl": {
        "prior": (None, None),
        "prior_grad": (None, None),
        "prox": (None, None)
    }
}

idx = 0 if hparams['IS_3D'] else 1

# ==========================================
# 1. CONFIGURAZIONE DELL'ESPERIMENTO
# ==========================================
N_SAMPLES = 50
ALGORITHM = 'pgd'
MAX_ITER = 2000  # Abbassato per il Monte Carlo (era 2000)
SAVE_PATH = "montecarlo_results.json"

lambdas_to_test = torch.logspace(-4, 0, steps=50).tolist()

wandb.init(
    project="whiteness-montecarlo",
    name=f"Z_mean_analysis_{N_SAMPLES}_samples",
    config={
        "n_samples": N_SAMPLES,
        "algorithm": ALGORITHM,
        "lambdas": lambdas_to_test,
        "max_iter": MAX_ITER
    }
)

# ==========================================
# 2. PRE-GENERAZIONE DEI 50 DATASET
# ==========================================
print(f"Generazione di {N_SAMPLES} dataset in corso...")
datasets_list = []

hparams['LOAD_FROM_FILE'] = False

for i in range(N_SAMPLES):
    torch.manual_seed(i)

    ds = prepare_ism_data(
        is_real=hparams['IS_REAL'],
        real_name=hparams['real_name'],
        load_path=None,
        phantom_type=hparams['real_name'],
        Nx=256, Ny=256, Nz=hparams['Nz'],
        pxsize=hparams['pxsize'],
        flux=hparams['flux'],
        device=device,
        show_plots=False
    )
    datasets_list.append(ds)

print(f"Generazione completata! ({N_SAMPLES} dataset pronti)\n")

# ==========================================
# 3. DIZIONARIO RISULTATI (con checkpoint)
# ==========================================
results_dict = {
    "lambdas": [],
    "z_means_per_lambda": [],   # lista di liste: [lambda_idx][sample_idx] — dati grezzi
    "z_mean_avg": [],
    "z_mean_std": []
}

# ==========================================
# 4. CICLO PRINCIPALE: LAMBDA -> SAMPLES
# ==========================================
cfg = CONFIG_REG[ALGORITHM]

for lam in lambdas_to_test:
    print(f"--- Testando Lambda: {lam:.5f} ---")

    z_means_current_lam = []

    for i, ds in enumerate(datasets_list):

        parameters = {
            "max_iter": MAX_ITER,
            "tollerance": 1e-4,
            "Lip_reg": ds["L_th"],
            "x_init": ds["x_init"],
            "physics": ds["physics"],
            "ground_truth": ds["ground_truth"],
            "back": ds["back_vec"],
            "lam": lam,

            "data_fid": kl.forward_25_3D if hparams['IS_3D'] else kl.forward_25,
            "grad_data_fid": kl.grad_25_3D if hparams['IS_3D'] else kl.grad_25,
            "single_data_fid": KL_metric if hparams['IS_3D'] else KL_metric,

            "prior": cfg["prior"][idx],
            "prox": cfg["prox"][idx],
            "prior_grad": cfg["prior_grad"][idx],

            "callback": None  # Nessun callback nel loop interno
        }

        solver = Pgd_Backtracking(
            parameters,
            algorithm=ALGORITHM,
            is_3d=hparams['IS_3D'],
            is_realdata=hparams['IS_REAL'],
            cfg_prior="l1"
        )

        results = solver.solve(y=ds["noise_image"])
        x_final = results['x_result']

        whiteness_val, z_mean_val, z_mean_true_val = compute_whiteness(
            x_final, ds["noise_image"], ds["ground_truth"],
            ds["physics"], ds["back_vec"], hparams['IS_3D'], mask_type='masked'
        )

        z_means_current_lam.append(z_mean_val.item())
        print(f"  Sample {i+1}/{N_SAMPLES} -> Z_mean = {z_mean_val.item():.5f}")

    # --- STATISTICHE SUI 50 SAMPLE ---
    z_means_np = np.array(z_means_current_lam)
    avg_z = float(np.mean(z_means_np))
    std_z = float(np.std(z_means_np))

    print(f"Risultato per lam={lam:.5f} -> Media Z: {avg_z:.5f} ± {std_z:.5f}\n")

    # Aggiorna il dizionario risultati
    results_dict["lambdas"].append(lam)
    results_dict["z_means_per_lambda"].append(z_means_current_lam)
    results_dict["z_mean_avg"].append(avg_z)
    results_dict["z_mean_std"].append(std_z)

    # --- CHECKPOINT: salva su disco dopo ogni lambda ---
    with open(SAVE_PATH, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"✅ Checkpoint salvato in '{SAVE_PATH}'")

    # --- LOG SU W&B ---
    wandb.log({
        "lambda_val": lam,
        "Z_mean_avg": avg_z,
        "Z_mean_std": std_z,
        "Z_mean_upper_bound": avg_z + std_z,
        "Z_mean_lower_bound": avg_z - std_z
    })

# ==========================================
# 5. PLOT FINALE CON UNCERTAINTY BANDS
# ==========================================
lams = results_dict["lambdas"]
avg  = np.array(results_dict["z_mean_avg"])
std  = np.array(results_dict["z_mean_std"])

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(lams, avg, 'b-o', label=f'Z mean (media su {N_SAMPLES} sample)')
ax.fill_between(
    lams,
    avg - std,
    avg + std,
    color='blue', alpha=0.2, label='± 1 Deviazione Standard'
)
ax.axhline(0, color='red', linestyle='--', label='Z mean Teorico Ideale (0)')
ax.set_xscale('log')
ax.set_xlabel('Lambda (Regolarizzazione)')
ax.set_ylabel('Media Spaziale del Residuo Z')
ax.set_title(f'Test di Whiteness al variare di Lambda ({N_SAMPLES} sample Monte Carlo)')
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig("montecarlo_z_mean_curve.png", dpi=150)

wandb.log({"Final_Z_mean_Curve": wandb.Image(fig)})
plt.show()

wandb.finish()
print("✅ Esperimento completato. Risultati salvati in:")
print(f"   - {SAVE_PATH}  (dati grezzi, riplotabili)")
print(f"   - montecarlo_z_mean_curve.png  (grafico finale)")


# ==========================================
# 6. UTILITY: RIPLOT DA FILE (eseguibile separatamente)
# ==========================================
# Per riplotare in qualsiasi momento senza rieseguire il Monte Carlo:
#
# import json, numpy as np, matplotlib.pyplot as plt
#
# with open("montecarlo_results.json", "r") as f:
#     res = json.load(f)
#
# lams = res["lambdas"]
# avg  = np.array(res["z_mean_avg"])
# std  = np.array(res["z_mean_std"])
#
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.plot(lams, avg, 'b-o', label='Media di Z_mean')
# ax.fill_between(lams, avg - std, avg + std, alpha=0.2, label='± 1 std')
# ax.axhline(0, color='red', linestyle='--', label='Ideale = 0')
# ax.set_xscale('log')
# ax.set_xlabel('Lambda')
# ax.set_ylabel('Z_mean medio')
# ax.legend()
# ax.grid(True, which="both", ls="--", alpha=0.4)
# plt.show()