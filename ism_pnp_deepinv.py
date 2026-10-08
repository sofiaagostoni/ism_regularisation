#%%
"""Run the full grid of experiments. Safe to restart: finished runs are skipped."""
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # must be set before importing torch

import gc
import itertools
import traceback

import torch
from tqdm.auto import tqdm

from opt_functions.Solver_functions.pnp_utils import (
    Config, log, make_tag, load_data, make_measurements, get_denoiser, run_experiment,
)
#%%
# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
cfg = Config(
    prior_name="Drunet_finetune",
    save_dir="Results/results_gridsearch",
    # sigma_grid=torch.logspace(-3, -1, 10),
    # iter_grid=torch.linspace(10, 100, 10).round().long(),
)

data_list     = ["tub_level", "tub_balls"]
flux_list     = [5, 10, 20, 40, 60]
loss_list     = ["L2", "L1", "FFL", "L2_FFL", "L1_FFL"]
fidelity_list = ["KL", 'l2']
algo_list     = ["MD", "PGD"]   # MD -> RED, PGD -> PnP, "PNP_MD" -> PnP mirror descent

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    n_total = (len(data_list) * len(flux_list) * len(loss_list)
               * len(fidelity_list) * len(algo_list))
    outer = tqdm(total=n_total, desc="experiments")

    for name in data_list:
        x = load_data(name, cfg)

        for flux in flux_list:
            meas = make_measurements(x, flux, cfg)

            for loss in loss_list:
                denoiser = get_denoiser(loss, cfg)   # wrapped as RED/PnP per algo

                for fidelity_name, algo in itertools.product(fidelity_list, algo_list):
                    try:
                        run_experiment(cfg, name, flux, loss, fidelity_name, algo=algo,
                                       x=x, meas=meas, denoiser=denoiser)
                    except Exception:
                        tag = make_tag(name, flux, loss, fidelity_name,
                                       cfg.prior_name, algo)
                        log(f"FAILED {tag}\n{traceback.format_exc()}", cfg)
                    finally:
                        outer.update(1)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                del denoiser

    outer.close()



#%%
# SINGLE EXPERIMENT: choose here

# cfg = Config(prior_name="Drunet_finetune",
#              save_dir="Results/results_gridsearch")
 
# # SINGLE EXPERIMENT: choose here
 
# data     = "tub_level"     # 'tub_level', 'tub_balls'
# flux     = 10
# loss     = "L2"            # 'L2', 'L1', 'FFL', 'L2_FFL', 'L1_FFL'
# fidelity = "KL"            # 'KL', 'L2'
 
# algo     = "MD"            # 'MD'     -> RED prior (gradient step on the prior)
#                            # 'PGD'    -> PnP prior (denoiser as prox)
#                            # 'PNP_MD' -> PnP mirror descent (Burg entropy)
# lam      = 1.0             # regularization weight (mainly matters for RED / MD)
 
# # None = full grids from cfg; one value each = single reconstruction, no search
# sigmas = None              # e.g. [0.01]
# iters  = None              # e.g. [50]
 
# res = run_experiment(cfg, data, flux, loss, fidelity,
#                      algo=algo, lam=lam,
#                      sigmas=sigmas, iters=iters,
#                      save=False,        # True: also write .pth + summary.csv
#                      overwrite=True,
#                      verbose=True)
 
# if res["sigma"] is not None:
#     print(f"\n[{res['algo']} / {res['prior_type']}] Best: sigma = {res['sigma']:.4f}, "
#           f"max_iter = {res['iter']}, PSNR = {res['psnr']:.2f} dB")
# else:
#     print(f"\n[{res['algo']} / {res['prior_type']}] All configurations diverged. "
#           f"Try a smaller lam or cfg.kl_step_scale.")


# %%
