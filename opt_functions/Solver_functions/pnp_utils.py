"""
Reusable helpers for the PnP / RED grid-search experiments.

All settings that used to be script globals (device, Nx, pxsizex, save_dir,
grids, ...) live in a Config object that is passed explicitly to the functions.

How the denoiser is used depends on the algorithm:
  - gradient algorithms ('MD', 'GD')    -> RED prior   (grad = x - D(x))
  - prox algorithms ('PGD', 'HQS', ...) -> PnP prior   (prox = D(x))
  - 'PNP_MD'                            -> PnP mirror descent (custom iteration,
                                           Burg-entropy mirror step on f, then D)
"""
import os
import csv
import time
import itertools
from dataclasses import dataclass, field

import torch
import deepinv as dinv
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.optim_iterators import OptimIterator
from deepinv.optim.bregman import BurgEntropy
from tqdm.auto import tqdm

from opt_functions.data_preparation import generate_meas_ism


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class Config:
    device: torch.device = field(default_factory=default_device)

    # acquisition / simulation
    Nx: int = 256
    Nz: int = 1
    pxsizex: float = 40

    # prior
    prior_name: str = "Drunet_finetune"       # 'TV', 'Drunet', 'Drunet_finetune'
    ckpt_dir: str = "training_drunet_newloss"

    # paths
    data_dir: str = "Data/Simul_data"
    save_dir: str = "Results/results_gridsearch"

    # grid search
    sigma_grid: torch.Tensor = field(default_factory=lambda: torch.logspace(-3, -1, 10))
    iter_grid: torch.Tensor = field(
        default_factory=lambda: torch.linspace(10, 100, 10).round().long())

    # solver
    algo: str = "PGD"                         # 'MD' (RED), 'PGD' (PnP), 'PNP_MD', 'HQS', ...
    lam: float = 1.0                          # regularization weight (matters for RED)
    kl_step_scale: float = 5e3
    seed: int = 0

    def __post_init__(self):
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def log_path(self):
        return os.path.join(self.save_dir, "log.txt")

    @property
    def csv_path(self):
        return os.path.join(self.save_dir, "summary.csv")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg, cfg):
    tqdm.write(msg)
    with open(cfg.log_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}\n")


def make_tag(name, flux, loss, fidelity_name, prior_name, algo):
    return f"Fidelity{fidelity_name}_{name}_flux{flux}_{prior_name}_loss{loss}_{algo}"


# ---------------------------------------------------------------------------
# Denoiser and prior
# ---------------------------------------------------------------------------

# Algorithms that take a gradient step on the prior -> need RED
GRADIENT_ALGOS = {"MD", "GD"}


class PositiveDenoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, x, sigma, **kwargs):
        return self.denoiser(x, sigma, **kwargs).clamp(0, 1)


def get_denoiser(loss, cfg):
    """Load the bare denoiser (not yet wrapped as PnP or RED)."""
    if cfg.prior_name == "TV":
        return dinv.models.TVDenoiser(n_it_max=20)

    if cfg.prior_name == "Drunet":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained="download", device=cfg.device)
    elif cfg.prior_name == "Drunet_finetune":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained=None, device=cfg.device)
        ckpt_path = os.path.join(cfg.ckpt_dir, f"best_model_checkpoint_drunet_{loss}.pth")
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        drunet.load_state_dict(ckpt["model_state_dict"])
    else:
        raise ValueError(f"Unknown prior: {cfg.prior_name}")

    drunet.eval()
    for p in drunet.parameters():
        p.requires_grad_(False)
    return PositiveDenoiser(drunet)


def wrap_prior(denoiser, algo):
    """RED for gradient-based algorithms, PnP for prox-based ones."""
    if algo.upper() in GRADIENT_ALGOS:
        return dinv.optim.RED(denoiser=denoiser)
    return dinv.optim.PnP(denoiser=denoiser)


# ---------------------------------------------------------------------------
# Data and fidelity
# ---------------------------------------------------------------------------

# Must match the order of generate_meas_ism's return statement
MEAS_KEYS = ("PSF", "y", "avg_y", "sum_y", "finger_print", "physics",
             "df_kl", "df_l2", "L_kl", "L_l2")


def load_data(name, cfg):
    x = torch.load(os.path.join(cfg.data_dir, f"{name}.pth"), map_location=cfg.device)
    return x / x.max() + 1e-5


def make_measurements(x, flux, cfg):
    """Same seed -> same noise realization for every loss/fidelity at this (data, flux)."""
    torch.manual_seed(cfg.seed)
    out = generate_meas_ism(x, cfg.Nx, cfg.Nz, cfg.pxsizex, flux, cfg.device)
    if len(out) != len(MEAS_KEYS):
        raise ValueError(f"generate_meas_ism returned {len(out)} values but MEAS_KEYS "
                         f"has {len(MEAS_KEYS)}. Update MEAS_KEYS to match its return.")
    return dict(zip(MEAS_KEYS, out))


def get_fidelity(fidelity_name, meas, cfg):
    if fidelity_name.upper() == "KL":
        return meas["df_kl"], (1 / meas["L_kl"].item()) * cfg.kl_step_scale
    if fidelity_name.upper() == "L2":
        return meas["df_l2"], 1 / meas["L_l2"].item()
    raise ValueError(f"Unknown fidelity: {fidelity_name}")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PnPMDIteration(OptimIterator):
    """PnP mirror descent: Burg-entropy mirror step on f, then the denoiser."""

    def __init__(self, bregman_potential=None, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.bregman_potential = bregman_potential or BurgEntropy()
        self.eps = eps

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics,
                *args, **kwargs):
        x_prev = X["est"][0].clamp_min(self.eps)          # Burg entropy needs x > 0
        grad_f = cur_data_fidelity.grad(x_prev, y, physics)
        z = self.bregman_potential.grad_conj(
            self.bregman_potential.grad(x_prev) - cur_params["stepsize"] * grad_f)
        x = cur_prior.prox(z, cur_params["g_param"]).clamp_min(self.eps)
        return {"est": (x,), "cost": None}


def get_iteration(algo):
    if algo.upper() == "PNP_MD":
        return PnPMDIteration()
    return algo.upper()


def custom_output(X):
    return X["est"][0].clamp(0, 1)


def init_fn(observation, physics):
    x0 = physics.A_adjoint(observation)
    x0 = x0 / x0.max()
    return {"est": (x0, x0)}


psnr_fn = dinv.metric.PSNR()


def build_model(prior, data_fidelity, stepsize, sigma, n_iter, algo, lam):
    return optim_builder(
        iteration=get_iteration(algo), prior=prior, g_first=False,
        data_fidelity=data_fidelity,
        params_algo={"stepsize": stepsize, "g_param": sigma, "sigma": sigma,
                     "lambda": lam},
        early_stop=False, max_iter=n_iter,
        crit_conv="residual", thres_conv=1e-5,
        get_output=custom_output, verbose=False, custom_init=init_fn,
    )


def grid_search(x, y, physics, prior, data_fidelity, stepsize,
                sigmas, iters, algo, lam, desc="grid search", verbose=False):
    psnr_map = torch.full((len(sigmas), len(iters)), torch.nan)
    best = {"psnr": -float("inf"), "sigma": None, "iter": None, "x": None, "metrics": None}
    combos = list(itertools.product(enumerate(sigmas), enumerate(iters)))

    for (i, s), (j, n_it) in tqdm(combos, desc=desc, leave=False):
        model = build_model(prior, data_fidelity, stepsize, float(s), int(n_it), algo, lam)
        with torch.no_grad():
            x_model, metrics = model(y, physics, x_gt=x, compute_metrics=True)

        if not torch.isfinite(x_model).all():
            if verbose:
                tqdm.write(f"sigma = {float(s):.4f}, iter = {int(n_it)}: diverged (NaN/inf)")
            continue

        p = psnr_fn(x, x_model / x_model.max()).item()
        psnr_map[i, j] = p
        if verbose:
            tqdm.write(f"sigma = {float(s):.4f}, iter = {int(n_it)}  PSNR: {p:.2f} dB")

        if p > best["psnr"]:
            best = {"psnr": p, "sigma": float(s), "iter": int(n_it),
                    "x": x_model.cpu().clone(), "metrics": metrics}
    return best, psnr_map


# ---------------------------------------------------------------------------
# One experiment
# ---------------------------------------------------------------------------

def save_result(result, out_path, cfg):
    torch.save(result, out_path)
    new_file = not os.path.exists(cfg.csv_path)
    with open(cfg.csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["data", "flux", "loss", "fidelity", "algo", "lambda",
                        "sigma", "iter", "psnr"])
        w.writerow([result["name"], result["flux"], result["loss"], result["fidelity"],
                    result["algo"], result["lambda"], result["sigma"], result["iter"],
                    result["psnr"]])


def run_experiment(cfg, name, flux, loss, fidelity_name,
                   algo=None, lam=None, sigmas=None, iters=None,
                   x=None, meas=None, denoiser=None,
                   save=True, overwrite=False, verbose=False):
    """
    Grid search for one (data, flux, loss, fidelity, algo) configuration.

    algo              : 'MD' (RED), 'PGD' (PnP), 'PNP_MD', ... None = cfg.algo.
    lam               : regularization weight. None = cfg.lam.
    sigmas / iters    : override the grids in cfg. Give one value each
                        (e.g. sigmas=[0.01], iters=[50]) for a single reconstruction.
    x, meas, denoiser : pre-computed objects, to avoid reloading inside loops.
                        The denoiser is wrapped as RED or PnP depending on algo.
    save              : write best_rec_<tag>.pth and a row in summary.csv.
    overwrite         : rerun even if the .pth file already exists.

    Returns the result dict, or None if skipped.
    """
    algo = (cfg.algo if algo is None else algo).upper()
    lam = cfg.lam if lam is None else lam
    tag = make_tag(name, flux, loss, fidelity_name, cfg.prior_name, algo)
    out_path = os.path.join(cfg.save_dir, f"best_rec_{tag}.pth")

    if save and not overwrite and os.path.exists(out_path):
        log(f"skip {tag}", cfg)
        return None

    sigmas = cfg.sigma_grid if sigmas is None else torch.as_tensor(sigmas)
    iters = cfg.iter_grid if iters is None else torch.as_tensor(iters)

    if x is None:
        x = load_data(name, cfg)
    if meas is None:
        meas = make_measurements(x, flux, cfg)
    if denoiser is None:
        denoiser = get_denoiser(loss, cfg)
    prior = wrap_prior(denoiser, algo)

    y, physics = meas["y"], meas["physics"]
    data_fidelity, stepsize = get_fidelity(fidelity_name, meas, cfg)

    t0 = time.time()
    best, psnr_map = grid_search(x, y, physics, prior, data_fidelity, stepsize,
                                 sigmas, iters, algo, lam, desc=tag, verbose=verbose)

    result = {
        "x_rec": best["x"], "sigma": best["sigma"], "iter": best["iter"],
        "psnr": best["psnr"], "stepsize": stepsize, "flux": flux,
        "metrics": best["metrics"], "psnr_map": psnr_map,
        "sigma_grid": sigmas, "iter_grid": iters,
        "name": name, "loss": loss, "fidelity": fidelity_name,
        "prior": cfg.prior_name, "algo": algo, "lambda": lam,
        "prior_type": "RED" if algo in GRADIENT_ALGOS else "PnP",
        "x_gt": x.cpu(),
    }

    if save:
        save_result(result, out_path, cfg)

    if best["sigma"] is None:
        log(f"done {tag}: all configurations diverged", cfg)
    else:
        log(f"done {tag}: sigma={best['sigma']:.4f}, iter={best['iter']}, "
            f"PSNR={best['psnr']:.2f} dB ({(time.time() - t0) / 60:.1f} min)", cfg)
    return result