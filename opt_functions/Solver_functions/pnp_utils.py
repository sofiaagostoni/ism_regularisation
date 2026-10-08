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
import os
import sys
_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_WCRR_ROOT = os.path.join(_WORKSPACE, "weakly_convex_ridge_regularizer")

if _WCRR_ROOT not in sys.path:
    sys.path.append(_WCRR_ROOT)

from models import utils
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
    lam_grid : torch.Tensor = field(default_factory=lambda: torch.logspace(-3, -1, 10))

    # solver
    algo: str = "PGD"                         # 'MD' (RED), 'PGD' (PnP), 'PNP_MD', 'HQS', ...
    lam: float = 1.0                          # regularization weight (matters for RED)
    kl_step_scale: float = 5e3
    l2_step_scale: float = 1

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
    

class ProjGDIteration(OptimIterator):
    """Projected gradient descent on f + lambda*R, projection onto x >= 0."""

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics,
                *args, **kwargs):
        x_prev = X["est"][0]
        lam, tau, sigma = cur_params["lambda"], cur_params["stepsize"], cur_params["g_param"]

        grad = cur_data_fidelity.grad(x_prev, y, physics) + lam * cur_prior.grad(x_prev, sigma)
        x = (x_prev - tau * grad).clamp_min(0)

        cost = cur_data_fidelity(x, y, physics) + lam * cur_prior(x, sigma)
        return {"est": (x,), "cost": cost}
    
class WCRRPrior(dinv.optim.Prior):
    def __init__(self, sigma=25.0, device="cpu", n_power_steps=200):
        super().__init__()
        self.explicit_prior = True
        self.model = utils.load_model("WCRR-CNN", str(device))

        # Update the conv layer's Lipschitz bound once, at construction
        with torch.no_grad():
            self.model.conv_layer.spectral_norm(mode="power_method", n_steps=n_power_steps)
            
                # Move tensors stored as plain attributes (spline grids) to the device
        self.model.to(device)
        for m in self.model.modules():
            for attr, val in list(vars(m).items()):
                if torch.is_tensor(val) and not isinstance(val, torch.nn.Parameter):
                    setattr(m, attr, val.to(device))

        self.sigma = torch.tensor([sigma], device=device).view(-1, 1, 1, 1)

    def _sigma_for(self, x, sigma):
        # match the batch size of x
        s = self.sigma if sigma is None else torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        return s.reshape(-1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)

    def fn(self, x, sigma, *args, **kwargs):
        # R(x); assumes the model exposes its cost as model.cost(x, sigma)
        return self.model.cost(x, self._sigma_for(x, sigma))

    def grad(self, x, sigma, *args, **kwargs):
        return self.model.grad(x, self._sigma_for(x, sigma))
    
    def get_mu(self, *args, **kwargs):
        return self.model.get_mu()


def get_denoiser(loss, cfg):
    """Load the bare denoiser (not yet wrapped as PnP or RED)."""
    if cfg.prior_name == "TV":
        return dinv.models.TVDenoiser(n_it_max=20)
    
    if cfg.prior_name == 'weakly_convex':
        return WCRRPrior(sigma=25.0, device=cfg.device)
    
    if cfg.prior_name == "Drunet":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained="download", device=cfg.device)
        return PositiveDenoiser(drunet)
    elif cfg.prior_name == "Drunet_finetune":
        drunet = dinv.models.DRUNet(in_channels=1, out_channels=1,
                                    pretrained=None, device=cfg.device)
        ckpt_path = os.path.join(cfg.ckpt_dir, f"best_model_checkpoint_drunet_{loss}.pth")
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        drunet.load_state_dict(ckpt["model_state_dict"])
        return PositiveDenoiser(drunet)
    else:
        raise ValueError(f"Unknown prior: {cfg.prior_name}")

    


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


def get_fidelity(fidelity_name, meas, cfg, lam=1.0, prior=None):
    name = fidelity_name.upper()
    if name == "KL":
        datafid = meas["df_kl"]
        L_f = meas["L_kl"].item() *cfg.kl_step_scale
    elif name == "L2":
        datafid = meas["df_l2"]
        L_f = meas["L_l2"].item() *cfg.l2_step_scale
    else:
        raise ValueError(f"Unknown fidelity: {fidelity_name}")

    L_tot = L_f
    if cfg.prior_name == "weakly_convex":
        L_R = float(prior.get_mu())   # Lipschitz constant of grad R
        L_tot = L_f + lam * L_R

    stepsize = 1 / L_tot
    return datafid, stepsize


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
    if algo.upper() == "PROJ_GD":
        return ProjGDIteration()
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
        backtracking=True,
        early_stop=False, max_iter=n_iter,
        crit_conv="residual", thres_conv=1e-5,
        get_output=custom_output, verbose=False, custom_init=init_fn,
    )


def grid_search(x, y, physics, prior, data_fidelity, stepsizes,
                sigmas, iters, lambdas, algo, desc="grid search", verbose=False):
    psnr_map = torch.full((len(sigmas), len(iters), len(lambdas)), torch.nan)
    best = {"psnr": -float("inf"), "sigma": None, "iter": None, "lambda": None,
            "stepsize": None, "x": None, "metrics": None}
    combos = list(itertools.product(enumerate(sigmas), enumerate(iters), enumerate(lambdas)))

    for (i, s), (j, n_it), (k, l) in tqdm(combos, desc=desc, leave=False):
        l = float(l)
        model = build_model(prior, data_fidelity, stepsizes[k], float(s), int(n_it), algo, l)
        with torch.no_grad():
            x_model, metrics = model(y, physics, x_gt=x, compute_metrics=True)

        if not torch.isfinite(x_model).all():
            if verbose:
                tqdm.write(f"sigma = {float(s):.4f}, iter = {int(n_it)}, lambda = {l:.4g}: diverged (NaN/inf)")
            continue

        p = psnr_fn(x, x_model / x_model.max()).item()
        psnr_map[i, j, k] = p
        if verbose:
            tqdm.write(f"sigma = {float(s):.4f}, iter = {int(n_it)}, lambda = {l:.4g}  PSNR: {p:.2f} dB")

        if p > best["psnr"]:
            best = {"psnr": p, "sigma": float(s), "iter": int(n_it), "lambda": l,
                    "stepsize": stepsizes[k], "x": x_model.cpu().clone(), "metrics": metrics}
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
                   algo=None, lam=None, sigmas=None, iters=None, lambdas = None,
                   x=None, meas=None, denoiser=None,
                   save=True, overwrite=False, verbose=True):
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
    if lambdas is None:
        lambdas = cfg.lam_grid if lam is None else [lam]   # lam singolo = nessuna griglia
    lambdas = torch.as_tensor(lambdas)

    if x is None:
        x = load_data(name, cfg)
    if meas is None:
        meas = make_measurements(x, flux, cfg)
    if denoiser is None:
        denoiser = get_denoiser(loss, cfg)
    prior = denoiser if cfg.prior_name == "weakly_convex" else wrap_prior(denoiser, algo)

    y, physics = meas["y"], meas["physics"]
    data_fidelity, _ = get_fidelity(fidelity_name, meas, cfg, float(lambdas[0]), prior)
    stepsizes = [get_fidelity(fidelity_name, meas, cfg, float(l), prior)[1] for l in lambdas]

    t0 = time.time()
    best, psnr_map = grid_search(x, y, physics, prior, data_fidelity, stepsizes,
                                 sigmas, iters, lambdas, algo, desc=tag, verbose=verbose)


    result = {
        "x_rec": best["x"], "sigma": best["sigma"], "iter": best["iter"],
        "lambda": best["lambda"], "psnr": best["psnr"], "stepsize": best["stepsize"],
        "flux": flux, "metrics": best["metrics"], "psnr_map": psnr_map,
        "sigma_grid": sigmas, "iter_grid": iters, "lambda_grid": lambdas,
        "name": name, "loss": loss, "fidelity": fidelity_name,
        "prior": cfg.prior_name, "algo": algo,
        "prior_type": "RED" if algo in GRADIENT_ALGOS else "PnP",
        "x_gt": x.cpu(),
    }

    if save:
        save_result(result, out_path, cfg)

    if best["sigma"] is None:
        log(f"done {tag}: all configurations diverged", cfg)
    else:
        log(f"done {tag}: sigma={best['sigma']:.4f}, iter={best['iter']}, lambda = {best['lambda']} "
            f"PSNR={best['psnr']:.2f} dB ({(time.time() - t0) / 60:.1f} min)", cfg)
    return result