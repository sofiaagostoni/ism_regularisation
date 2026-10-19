"""
Custom optimization iterations and the model builder.

Custom iterations (they store intermediate iterates, so ONE run of max(iters)
iterations gives the result for every iteration count of the grid):
    'PGD'      PnP proximal gradient:  x <- D(x - tau grad f(x))
    'PNP_MD'   PnP mirror descent (Burg entropy), then D
    'PROJ_GD'  explicit priors, x >= 0, optional adaptive backtracking:
    'PROX_GD'  gradient step on R if smooth (WCRR, TV_smooth), prox step on R
               if not (L1). The two names are the same iteration.
Any other algo name is passed to deepinv as a built-in iteration.
"""
from functools import partial

import torch
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.optim_iterators import OptimIterator
from deepinv.optim.bregman import BurgEntropy
from tqdm.auto import tqdm


class SnapshotIterator(OptimIterator):
    """Stores the iterate after n iterations for every n in snap_iters.
    The iterate after n iterations is the input of call n."""

    def __init__(self, snap_iters=(), **kwargs):
        super().__init__(**kwargs)
        self.snap_iters = {int(n) for n in snap_iters}
        self.snapshots = {}
        self.n_calls = 0

    def _snap(self, x):
        if self.n_calls in self.snap_iters:
            self.snapshots[self.n_calls] = x.detach().clamp(0, 1).cpu().clone()
        self.n_calls += 1


class PnPPGDIteration(SnapshotIterator):
    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        x_prev = X["est"][0]
        self._snap(x_prev)
        z = x_prev - cur_params["stepsize"] * cur_data_fidelity.grad(x_prev, y, physics)
        x = cur_prior.prox(z, cur_params["g_param"])
        return {"est": (x,), "cost": None}


class PnPMDIteration(SnapshotIterator):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.bregman = BurgEntropy()
        self.eps = eps                      # Burg entropy needs x > 0

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        self._snap(X["est"][0])
        x_prev = X["est"][0].clamp_min(self.eps)
        grad_f = cur_data_fidelity.grad(x_prev, y, physics)
        z = self.bregman.grad_conj(self.bregman.grad(x_prev) - cur_params["stepsize"] * grad_f)
        x = cur_prior.prox(z, cur_params["g_param"]).clamp_min(self.eps)
        return {"est": (x,), "cost": None}


class ProxGDIteration(SnapshotIterator):
    """
    Proximal gradient descent on F = f + lam*R over x >= 0, for explicit priors.

    How R is used depends on prior.smooth_prior:
        smooth R (WCRR, TV_smooth):  x+ = max(0, x - tau * (grad f + lam grad R))
        non-smooth R (L1):           x+ = max(0, prox_{tau lam R}(x - tau * grad f))
    'PROJ_GD' and 'PROX_GD' are the same iteration; the prior decides.

    backtracking=False: tau fixed.
    backtracking=True : each iteration first tries a larger step,
        L = 1 / min(tau_k / delta, 1 / s),
    then multiplies L by eta until, with S the smooth part of F,
        S(x+) <= S(x) + <grad S(x), x+ - x> + L/2 ||x+ - x||^2,   tau = 1 / L.
    After max_bt loops, or once L >= L_max, the safe step 1/L_max is taken.
    The returned "cost" is S(x+), reused as S(x) by the next iteration.
    """

    def __init__(self, backtracking=False, s=1.0, eta=2.0, delta=0.9,
                 L_max=float("inf"), max_bt=50, **kwargs):
        super().__init__(**kwargs)
        self.backtracking = backtracking
        self.s, self.eta, self.delta = s, eta, delta
        self.L_max, self.max_bt = L_max, max_bt
        self.tau_k = None                   # last accepted step
        self.L_history = []
        self.n_backtracks = []

    def forward(self, X, cur_data_fidelity, cur_prior, cur_params, y, physics, *args, **kwargs):
        x_prev = X["est"][0]
        self._snap(x_prev)
        lam, tau, sigma = cur_params["lambda"], cur_params["stepsize"], cur_params["g_param"]
        smooth_R = getattr(cur_prior, "smooth_prior", True)

        def S(x):                       # smooth part of F
            out = cur_data_fidelity(x, y, physics)
            return out + lam * cur_prior(x, sigma) if smooth_R else out

        def step(tau):                  # one (proximal) gradient step, then x >= 0
            z = x_prev - tau * g
            if not smooth_R:
                z = cur_prior.prox(z, sigma, gamma=tau * lam)
            return z.clamp_min(0)

        g = cur_data_fidelity.grad(x_prev, y, physics)
        if smooth_R:
            g = g + lam * cur_prior.grad(x_prev, sigma)

        if not self.backtracking:
            x = step(tau)
            return {"est": (x,), "cost": S(x)}

        S_prev = X.get("cost")
        S_prev = float((S(x_prev) if S_prev is None else S_prev).sum())

        if self.tau_k is None:
            self.tau_k = tau
        L = 1.0 / min(self.tau_k / self.delta, 1.0 / self.s)
        n_bt = 0
        while True:
            x = step(1.0 / L)
            S_new = S(x)
            d = x - x_prev
            bound = S_prev + float((g * d).sum()) + 0.5 * L * float((d * d).sum())
            if float(S_new.sum()) <= bound:          # NaN fails the test -> smaller step
                break
            L *= self.eta
            n_bt += 1
            if L >= self.L_max or n_bt > self.max_bt:
                L = self.L_max
                x = step(1.0 / L)
                S_new = S(x)
                break

        self.tau_k = 1.0 / L
        self.L_history.append(L)
        self.n_backtracks.append(n_bt)
        return {"est": (x,), "cost": S_new}


CUSTOM_ITERATIONS = {"PGD": PnPPGDIteration, "PNP_MD": PnPMDIteration,
                     "PROJ_GD": ProxGDIteration, "PROX_GD": ProxGDIteration}


def get_iteration(algo, snap_iters=(), **bt_kwargs):
    """Custom iteration object, or the algo name for a deepinv built-in iteration.
    bt_kwargs (backtracking settings) are used by PROJ_GD / PROX_GD only."""
    cls = CUSTOM_ITERATIONS.get(algo.upper())
    if cls is None:
        return algo.upper()
    if cls is ProxGDIteration:
        return cls(snap_iters=snap_iters, **bt_kwargs)
    return cls(snap_iters=snap_iters)


def init_fn(observation, physics, init_choice):
    x_adj = physics.A_adjoint(observation)      # right shape, device and dtype
    if init_choice == "adjoint_y":
        x0 = x_adj / x_adj.abs().max().clamp_min(1e-12)
    elif init_choice == "random":
        x0 = torch.randn_like(x_adj)
    elif init_choice == "constant":
        x0 = 0.5 * torch.ones_like(x_adj)
    else:
        raise ValueError(f"Unknown init: {init_choice}")
    return {"est": (x0, x0.clone())}


def build_model(prior, data_fidelity, iteration, stepsize, sigma, lam, n_iter,
                init_choice="adjoint_y", deepinv_backtracking=False):
    return optim_builder(
        iteration=iteration, prior=prior, data_fidelity=data_fidelity, g_first=False,
        params_algo={"stepsize": stepsize, "g_param": sigma, "lambda": lam},
        backtracking=deepinv_backtracking, early_stop=False, max_iter=n_iter,
        get_output=lambda X: X["est"][0].clamp(0, 1),
        custom_init=partial(init_fn, init_choice=init_choice),
        verbose=False,
    )