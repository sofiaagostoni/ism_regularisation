#%%
"""
Whiteness curves of the 2D SIMULATED runs in Results/WP (white_princ.py, old explicit-prior
version): W (|W_sum|) and PSNR vs the regularization weight mu, one panel per file.
All of them are tub_level ("tubulin"); flux, algorithm, mask and grid differ per file
(shown in the titles). Real-data files (no ground truth, psnr_vecs = None) are skipped.

    python plot_wp_2d.py                     # every Results/WP/wp_*2D_*_tubulin.pth
    python plot_wp_2d.py --files Results/WP/wp_tv_2D_pgd_masked_tubulin.pth

Output: Results/WP/wp_2d_curves.png (+ a table printed: mu at min W, oracle mu, PSNRs).
mu = 0 points are dropped from the plot (log axis).
"""
import argparse
import glob
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_curve(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    h, r = d["hparams"], d["results"]
    gk = next((k for k in h if k.endswith("_grid")), None)
    if gk is None or r.get("psnr_vecs") is None:
        return None
    mu = np.asarray(h[gk], float)
    W = np.abs(np.asarray(r["W_sum"], float))
    P = np.asarray(r["psnr_vecs"], float)
    wt = r.get("wh_true")
    return {"mu": mu, "W": W, "psnr": P, "flux": h.get("flux"),
            "data": os.path.basename(h.get("path", "?")).replace(".pth", ""),
            "W_true": None if wt is None else float(wt), "grid": gk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+",
                    default=sorted(glob.glob("Results/WP/wp_*2D_*_tubulin.pth")))
    ap.add_argument("--out", default="Results/WP/wp_2d_curves.png")
    args = ap.parse_args()

    curves = {}
    for f in args.files:
        c = load_curve(f)
        if c is None:
            print(f"skip {f}: no ground truth / no grid")
            continue
        curves[os.path.basename(f).replace(".pth", "")] = c

    n = len(curves)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows),
                             squeeze=False, layout="constrained")
    print(f"{'file':38s} {'data':10s} flux  mu(min W)  PSNR   mu(oracle)  PSNR   loss")
    for ax, (name, c) in zip(axes.flat, curves.items()):
        keep = c["mu"] > 0
        mu, W, P = c["mu"][keep], c["W"][keep], c["psnr"][keep]
        i, j = int(np.nanargmin(W)), int(np.nanargmax(P))
        ax.semilogx(mu, W, color="tab:blue", label="W")
        if c["W_true"] is not None:
            ax.axhline(c["W_true"], color="tab:blue", ls=":", lw=1, label="W true")
        ax.axvline(mu[i], color="tab:blue", ls="--", lw=1, label=f"min W: {mu[i]:.3g}")
        ax.set_ylabel("W", color="tab:blue")
        ax2 = ax.twinx()
        ax2.semilogx(mu, P, color="tab:red", label="PSNR")
        ax2.axvline(mu[j], color="tab:red", ls="--", lw=1, label=f"oracle: {mu[j]:.3g}")
        ax2.set_ylabel("PSNR (dB)", color="tab:red")
        ax.set_xlabel(c["grid"].replace("_grid", ""))
        ax.set_title(f"{name}\n{c['data']}, flux {c['flux']}", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7)
        print(f"{name:38s} {c['data']:10s} {c['flux']!s:4s}  {mu[i]:9.3g}  {P[i]:5.2f}  "
              f"{mu[j]:9.3g}  {P[j]:5.2f}  {P[j] - P[i]:.2f}")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()
