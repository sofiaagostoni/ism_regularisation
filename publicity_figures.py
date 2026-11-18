"""
Publicity figures for model-based ISM reconstruction of real data.

Builds, for each real dataset:
  * <name>_split_tv.png / _split_l1.png : diagonal split, raw data | reconstruction
  * <name>_bands.png                    : three diagonal bands, raw | TV | L1 reconstruction
  * <name>_wipe_tv.mp4 / .gif           : animated diagonal wipe raw <-> reconstruction

"Raw" is the sum over the 25 SPAD channels (= open-pinhole confocal image).
The reconstruction shown is the in-focus plane (index 1) of the 2-plane 3D
S2ISM reconstruction; plane 0 is the out-of-focus background.

Usage:
    python publicity_figures.py                       # all datasets, magma
    python publicity_figures.py --data 02_TUB --cmap glow_cyan
    python publicity_figures.py --data 02_TUB --crop 300 200 500   # y0 x0 size (px)
    python publicity_figures.py --no-anim

16:9 frames (slides / screens), PNG only, 1920 x 1080 (--width 3840 for 4K):
    python publicity_figures.py --aspect 16:9 --no-anim --out publicity_16x9
    python publicity_figures.py --aspect 16:9 --no-anim --data 02_TUB --ypos 0.3
The square field is cut to a full-width 16:9 window, at vertical position --ypos
(0 = top, 0.5 = centre, 1 = bottom; default: the dataset's "ypos" in DATASETS, else 0.5).
With --crop Y0 X0 SIZE the window keeps the centre and the height of that square and is
widened to 16:9 (shifted / clamped to stay inside the image).
The seams run along the frame diagonal (corner to corner), so in a 16:9 frame they are
tilted at ~29 degrees instead of 45; square frames are unchanged.
"""
import argparse
import io
import os
import subprocess

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

RES = "Results/ism_results"

# Real datasets with both a TV- and an L1-regularised S2ISM reconstruction (λ selected per dataset).
DATASETS = {
    # ypos: vertical position of the --aspect (e.g. 16:9) window, 0 = top, 1 = bottom
    "02_TUB": dict(title="Tubulin", ypos=0.4,
                   tv="ism_3D_real_md_02_TUB_lam0.1930698.pth",
                   l1="ism_l1_3D_real_md_02_TUB_lam0.2442053.pth"),
    "03_H3": dict(title="H3K9 · nucleus", ypos=0.55,
                  tv="ism_3D_real_md_03_H3_lam0.0184207.pth",
                  l1="ism_l1_3D_real_md_03_H3_lam0.07543119.pth"),
    "02_vimentine": dict(title="Vimentin", ypos=0.3,
                         tv="ism_3D_real_md_02_vimentine_lam0.2442053.pth",
                         l1="ism_l1_3D_real_md_02_vimentine_lam0.6250552.pth"),
    # TOMM20 (mitochondria): λ as selected in results_ism_pnp.ipynb
    "01_tomm20": dict(title="TOMM20 · mitochondria", ypos=0.75,
                      tv="ism_3D_real_md_01_tomm20_lam0.00304.pth",
                      l1="ism_l1_3D_real_md_01_tomm20_lam0.00708.pth"),
    "02_tomm20": dict(title="TOMM20 · mitochondria", ypos=0.6,
                      tv="ism_3D_real_md_02_tomm20_lam0.0485.pth",
                      l1="ism_l1_3D_real_md_02_tomm20_lam0.00672.pth"),
    "03_tomm20": dict(title="TOMM20 · mitochondria",
                      tv="ism_3D_real_md_03_tomm20_lam0.0485.pth",
                      l1="ism_l1_3D_real_md_03_tomm20_lam0.00672.pth"),
    "04_tomm20": dict(title="TOMM20 · mitochondria",
                      tv="ism_3D_real_md_04_tomm20_lam0.00102.pth",
                      l1="ism_l1_3D_real_md_04_tomm20_lam0.0738.pth"),
}

CMAPS = {
    "magma": plt.get_cmap("magma"),
    "inferno": plt.get_cmap("inferno"),
    "hot": plt.get_cmap("afmhot"),
    "glow_cyan": LinearSegmentedColormap.from_list(
        "glow_cyan", ["#000000", "#03141c", "#0b4f6c", "#14a3c7", "#7fe7ff", "#ffffff"]),
    "glow_green": LinearSegmentedColormap.from_list(
        "glow_green", ["#000000", "#03160a", "#0c5a26", "#23b14d", "#a6ff9e", "#ffffff"]),
}

FONT = dict(family="DejaVu Sans", weight="bold")


# ----------------------------------------------------------------------------- data
def load_dataset(key):
    d = DATASETS[key]
    ck = torch.load(os.path.join(RES, d["tv"]), map_location="cpu", weights_only=False)
    raw = ck["dataset"]["noise_image"].float().sum(dim=(0, 1)).numpy()   # (Nx, Ny)
    tv = ck["results"]["x_result"][0, 1].float().numpy()                  # in-focus plane
    pxsize_nm = float(ck["hparams"]["pxsize"])
    l1 = torch.load(os.path.join(RES, d["l1"]), map_location="cpu",
                    weights_only=False)["results"]["x_result"][0, 1].float().numpy()
    return dict(raw=raw, tv=tv, l1=l1, pxsize_nm=pxsize_nm, title=d["title"])


def normalise(img, pct=99.8, gamma=0.8):
    img = np.clip(img, 0, None)
    v = np.percentile(img, pct)
    return np.clip(img / v, 0, 1) ** gamma


def to_rgb(img, cmap):
    return cmap(img)[..., :3]


# ----------------------------------------------------------------------------- compositing
def parse_aspect(a):
    """'16:9' / '16/9' / 1.777 -> width / height (float); None -> None."""
    if a is None:
        return None
    if isinstance(a, str):
        for sep in (":", "/", "x"):
            if sep in a:
                w, h = a.split(sep)
                return float(w) / float(h)
    return float(a)


def aspect_window(shape, aspect, region=None, ypos=0.5):
    """(y0, y1, x0, x1) of a window of `aspect` (width / height) in an image of `shape`.
    region=None: the largest such window (full width for a square image), at vertical
    position ypos (0 = top, 1 = bottom). region=(y0, x0, size): the window has the centre
    and the height of that square, widened to `aspect` and shifted to stay inside the image
    (narrowed in height if the image is not wide enough)."""
    H, W = shape
    if region is None:
        if W / H >= aspect:
            w = int(round(H * aspect))
            x0 = (W - w) // 2
            return 0, H, x0, x0 + w
        h = int(round(W / aspect))
        y0 = int(round(np.clip(ypos, 0, 1) * (H - h)))
        return y0, y0 + h, 0, W
    y0, x0, s = region
    cy, cx = y0 + s / 2, x0 + s / 2
    h = min(s, H)
    w = int(round(h * aspect))
    if w > W:
        w, h = W, int(round(W / aspect))
    y0 = int(round(np.clip(cy - h / 2, 0, H - h)))
    x0 = int(round(np.clip(cx - w / 2, 0, W - w)))
    return y0, y0 + h, x0, x0 + w


def diag_reach(shape):
    """Distance (px) from the centre to the upper-left corner along the normal of the
    frame diagonal (W / sqrt(2) for a square)."""
    H, W = shape
    return W * H / np.hypot(W, H)


def signed_distance(shape, offset):
    """Signed distance (px) from the '/' frame diagonal (lower-left to upper-right corner,
    45 degrees for a square), shifted by `offset` px along the normal.
    Negative = upper-left side, positive = lower-right side."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    return ((xx - cx) / W + (yy - cy) / H) * diag_reach(shape) - offset


def add_seam(rgb, d, color=(1, 1, 1), core=1.2, glow=10.0, glow_strength=0.35):
    a = np.exp(-(d / core) ** 2) + glow_strength * np.exp(-(d / glow) ** 2)
    a = np.clip(a, 0, 1)[..., None]
    return rgb * (1 - a) + np.asarray(color) * a


def compose(layers, d, edges):
    """layers: list of RGB images (n), edges: n-1 increasing thresholds on d."""
    out = layers[0].copy()
    for rgb, e in zip(layers[1:], edges):
        m = (d > e)[..., None]
        out = np.where(m, rgb, out)
    for e in edges:
        out = add_seam(out, d - e)
    return out


# ----------------------------------------------------------------------------- decoration
def _txt(ax, x, y, s, size, ha, va, alpha=1.0, color="white", weight="bold"):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, color=color, alpha=alpha,
            fontsize=size, family=FONT["family"], weight=weight,
            path_effects=[pe.withStroke(linewidth=3, foreground=(0, 0, 0, 0.6 * alpha))])


def frame_map(w_px, h_px):
    """Positions in the axes fraction of a SQUARE frame -> the same distance (in units of
    sqrt(w_px * h_px)) from the nearest edge of a w_px x h_px frame. Identity for a square.
    Returns (X, Y, kx, ky): coordinate maps and the scale of x / y lengths."""
    S = np.sqrt(w_px * h_px)
    kx, ky = S / w_px, S / h_px

    def X(x):
        return x * kx if x <= 0.5 else 1 - (1 - x) * kx

    def Y(y):
        return y * ky if y <= 0.5 else 1 - (1 - y) * ky
    return X, Y, kx, ky


def scalebar(ax, shape, pxsize_nm, um=5, pad=0.04, y_up=0.085, s=1.0, fmap=None):
    H, W = shape
    X, Y, _, ky = fmap or (lambda x: x, lambda y: y, 1, 1)
    L = um * 1000 / pxsize_nm / W                   # axes fraction
    x1 = X(1 - pad)
    y = Y(y_up)
    ax.plot([x1 - L, x1], [y, y], transform=ax.transAxes, color="white", lw=5 * s,
            solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=8 * s, foreground=(0, 0, 0, 0.5))])
    _txt(ax, x1 - L / 2, y + 0.012 * ky, f"{um} µm", 17 * s, "center", "bottom")


def render(rgb, labels, pxsize_nm, out_path=None, size_px=1600, title=None, scale_um=None,
           aspect=None):
    """labels: list of (x, y, text, ha, va, alpha) in axes coords of a square frame (moved
    to keep their pixel margins in a non-square frame). aspect: output width / height
    (exact, e.g. 16/9: 1920 x 1080 for size_px=1920); None = the image's own aspect."""
    H, W = rgb.shape[:2]
    if scale_um is None:   # pick a round bar length close to 20% of the field of view
        fov_um = W * pxsize_nm / 1000
        scale_um = min([1, 2, 5, 10, 20], key=lambda u: abs(u - 0.2 * fov_um))
    dpi = 100
    h_px = size_px / aspect if aspect else size_px * H / W
    fig = plt.figure(figsize=(size_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos", aspect="auto")
    ax.set_axis_off()
    s = np.sqrt(size_px * h_px) / 1600
    fmap = frame_map(size_px, h_px)
    X, Y = fmap[:2]
    for x, y, t, ha, va, a in labels:
        if a > 0.01:
            _txt(ax, X(x), Y(y), t, 22 * s, ha, va, alpha=a)
    if title:
        _txt(ax, X(0.96), Y(0.965), title, 15 * s, "right", "top", alpha=0.85, weight="normal")
    scalebar(ax, (H, W), pxsize_nm, um=scale_um, s=s, fmap=fmap)
    if out_path is None:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return np.array(Image.open(buf).convert("RGB"))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------- figures
def fig_split(data, which, cmap, out, label, size_px=1600, aspect=None):
    raw, rec = to_rgb(data["raw"], cmap), to_rgb(data[which], cmap)
    d = signed_distance(raw.shape[:2], 0)
    rgb = compose([raw, rec], d, [0])
    labels = [(0.035, 0.965, "Raw", "left", "top", 1),
              (0.965, 0.035, label, "right", "bottom", 1)]
    render(rgb, labels, data["pxsize_nm"], out, title=data["title"], size_px=size_px,
           aspect=aspect)


def fig_bands(data, cmap, out, size_px=1600, aspect=None):
    raw, tv, l1 = (to_rgb(data[k], cmap) for k in ("raw", "tv", "l1"))
    H, W = raw.shape[:2]
    d = signed_distance((H, W), 0)
    half = diag_reach((H, W)) / 3.2         # half-width of the middle band (W/(2√2)/1.6 if square)
    rgb = compose([raw, tv, l1], d, [-half, half])
    labels = [(0.035, 0.965, "Raw", "left", "top", 1),
              (0.035, 0.035, "Reconstruction · TV", "left", "bottom", 1),
              (0.965, 0.035, "Reconstruction · ℓ1", "right", "bottom", 1)]
    render(rgb, labels, data["pxsize_nm"], out, title=data["title"], size_px=size_px,
           aspect=aspect)


def anim_wipe(data, which, cmap, out_base, label, n_frames=150, size_px=1000, fps=30,
              aspect=None):
    raw, rec = to_rgb(data["raw"], cmap), to_rgb(data[which], cmap)
    H, W = raw.shape[:2]
    d0 = signed_distance((H, W), 0)
    reach = diag_reach((H, W))                 # corner-to-centre distance along normal
    t = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    offsets = 0.8 * reach * np.sin(t)          # smooth back-and-forth sweep
    frames = []
    for off in offsets:
        d = d0 - off
        rgb = compose([raw, rec], d, [0])
        # fade each label out as the seam approaches/covers its corner
        a_raw = np.clip((reach + off - 0.25 * reach) / (0.15 * reach), 0, 1)
        a_rec = np.clip((reach - off - 0.25 * reach) / (0.15 * reach), 0, 1)
        labels = [(0.035, 0.965, "Raw", "left", "top", a_raw),
                  (0.965, 0.035, label, "right", "bottom", a_rec)]
        frames.append(render(rgb, labels, data["pxsize_nm"], None, size_px=size_px,
                             title=data["title"], aspect=aspect))

    # MP4 (h264, web-friendly)
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out_base + ".mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
    p.stdin.close()
    p.wait()

    # GIF fallback (smaller)
    step = 3
    small = [Image.fromarray(f).resize((480, int(480 * f.shape[0] / f.shape[1])), Image.LANCZOS)
             .quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames[::step]]
    small[0].save(out_base + ".gif", save_all=True, append_images=small[1:],
                  duration=int(1000 * step / fps), loop=0, optimize=True)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=list(DATASETS))
    ap.add_argument("--cmap", default="magma", choices=list(CMAPS))
    ap.add_argument("--crop", nargs=3, type=int, metavar=("Y0", "X0", "SIZE"))
    ap.add_argument("--pct", type=float, default=99.8, help="saturation percentile")
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--trim", type=int, default=12, help="px trimmed from each border (edge artefacts)")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--out", default="publicity")
    ap.add_argument("--aspect", default=None,
                    help="output frame width:height, e.g. 16:9 (default: the square field)")
    ap.add_argument("--ypos", type=float, default=None,
                    help="vertical position of the --aspect window, 0 = top, 1 = bottom "
                         "(default: DATASETS[...]['ypos'], else 0.5)")
    ap.add_argument("--width", type=int, default=None,
                    help="PNG width in px (default 1600, or 1920 with --aspect)")
    args = ap.parse_args()

    aspect = parse_aspect(args.aspect)
    aspect_tag = args.aspect.replace(":", "x").replace("/", "x") if aspect else ""
    width = args.width or (1920 if aspect else 1600)
    os.makedirs(args.out, exist_ok=True)
    cmap = CMAPS[args.cmap]
    for key in args.data:
        data = load_dataset(key)
        b = args.trim
        if b:
            for k in ("raw", "tv", "l1"):
                data[k] = data[k][b:-b, b:-b]
        if aspect:
            ypos = args.ypos if args.ypos is not None else DATASETS[key].get("ypos", 0.5)
            y0, y1, x0, x1 = aspect_window(data["raw"].shape, aspect, args.crop, ypos)
            for k in ("raw", "tv", "l1"):
                data[k] = data[k][y0:y1, x0:x1]
            print(f"{key}: {aspect_tag} window y {y0}:{y1}, x {x0}:{x1} (after trim)")
        elif args.crop:
            y0, x0, s = args.crop
            for k in ("raw", "tv", "l1"):
                data[k] = data[k][y0:y0 + s, x0:x0 + s]
        for k in ("raw", "tv", "l1"):
            data[k] = normalise(data[k], args.pct, args.gamma)

        tag = f"{key}_{args.cmap}" + (f"_crop{args.crop[0]}-{args.crop[1]}-{args.crop[2]}"
                                      if args.crop else "")
        if aspect:
            tag += f"_{aspect_tag}" + ("" if args.crop or args.ypos is None
                                       else f"_y{args.ypos:g}")
        base = os.path.join(args.out, tag)
        fig_split(data, "tv", cmap, base + "_split_tv.png", "Reconstruction", width, aspect)
        fig_split(data, "l1", cmap, base + "_split_l1.png", "Reconstruction", width, aspect)
        fig_bands(data, cmap, base + "_bands.png", width, aspect)
        print("saved", base + "_{split_tv,split_l1,bands}.png")
        if not args.no_anim:
            anim_wipe(data, "tv", cmap, base + "_wipe_tv", "Reconstruction", aspect=aspect,
                      size_px=1280 if aspect else 1000)
            print("saved", base + "_wipe_tv.{mp4,gif}")


if __name__ == "__main__":
    main()
