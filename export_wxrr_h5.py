"""
Esporta il dataset di phantom (.pth, lo stesso usato per DRUNet) nei file .h5
letti dal training del weakly convex ridge regularizer (WCRR).

Lancia da ism_regularisation_learned/:
    python export_wcrr_h5.py

Output:
    training_wcrr/data/train.h5        patch 40x40 augmentate, chiavi "0","1",...
    training_wcrr/data/validation.h5   immagini intere, chiavi "0","1",...
Ogni elemento: float32, shape (1, H, W), valori in [0, 1].
"""

import os
import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Parametri
# ---------------------------------------------------------------------------
SRC      = "training_drunet/training_dataset/varied_dataset.pth"   # stesso .pth di DRUNet
OUT_DIR  = "training_wcrr/data"

N_TRAIN  = 1000     # stesso split di DRUNet: i primi 1000 sono training
N_VAL    = 50       # sottoinsieme della validazione DRUNet (WCRR valida lentamente)

PATCH    = 40       # come il preprocessing BSD originale
STRIDE   = 20       # 10 = come l'originale, ma ~4x patch e file molto piu' grande
FG_THR   = 0.05     # pixel "di struttura" sopra questa soglia
MIN_FG   = 0.02     # tieni la patch se >= 2% dei pixel e' struttura...
KEEP_BG  = 0.1      # ...oppure con probabilita' 10% (esempi di solo fondo)
SEED     = 0

# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

data = torch.load(SRC).float()
assert data.dim() == 4 and data.shape[1] == 1, f"shape inattesa: {tuple(data.shape)}"
print(f"dataset: {tuple(data.shape)}, range [{data.min():.4f}, {data.max():.4f}]")
if data.min() < 0 or data.max() > 1.0 + 1e-6:
    raise ValueError("I dati devono stare in [0,1] (stessa scala usata per DRUNet).")

train = data[:N_TRAIN].numpy()
val   = data[N_TRAIN:N_TRAIN + N_VAL].numpy()


def augment(p):
    """Una delle 8 trasformazioni diedrali, scelta a caso."""
    p = np.rot90(p, k=rng.integers(4), axes=(-2, -1))
    if rng.random() < 0.5:
        p = np.flip(p, axis=-1)
    return np.ascontiguousarray(p)


def keep(p):
    return (p > FG_THR).mean() >= MIN_FG or rng.random() < KEEP_BG


# ---- training: patch ----
train_path = os.path.join(OUT_DIR, "train.h5")
n = 0
with h5py.File(train_path, "w") as f:
    for img in train:                                  # img: (1, H, W)
        H, W = img.shape[-2:]
        for i in range(0, H - PATCH + 1, STRIDE):
            for j in range(0, W - PATCH + 1, STRIDE):
                p = img[:, i:i + PATCH, j:j + PATCH]
                if keep(p):
                    f.create_dataset(str(n), data=augment(p).astype(np.float32))
                    n += 1
print(f"train patches: {n}  -> {os.path.abspath(train_path)}")

# ---- validazione: immagini intere ----
val_path = os.path.join(OUT_DIR, "validation.h5")
with h5py.File(val_path, "w") as f:
    for k, img in enumerate(val):
        f.create_dataset(str(k), data=img.astype(np.float32))
print(f"validation images: {len(val)}  -> {os.path.abspath(val_path)}")

# ---- controllo: rileggi un elemento per file ----
for path in (train_path, val_path):
    with h5py.File(path, "r") as f:
        k = list(f.keys())[0]
        x = f[k][()]
        print(f"{os.path.basename(path)}: {len(f.keys())} elementi, "
              f"es. '{k}' shape {x.shape} {x.dtype} range [{x.min():.3f}, {x.max():.3f}]")