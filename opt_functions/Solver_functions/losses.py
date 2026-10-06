"""
Scale-balanced denoising loss for images with both large low-frequency
structures (e.g. cell bodies) and fine high-frequency structures (e.g. filaments).

Composite = w_l1 * L1
          + w_ffl * FocalFrequencyLoss      (counters CNN spectral bias)
          + w_cldice * soft-clDice           (optional; preserves thin-structure topology)

Design rationale
----------------
- L1 instead of L2/MSE: L1 does not over-penalise the large errors on big
  structures, so it stops the cell bodies from dominating the gradient the way
  MSE does. It is the standard fidelity term in restoration hybrids.
- Focal Frequency Loss (Jiang et al., ICCV 2021): reweights the loss in the 2D
  DFT so the network is pushed onto the frequency components it is currently
  getting wrong. Because the filaments are exactly the high-frequency content the
  network neglects (spectral bias), FFL up-weights them automatically, with no
  hand-tuned band boundaries.
- soft-clDice (Shit et al., CVPR 2021): a differentiable centreline-Dice term.
  Only relevant if you care about *connectivity* of the filaments, not just their
  intensity. It runs on a soft foreground mask, so you either threshold the
  denoised output softly or (better, since your data is simulated) use the
  ground-truth filament mask. Drop it if you only care about intensity fidelity.

Author note: written for torch >= 1.8. FFT uses torch.fft (complex dtype).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Focal Frequency Loss
# ---------------------------------------------------------------------------
class FocalFrequencyLoss(nn.Module):
    """Focal Frequency Loss (Jiang et al., ICCV 2021), self-contained.

    Computes the 2D DFT of pred and target, measures the squared distance
    between the two complex spectra per frequency, then weights each frequency
    by how badly it is currently reconstructed (the "focal" / hard-frequency
    weighting) raised to power `alpha`.

    Args:
        loss_weight: scalar multiplier applied to the returned loss.
        alpha:       sharpness of the focal weighting (1.0 in the paper).
        patch_factor: split the image into patch_factor x patch_factor tiles and
                      take the DFT of each tile (like a blockwise DCT). 1 = whole image.
        ave_spectrum: average the spectrum over the batch before weighting.
        log_matrix:   compress the weight matrix with log(1+w) for stability.
        batch_matrix: normalise the weight matrix over the whole batch rather
                      than per-sample.

    Input shape: (N, C, H, W). H and W must be divisible by patch_factor.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        pf = self.patch_factor
        n, c, h, w = x.shape
        assert h % pf == 0 and w % pf == 0, (
            f"image size ({h}x{w}) must be divisible by patch_factor {pf}"
        )
        ph, pw = h // pf, w // pf

        # split into pf*pf patches -> (N, C, pf*pf, ph, pw)
        patches = (
            x.unfold(2, ph, ph)
            .unfold(3, pw, pw)
            .contiguous()
            .view(n, c, -1, ph, pw)
        )

        # 2D DFT of each patch; returns complex tensor (..., ph, pw)
        freq = torch.fft.fft2(patches, norm="ortho")
        # stack real/imag on a trailing axis -> (N, C, P, ph, pw, 2)
        return torch.stack([freq.real, freq.imag], dim=-1)

    def _loss_formulation(
        self, pred_freq: torch.Tensor, target_freq: torch.Tensor
    ) -> torch.Tensor:
        # weight matrix = distance between spectra, detached (no grad through it)
        if self.ave_spectrum:
            pred_freq = pred_freq.mean(0, keepdim=True)
            target_freq = target_freq.mean(0, keepdim=True)

        weight = (pred_freq - target_freq) ** 2
        weight = weight[..., 0] + weight[..., 1]  # |Δspectrum|^2
        weight = torch.sqrt(weight) ** self.alpha

        if self.log_matrix:
            weight = torch.log(weight + 1.0)

        # normalise weights into [0, 1]
        if self.batch_matrix:
            weight = weight / (weight.max() + 1e-12)
        else:
            # per-sample max over (C, P, H, W)
            wmax = weight.amax(dim=(1, 2, 3, 4), keepdim=True)
            weight = weight / (wmax + 1e-12)

        weight = torch.nan_to_num(weight, nan=0.0).clamp(0.0, 1.0).detach()

        # frequency distance (this one carries the gradient)
        freq_dist = (pred_freq - target_freq) ** 2
        freq_dist = freq_dist[..., 0] + freq_dist[..., 1]

        return (weight * freq_dist).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_freq = self._tensor2freq(pred)
        target_freq = self._tensor2freq(target)
        return self._loss_formulation(pred_freq, target_freq) * self.loss_weight


# ---------------------------------------------------------------------------
# 2. soft-clDice (topology / connectivity term for thin structures)
# ---------------------------------------------------------------------------
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    # min-pooling via -maxpool of the negative, in both axial directions
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton (Shit et al., CVPR 2021).

    `img` should be a soft foreground probability in [0, 1], shape (N, 1, H, W).
    `iters` should be >= the half-thickness (in px) of your thickest filament.
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class SoftClDiceLoss(nn.Module):
    """soft-clDice loss for one foreground channel.

    Operates on probabilities in [0, 1]. Returns 1 - clDice so it can be summed
    with the other terms.
    """

    def __init__(self, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred_fg: torch.Tensor, true_fg: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeletonize(pred_fg, self.iters)
        skel_true = soft_skeletonize(true_fg, self.iters)

        # topology precision / sensitivity
        tprec = (
            (skel_pred * true_fg).sum(dim=(1, 2, 3)) + self.smooth
        ) / (skel_pred.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = (
            (skel_true * pred_fg).sum(dim=(1, 2, 3)) + self.smooth
        ) / (skel_true.sum(dim=(1, 2, 3)) + self.smooth)

        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - cl_dice).mean()


# ---------------------------------------------------------------------------
# 3. Composite loss
# ---------------------------------------------------------------------------
class ScaleBalancedDenoisingLoss(nn.Module):
    """L1 + Focal Frequency (+ optional soft-clDice) for scale-balanced denoising.

    Args:
        w_l1:     weight on the L1 fidelity term.
        w_ffl:    weight on the focal frequency term. Start ~1.0 and tune; FFL
                  and L1 are on different scales so expect to sweep this.
        w_cldice: weight on soft-clDice. 0.0 disables it (and skips the cost).
        ffl_kwargs:   passed to FocalFrequencyLoss.
        cldice_iters: skeleton iterations (>= thickest filament half-width).

    forward(pred, target, pred_fg=None, true_fg=None)
        pred, target : (N, C, H, W) denoised output and clean target.
        pred_fg, true_fg : (N, 1, H, W) soft foreground masks in [0, 1], only
            needed when w_cldice > 0. Because your data is simulated, pass the
            ground-truth filament mask as true_fg; for pred_fg either threshold
            the output softly (e.g. sigmoid((pred - t)/temp)) or add a small
            segmentation head to the network.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_ffl: float = 1.0,
        w_l2: float = 0.0,
        w_cldice: float = 0.0,
        ffl_kwargs: dict | None = None,
        cldice_iters: int = 10,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ffl = w_ffl
        self.w_l2 = w_l2
        self.w_cldice = w_cldice
        self.ffl = FocalFrequencyLoss(**(ffl_kwargs or {}))
        self.cldice = SoftClDiceLoss(iters=cldice_iters) if w_cldice > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_fg: torch.Tensor | None = None,
        true_fg: torch.Tensor | None = None,
    ):
        if self.w_l1 > 0.0:
            l1 = F.l1_loss(pred, target)
        else: 
            l1 = torch.tensor(0.0)
            
        if self.w_ffl > 0.0:
            ffl = self.ffl(pred, target)
        else: 
            ffl = torch.tensor(0.0)
        
        if self.w_l2 > 0.0:
            l2 = F.mse_loss(pred, target)
        else: 
            l2 = torch.tensor(0.0)
            
            
        total = self.w_l1 * l1 + self.w_ffl * ffl + self.w_l2* l2
        weights = {'l1':self.w_l1, 'ffl': self.w_ffl, 'l2':self.w_l2}
        parts = {"l1": l1.detach() , "ffl": ffl.detach(),"l2": l2.detach() }

        if self.w_cldice > 0:
            if pred_fg is None or true_fg is None:
                raise ValueError("w_cldice > 0 requires pred_fg and true_fg masks")
            cld = self.cldice(pred_fg, true_fg)
            total = total + self.w_cldice * cld
            parts["cldice"] = cld.detach()

        parts["total"] = total.detach()
        return total, parts, weights

# ---------------------------------------------------------------------------
# Minimal usage sketch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Intensity-only version (no topology term):
    criterion = ScaleBalancedDenoisingLoss(w_l1=1.0, w_ffl=1.0, w_cldice=0.0)

    pred = torch.rand(2, 1, 128, 128, requires_grad=True)
    target = torch.rand(2, 1, 128, 128)
    loss, parts = criterion(pred, target)
    loss.backward()
    print("intensity-only:", {k: round(v.item(), 4) for k, v in parts.items()})

    # With connectivity term (simulated data -> use GT filament mask as true_fg):
    criterion2 = ScaleBalancedDenoisingLoss(
        w_l1=1.0, w_ffl=1.0, w_cldice=0.5, cldice_iters=10
    )
    pred_fg = torch.sigmoid((pred.detach() - 0.5) * 10).requires_grad_(True)
    true_fg = (target > 0.5).float()
    loss2, parts2 = criterion2(pred, target, pred_fg=pred_fg, true_fg=true_fg)
    loss2.backward()
    print("with cldice:", {k: round(v.item(), 4) for k, v in parts2.items()})