"""
Scale-balanced denoising loss for images with both large low-frequency
structures (e.g. cell bodies) and fine high-frequency structures (e.g. filaments).

Composite = w_l1 * L1
          + w_l2 * L2
          + w_ffl * FocalFrequencyLoss      (counters CNN spectral bias)
          + w_cldice * soft-clDice           (optional; preserves thin-structure topology)

Design rationale
----------------
- L1 instead of L2/MSE: L1 does not over-penalise the large errors on big
  structures, so it stops the cell bodies from dominating the gradient the way
  MSE does. It is the standard fidelity term in restoration hybrids.
- Focal Frequency Loss (Jiang et al., ICCV 2021): reweights the loss in the 2D
  DFT so the network is pushed onto the frequency components it is currently
  getting wrong.
- soft-clDice (Shit et al., CVPR 2021): a differentiable centreline-Dice term
  that rewards preserving the connectivity of thin structures. It works on soft
  foreground masks; if none are passed, they are built automatically:
      true_fg = target > t                       (binary, from the clean image)
      pred_fg = sigmoid((pred - t) / temp)       (soft, carries the gradient)
  with t and temp relative to each image's maximum (see fg_threshold, fg_temp).

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
        patches = (
            x.unfold(2, ph, ph)
            .unfold(3, pw, pw)
            .contiguous()
            .view(n, c, -1, ph, pw)
        )
        freq = torch.fft.fft2(patches, norm="ortho")
        return torch.stack([freq.real, freq.imag], dim=-1)

    def _loss_formulation(
        self, pred_freq: torch.Tensor, target_freq: torch.Tensor
    ) -> torch.Tensor:
        if self.ave_spectrum:
            pred_freq = pred_freq.mean(0, keepdim=True)
            target_freq = target_freq.mean(0, keepdim=True)

        weight = (pred_freq - target_freq) ** 2
        weight = weight[..., 0] + weight[..., 1]
        weight = torch.sqrt(weight) ** self.alpha

        if self.log_matrix:
            weight = torch.log(weight + 1.0)

        if self.batch_matrix:
            weight = weight / (weight.max() + 1e-12)
        else:
            wmax = weight.amax(dim=(1, 2, 3, 4), keepdim=True)
            weight = weight / (wmax + 1e-12)

        weight = torch.nan_to_num(weight, nan=0.0).clamp(0.0, 1.0).detach()

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
    p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Differentiable morphological skeleton (Shit et al., CVPR 2021).

    `img`: soft foreground probability in [0, 1], shape (N, 1, H, W).
    `iters` should be >= the half-thickness (in px) of the thickest structure.
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
    """soft-clDice loss for one foreground channel. Returns 1 - clDice."""

    def __init__(self, iters: int = 10, smooth: float = 1.0):
        super().__init__()
        self.iters = iters
        self.smooth = smooth

    def forward(self, pred_fg: torch.Tensor, true_fg: torch.Tensor) -> torch.Tensor:
        skel_pred = soft_skeletonize(pred_fg, self.iters)
        skel_true = soft_skeletonize(true_fg, self.iters)

        tprec = (
            (skel_pred * true_fg).sum(dim=(1, 2, 3)) + self.smooth
        ) / (skel_pred.sum(dim=(1, 2, 3)) + self.smooth)
        tsens = (
            (skel_true * pred_fg).sum(dim=(1, 2, 3)) + self.smooth
        ) / (skel_true.sum(dim=(1, 2, 3)) + self.smooth)

        cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - cl_dice).mean()


# ---------------------------------------------------------------------------
# 3. Foreground masks for clDice
# ---------------------------------------------------------------------------
def make_fg_masks(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.2,
    temp: float = 0.05,
    relative: bool = True,
):
    """Build (pred_fg, true_fg) for soft-clDice.

    true_fg: binary mask of the clean image, target > t.
    pred_fg: soft mask of the output, sigmoid((pred - t) / temp); differentiable.

    With relative=True, t = threshold * max(target) and temp = temp * max(target)
    per image, so the masks don't depend on each image's brightness. pred_fg uses
    the threshold computed on the TARGET, so a dimmer output is penalised too.
    """
    if relative:
        scale = target.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    else:
        scale = torch.ones_like(target[:, :1, :1, :1])
    t = threshold * scale
    tau = temp * scale

    true_fg = (target > t).float()
    pred_fg = torch.sigmoid((pred - t) / tau)
    return pred_fg, true_fg


# ---------------------------------------------------------------------------
# 4. Composite loss
# ---------------------------------------------------------------------------
class ScaleBalancedDenoisingLoss(nn.Module):
    """L1 + L2 + Focal Frequency (+ optional soft-clDice).

    Args:
        w_l1, w_l2, w_ffl, w_cldice: weights of each term (0 disables it).
            w_cldice can be changed during training (e.g. warm-up): the clDice
            module is created whenever w_cldice > 0 at construction OR
            use_cldice=True.
        ffl_kwargs:   passed to FocalFrequencyLoss.
        cldice_iters: skeleton iterations (>= half-thickness of thickest structure, px).
        fg_threshold: foreground threshold, fraction of each target's max.
        fg_temp:      sigmoid temperature for pred_fg, fraction of each target's max.
                      Smaller = sharper mask, but gradient only near the threshold.
        use_cldice:   create the clDice module even if w_cldice starts at 0.

    forward(pred, target, pred_fg=None, true_fg=None) -> (total, parts, weights)
        If the masks are not given, they are built with make_fg_masks.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_ffl: float = 1.0,
        w_l2: float = 0.0,
        w_cldice: float = 0.0,
        ffl_kwargs: dict | None = None,
        cldice_iters: int = 10,
        fg_threshold: float = 0.2,
        fg_temp: float = 0.05,
        use_cldice: bool = False,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_ffl = w_ffl
        self.w_l2 = w_l2
        self.w_cldice = w_cldice
        self.fg_threshold = fg_threshold
        self.fg_temp = fg_temp
        self.ffl = FocalFrequencyLoss(**(ffl_kwargs or {}))
        self.cldice = (
            SoftClDiceLoss(iters=cldice_iters)
            if (w_cldice > 0 or use_cldice) else None
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_fg: torch.Tensor | None = None,
        true_fg: torch.Tensor | None = None,
    ):
        zero = pred.new_zeros(())          # on the same device/dtype as pred

        l1  = F.l1_loss(pred, target)  if self.w_l1  > 0.0 else zero
        ffl = self.ffl(pred, target)   if self.w_ffl > 0.0 else zero
        l2  = F.mse_loss(pred, target) if self.w_l2  > 0.0 else zero

        total = self.w_l1 * l1 + self.w_ffl * ffl + self.w_l2 * l2

        weights = {"l1": self.w_l1, "ffl": self.w_ffl, "l2": self.w_l2,
                   "cldice": self.w_cldice}
        parts = {"l1": l1.detach(), "ffl": ffl.detach(), "l2": l2.detach(),
                 "cldice": zero.detach()}

        if self.w_cldice > 0.0:
            if self.cldice is None:
                raise ValueError("w_cldice > 0 but the clDice module was not "
                                 "created: pass use_cldice=True at construction")
            if pred_fg is None or true_fg is None:
                pred_fg, true_fg = make_fg_masks(
                    pred, target, self.fg_threshold, self.fg_temp)
            cld = self.cldice(pred_fg, true_fg)
            total = total + self.w_cldice * cld
            parts["cldice"] = cld.detach()

        parts["total"] = total.detach()
        return total, parts, weights


# ---------------------------------------------------------------------------
# Minimal usage sketch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pred = torch.rand(2, 1, 128, 128, requires_grad=True)
    target = torch.rand(2, 1, 128, 128)

    # Intensity-only
    criterion = ScaleBalancedDenoisingLoss(w_l1=1.0, w_ffl=1.0, w_cldice=0.0)
    loss, parts, weights = criterion(pred, target)
    loss.backward()
    print("intensity-only:", {k: round(v.item(), 4) for k, v in parts.items()})

    # With clDice, masks built automatically
    pred.grad = None
    criterion2 = ScaleBalancedDenoisingLoss(w_l1=1.0, w_ffl=1.0, w_cldice=0.1,
                                            cldice_iters=10)
    loss2, parts2, weights2 = criterion2(pred, target)
    loss2.backward()
    print("with cldice:", {k: round(v.item(), 4) for k, v in parts2.items()})