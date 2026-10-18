"""Full-image and mask-region metrics for image-inpainting evaluation."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch


def compute_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    compute_ssim: bool = True,
) -> Dict[str, Optional[float]]:
    """Compute metrics on tensors normalized to [-1, 1].

    Returns full-image MSE/PSNR/SSIM plus missing- and known-region MSE/MAE/PSNR.
    The missing mask convention is 1 = missing, 0 = known.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must have equal [B,C,H,W] shapes")
    if missing_mask.ndim != 4 or missing_mask.shape[1] != 1:
        raise ValueError("missing_mask must have shape [1 or B,1,H,W]")
    if missing_mask.shape[-2:] != target.shape[-2:]:
        raise ValueError("missing_mask spatial size must match the images")

    pred = (prediction.detach().clamp(-1, 1) + 1.0) / 2.0
    gt = (target.detach().clamp(-1, 1) + 1.0) / 2.0
    mask = missing_mask.to(device=pred.device, dtype=pred.dtype)
    mask = mask.expand(pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3])
    known = 1.0 - mask
    squared = (pred - gt).square()
    absolute = (pred - gt).abs()

    def region_mean(values: torch.Tensor, region: torch.Tensor):
        denom = region.sum()
        if float(denom.item()) == 0.0:
            return None
        return float(((values * region).sum() / denom).item())

    full_mse = float(squared.mean().item())
    missing_mse = region_mean(squared, mask)
    known_mse = region_mean(squared, known)
    missing_mae = region_mean(absolute, mask)
    known_mae = region_mean(absolute, known)

    def psnr(mse):
        if mse is None:
            return None
        # Cap exact/near-exact values so JSON remains standards-compliant.
        return min(120.0, 10.0 * math.log10(1.0 / max(mse, 1e-12)))

    full_ssim: Optional[float] = None
    if compute_ssim:
        try:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
            metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
            with torch.no_grad():
                full_ssim = float(metric(pred, gt).item())
        except (ImportError, ModuleNotFoundError):
            # Metrics other than SSIM remain usable if torchmetrics was not installed.
            full_ssim = None

    return {
        "full_mse": full_mse,
        "full_psnr_db": psnr(full_mse),
        "full_ssim": full_ssim,
        "missing_mse": missing_mse,
        "missing_mae": missing_mae,
        "missing_psnr_db": psnr(missing_mse),
        "known_mse": known_mse,
        "known_mae": known_mae,
        "known_psnr_db": psnr(known_mse),
    }
