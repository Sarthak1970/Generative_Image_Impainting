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


def _region_error(prediction: torch.Tensor, target: torch.Tensor, region: torch.Tensor) -> Dict[str, Optional[float]]:
    """MSE/MAE/PSNR over a [B,1,H,W] region, in [0, 1] intensity units."""
    pred = (prediction.detach().clamp(-1, 1) + 1.0) / 2.0
    gt = (target.detach().clamp(-1, 1) + 1.0) / 2.0
    region = region.to(device=pred.device, dtype=pred.dtype).expand_as(pred)
    denom = float(region.sum().item())
    if denom == 0.0:
        return {"mse": None, "mae": None, "psnr_db": None}
    mse = float((((pred - gt) ** 2) * region).sum().item() / denom)
    mae = float(((pred - gt).abs() * region).sum().item() / denom)
    return {"mse": mse, "mae": mae, "psnr_db": min(120.0, 10.0 * math.log10(1.0 / max(mse, 1e-12)))}


def compute_inpainting_metrics(
    raw_prediction: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    boundary_width: int = 8,
    compute_ssim: bool = True,
    lpips_fn=None,
) -> Dict[str, Optional[float]]:
    """Metrics that keep the raw sampler output and the composite separate.

    * ``raw_*``: the sampler output exactly as returned. ``raw_known_*``
      measures how well the raw sampler itself preserved the known pixels.
    * ``composite_*``: the raw output with known pixels restored from
      ``target``; this is what is normally shown and is used for full-image
      PSNR/SSIM/LPIPS. Missing-region errors are identical for both versions.
    * ``boundary_inner_*``: missing pixels within ``boundary_width`` px of the
      seam (identical for raw and composite).
    * ``lpips_composite``: only if a callable ``lpips_fn(pred, gt)`` taking
      [-1, 1] tensors is supplied (for example ``lpips.LPIPS(net="alex")``);
      otherwise ``None``. LPIPS is optional and not a project dependency.

    Intensity metrics use [0, 1] units, as ``compute_metrics`` does.
    """
    from src.mask import create_boundary_band

    mask = missing_mask.to(device=target.device, dtype=target.dtype)
    if mask.shape[0] == 1 and target.shape[0] > 1:
        mask = mask.expand(target.shape[0], -1, -1, -1)
    composite = (1.0 - mask) * target + mask * raw_prediction
    raw = compute_metrics(raw_prediction, target, mask, compute_ssim=compute_ssim)
    comp = compute_metrics(composite, target, mask, compute_ssim=compute_ssim)
    band = create_boundary_band(mask, boundary_width, side="inner")
    inner = _region_error(composite, target, band)

    result: Dict[str, Optional[float]] = {
        "missing_fraction": float(mask.mean().item()),
        "composite_full_psnr_db": comp["full_psnr_db"],
        "composite_full_ssim": comp["full_ssim"],
        "composite_full_mse": comp["full_mse"],
        "missing_mse": comp["missing_mse"],
        "missing_mae": comp["missing_mae"],
        "missing_psnr_db": comp["missing_psnr_db"],
        "boundary_width_px": float(boundary_width),
        "boundary_inner_mse": inner["mse"],
        "boundary_inner_mae": inner["mae"],
        "boundary_inner_psnr_db": inner["psnr_db"],
        "raw_full_psnr_db": raw["full_psnr_db"],
        "raw_full_ssim": raw["full_ssim"],
        "raw_known_mse": raw["known_mse"],
        "raw_known_mae": raw["known_mae"],
        "raw_known_psnr_db": raw["known_psnr_db"],
        "raw_known_max_abs_error": None,
        "lpips_composite": None,
    }
    known = (1.0 - mask).expand_as(raw_prediction)
    if float(known.sum().item()) > 0:
        # [-1, 1] units, matching run_local.py's raw_known_region_final_max_abs_error.
        result["raw_known_max_abs_error"] = float(((raw_prediction - target).abs() * known).max().item())
    if lpips_fn is not None:
        with torch.no_grad():
            result["lpips_composite"] = float(lpips_fn(composite.clamp(-1, 1), target.clamp(-1, 1)).mean().item())
    return result
