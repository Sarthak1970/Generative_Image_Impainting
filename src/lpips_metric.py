"""LPIPS for the study, with explicit failure reporting.

Implementation: the ``lpips`` package (Zhang et al., CVPR 2018, "The
Unreasonable Effectiveness of Deep Features as a Perceptual Metric"; the
official PerceptualSimilarity code, https://github.com/richzhang/PerceptualSimilarity),
version 0.1 linear calibration weights, AlexNet backbone by default (the
backbone used by the RePaint paper's evaluation). Inputs are [-1, 1] RGB
tensors, which is the package's native range. Lower = more perceptually
similar. LPIPS is a learned perceptual distance; it is not a measure of
semantic correctness or of human preference.

Two values are produced per image pair (both on the composite image, i.e. the
sampler output with known pixels restored, against the ground truth):

* ``lpips_full``: standard LPIPS on the full image (the package's spatial
  average over every layer).
* ``lpips_missing_masked``: the package's *spatial* LPIPS map (each layer's
  distance map bilinearly upsampled to the input resolution and summed over
  layers; ``LPIPS(spatial=True)``) averaged over the missing pixels only:
  sum(M * map) / sum(M). Nothing is cropped. Because deep features have large
  receptive fields, the map near the hole is influenced by context outside it
  and vice versa, so this is a hole-weighted perceptual distance, not an
  isolated LPIPS of the hole content. ``None`` if the mask is empty.

Dependencies: ``pip install lpips`` (needs scipy). The linear weights ship
with the package; the AlexNet backbone weights are fetched by torchvision on
first use (``alexnet-owt-7be5be79.pth`` from download.pytorch.org into the
torch hub cache, or a pre-populated ``$TORCH_HOME/hub/checkpoints``). If any
of this fails, :func:`load_lpips` raises :class:`LPIPSUnavailableError` with
the reason; callers must report it rather than silently dropping LPIPS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


class LPIPSUnavailableError(RuntimeError):
    """LPIPS could not be initialised (package missing, weights unavailable, ...)."""


@dataclass
class LPIPSEvaluator:
    full_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    spatial_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    info: dict = field(default_factory=dict)

    @torch.no_grad()
    def full(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        value = self.full_fn(prediction.clamp(-1, 1), target.clamp(-1, 1))
        return float(value.detach().float().mean().item())

    @torch.no_grad()
    def masked(self, prediction: torch.Tensor, target: torch.Tensor, missing_mask: torch.Tensor) -> Optional[float]:
        if self.spatial_fn is None:
            return None
        mask = missing_mask.to(device=prediction.device, dtype=torch.float32)
        if float(mask.sum().item()) == 0.0:
            return None
        dmap = self.spatial_fn(prediction.clamp(-1, 1), target.clamp(-1, 1)).detach().float()
        if dmap.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"Spatial LPIPS map {tuple(dmap.shape)} does not match mask {tuple(mask.shape)}")
        mask = mask.expand(dmap.shape[0], 1, -1, -1)
        return float((dmap * mask).sum().item() / mask.sum().item())

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Callable form accepted by ``compute_inpainting_metrics(lpips_fn=...)``."""
        with torch.no_grad():
            return self.full_fn(prediction.clamp(-1, 1), target.clamp(-1, 1))


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def load_lpips(net: str = "alex", device="cpu", random_backbone: bool = False) -> LPIPSEvaluator:
    """Initialise LPIPS or raise :class:`LPIPSUnavailableError` explaining why.

    ``random_backbone=True`` (``pnet_rand``) is for software tests only: it skips
    the ImageNet backbone download and gives values that are NOT valid LPIPS.
    """
    try:
        import lpips  # noqa: WPS433 (optional dependency)
    except Exception as exc:  # ImportError, or scipy missing inside lpips
        raise LPIPSUnavailableError(
            f"Could not import the 'lpips' package ({type(exc).__name__}: {exc}). "
            "Install it explicitly, e.g. `pip install lpips` (requires scipy); see requirements-study.txt."
        ) from exc
    try:
        full = lpips.LPIPS(net=net, version="0.1", spatial=False, pnet_rand=random_backbone, verbose=False)
        spatial = lpips.LPIPS(net=net, version="0.1", spatial=True, pnet_rand=random_backbone, verbose=False)
        spatial.load_state_dict(full.state_dict())  # identical weights in both modes
    except Exception as exc:
        raise LPIPSUnavailableError(
            f"lpips.LPIPS(net={net!r}) failed to initialise ({type(exc).__name__}: {exc}). "
            "The backbone weights are downloaded by torchvision on first use; without internet, place "
            "the torchvision checkpoint in $TORCH_HOME/hub/checkpoints/."
        ) from exc
    device = torch.device(device)
    full = full.eval().requires_grad_(False).to(device)
    spatial = spatial.eval().requires_grad_(False).to(device)
    info = {
        "lpips_implementation": "lpips (PerceptualSimilarity)",
        "lpips_package_version": _package_version("lpips"),
        "lpips_net": net,
        "lpips_version": "0.1",
        "lpips_input_range": "[-1, 1]",
        "lpips_backbone": "random (TEST ONLY, values invalid)" if random_backbone else "torchvision ImageNet weights",
        "lpips_valid": not random_backbone,
    }
    return LPIPSEvaluator(full_fn=full, spatial_fn=spatial, info=info)


def lpips_status(evaluator: Optional[LPIPSEvaluator], error: Optional[BaseException] = None) -> dict:
    """Metadata describing whether LPIPS values are present and valid."""
    if evaluator is not None:
        status = "ok" if evaluator.info.get("lpips_valid", False) else "invalid_test_backbone"
        return dict(evaluator.info, lpips_status=status, lpips_error=None)
    return {"lpips_status": "unavailable", "lpips_valid": False,
            "lpips_error": None if error is None else f"{type(error).__name__}: {error}"}
