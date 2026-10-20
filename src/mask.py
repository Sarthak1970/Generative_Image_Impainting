"""Mask generators. Convention: 1.0 = missing/repaint, 0.0 = known."""
import torch


def create_center_mask(image_size: int = 256, mask_ratio: float = 0.35) -> torch.Tensor:
    """Create a centered square hole whose side is ``mask_ratio * image_size``.

    Note: a square side ratio of 0.35 masks about 12.25% of the image area.
    Returns [1, 1, H, W].
    """
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    box_size = max(1, int(image_size * mask_ratio))  # preserve the repository's floor/truncation behavior
    start = (image_size - box_size) // 2
    end = start + box_size
    mask[:, :, start:end, start:end] = 1.0
    return mask


def create_half_mask(image_size: int = 256, mode: str = "bottom") -> torch.Tensor:
    """Create a half-image mask for one of ``left/right/top/bottom``."""
    if image_size < 2:
        raise ValueError("image_size must be >= 2")
    mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32)
    half = image_size // 2
    if mode == "bottom":
        mask[:, :, half:, :] = 1.0
    elif mode == "top":
        mask[:, :, :half, :] = 1.0
    elif mode == "left":
        mask[:, :, :, :half] = 1.0
    elif mode == "right":
        mask[:, :, :, half:] = 1.0
    else:
        raise ValueError(f"Unknown half-mask mode: {mode!r}")
    return mask


def create_boundary_band(mask: torch.Tensor, width: int = 8, side: str = "both") -> torch.Tensor:
    """Return a binary band of pixels within ``width`` px of the hole boundary.

    ``mask`` is [B, 1, H, W] with 1 = missing. Distances use the Chebyshev
    (square) neighbourhood implemented by max-pooling, so no extra packages
    are needed. ``side`` selects:

    * ``"inner"``: missing pixels with a known pixel within ``width`` px;
    * ``"outer"``: known pixels with a missing pixel within ``width`` px;
    * ``"both"``: the union of the two (a ring straddling the seam).

    An all-known or all-missing mask has no boundary and yields all zeros.
    Pixels beyond the image border are not treated as known or missing.
    """
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W]")
    if int(width) != width or width < 1:
        raise ValueError("width must be a positive integer")
    if side not in ("both", "inner", "outer"):
        raise ValueError(f"Unknown boundary side: {side!r}")
    missing = (mask > 0.5).to(torch.float32)
    known = 1.0 - missing
    kernel = 2 * int(width) + 1
    near_missing = torch.nn.functional.max_pool2d(missing, kernel, stride=1, padding=int(width))
    near_known = torch.nn.functional.max_pool2d(known, kernel, stride=1, padding=int(width))
    inner = missing * near_known
    outer = known * near_missing
    if side == "inner":
        band = inner
    elif side == "outer":
        band = outer
    else:
        band = torch.clamp(inner + outer, max=1.0)
    return band.to(dtype=mask.dtype if torch.is_floating_point(mask) else torch.float32)
