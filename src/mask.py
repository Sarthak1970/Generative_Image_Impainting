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
