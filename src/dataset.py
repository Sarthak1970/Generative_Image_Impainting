"""Image and mask loading utilities for the RePaint benchmark."""
import torch
import torchvision.transforms as T
from PIL import Image


def get_transforms(image_size: int = 256):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),  # [0, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1]
    ])


def load_image(path: str, image_size: int = 256) -> torch.Tensor:
    """Load RGB image as [1, 3, H, W] float tensor in [-1, 1]."""
    with Image.open(path) as image:
        img = image.convert("RGB")
    return get_transforms(image_size)(img).unsqueeze(0)


def load_mask(path: str, image_size: int = 256, white_is_missing: bool = True) -> torch.Tensor:
    """Load mask as [1, 1, H, W], with 1 = missing and 0 = known.

    By default, white means missing to match this project's generated masks.
    Official RePaint keep-masks instead use white=known and black=missing, so
    pass ``white_is_missing=False`` when loading those masks. Nearest-neighbor
    resizing preserves hard binary boundaries.
    """
    with Image.open(path) as image:
        mask_image = image.convert("L")
    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])
    mask_tensor = transform(mask_image).unsqueeze(0)
    binary_mask = (mask_tensor > 0.5).float()
    return binary_mask if white_is_missing else 1.0 - binary_mask


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] or [3, H, W] tensor in [-1, 1] to RGB PIL."""
    tensor = tensor.detach().clamp(-1, 1)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("tensor_to_pil expects a batch size of 1")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W] or [1,3,H,W], got {tuple(tensor.shape)}")
    array = ((tensor.cpu().permute(1, 2, 0) + 1.0) * 127.5).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")
