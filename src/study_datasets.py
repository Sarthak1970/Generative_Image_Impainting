"""Configurable image loaders for the faces / Places2 study.

* ``project_resize``: the existing project loader ``src.dataset.load_image``
  (bilinear resize to S x S; it stretches non-square CelebA images). Used for
  faces so that results stay continuous with the existing baseline.
* ``center_crop``: the guided-diffusion / official RePaint preprocessing
  (``center_crop_arr`` in guided_diffusion/image_datasets.py): halve with a BOX
  filter while the short side is >= 2S, resize the short side to S with
  BICUBIC, then take the central S x S crop. No aspect-ratio distortion.
  Used for Places2 (Places365 ``val_256`` images and the RePaint test images
  ``gts/p256`` are already 256 x 256, for which it is the identity).

Images are returned as [1, 3, S, S] float tensors in [-1, 1].
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

from src.dataset import load_image as project_load_image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
PREPROCESSING = ("project_resize", "center_crop")


def center_crop_arr(pil_image: Image.Image, image_size: int) -> np.ndarray:
    """guided-diffusion / RePaint ``center_crop_arr`` (BOX halving, BICUBIC, centre crop)."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]


def load_image_center_crop(path: Union[str, Path], image_size: int = 256) -> torch.Tensor:
    with Image.open(path) as image:
        arr = center_crop_arr(image.convert("RGB"), image_size)
    tensor = torch.from_numpy(arr.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous()


def load_study_image(path: Union[str, Path], image_size: int, preprocessing: str) -> torch.Tensor:
    if preprocessing == "project_resize":
        return project_load_image(str(path), image_size=image_size)
    if preprocessing == "center_crop":
        return load_image_center_crop(path, image_size)
    raise ValueError(f"Unknown preprocessing {preprocessing!r}; choose from {PREPROCESSING}")


def resolve_data_dir(candidates: Sequence[Union[str, Path]], project_root: Path) -> Optional[Path]:
    """First existing directory among ``candidates`` (relative paths are taken from the project root)."""
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if path.is_dir():
            return path.resolve()
    return None


def list_images(data_dir: Path, extensions: Iterable[str] = IMAGE_EXTENSIONS,
                file_list: Optional[Union[str, Path]] = None) -> List[Path]:
    """Sorted image paths under ``data_dir`` (recursive), or those named in ``file_list``.

    ``file_list`` is a text file with one path per line, relative to ``data_dir``.
    Listed files that do not exist raise ``FileNotFoundError``.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {data_dir}")
    if file_list is not None:
        names = [line.strip() for line in Path(file_list).read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [data_dir / name for name in names]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} listed images are missing, e.g. {missing[:3]}")
        return paths
    exts = {e.lower() for e in extensions}
    return sorted((p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts),
                  key=lambda p: str(p.relative_to(data_dir)).lower())


def select_images(paths: Sequence[Path], max_images: Optional[int], selection: str = "first",
                  seed: int = 0) -> List[Path]:
    """``first`` keeps sorted order; ``seeded_random`` samples reproducibly, then sorts."""
    paths = list(paths)
    if max_images is None or max_images >= len(paths):
        chosen = paths
    elif selection == "first":
        chosen = paths[: int(max_images)]
    elif selection == "seeded_random":
        chosen = sorted(random.Random(int(seed)).sample(paths, int(max_images)), key=lambda p: str(p).lower())
    else:
        raise ValueError("selection must be 'first' or 'seeded_random'")
    return chosen


def dataset_images(dataset_cfg: dict, project_root: Path) -> List[Path]:
    """Resolve the configured image list for one dataset (raises a clear error if absent)."""
    data_dir = resolve_data_dir(dataset_cfg["data_dirs"], project_root)
    if data_dir is None:
        raise FileNotFoundError(
            f"None of the configured data directories exist for dataset "
            f"{dataset_cfg.get('name', '?')!r}: {dataset_cfg['data_dirs']}. See Docs/faces_places2_mask_lpips_study.md."
        )
    file_list = dataset_cfg.get("file_list")
    if file_list:
        file_list = Path(file_list)
        if not file_list.is_absolute():
            file_list = project_root / file_list
    paths = list_images(data_dir, dataset_cfg.get("extensions", IMAGE_EXTENSIONS), file_list)
    if not paths:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return select_images(paths, dataset_cfg.get("max_images"), dataset_cfg.get("selection", "first"),
                         dataset_cfg.get("selection_seed", 0))
