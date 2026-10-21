"""Reproducible mask families for the faces / Places2 mask study.

Convention (same as ``src/mask.py``): 1.0 = missing / repaint, 0.0 = known.
Every generator returns a float32 tensor of shape [1, 1, S, S].

Masks depend only on (family, target ratio, image index, base seed) and the
image size, *not* on the dataset. The same mask is therefore used for image
index ``i`` of the faces set and image index ``i`` of the Places2 set, which
gives matched mask geometry across the two domains.

Families (target = fraction of missing pixels):

* ``center_square``    one centred square, side = round(S * sqrt(target)).
* ``large_rectangle``  one axis-aligned rectangle of area target*S^2, random
                       aspect ratio in [1/2, 2] and random position. Treated as
                       not applicable below 25 % (a 10 % rectangle is not
                       "large"; see ``DEFAULT_NOT_APPLICABLE``).
* ``random_rectangles`` several rectangles (sides 8-35 % of S) at random
                       positions, possibly overlapping.
* ``multiple_holes``   many small ellipses (radii 3-10 % of S).
* ``irregular_brush``  free-form brush strokes: polylines of 4-10 segments with
                       random angles, lengths 5-25 % of S and widths 3-8 % of
                       S, drawn with round joints (in the spirit of the
                       free-form masks of Yu et al. 2019 / LaMa; not their code).
* ``stripes_horizontal`` / ``stripes_vertical`` exactly round(target * n)
                       of the n bands of thickness S/64 (4 px at 256) are
                       missing, spread as evenly as possible (error diffusion)
                       with a random phase; adjacent bands merge into thicker
                       stripes at high ratios. Coverage is within half a band.
* ``random_blocks``    a 16 x 16 grid; round(target * 256) cells chosen at random.
* ``border_frame``     "boundary-focused": a frame of missing pixels along the
                       image border around a centred known square, so the
                       model must extend content outwards to the image
                       boundary. Object-edge-following masks were not
                       implemented: they depend on image content and would
                       break mask matching across datasets.

For the stochastic families (``random_rectangles``, ``multiple_holes``,
``irregular_brush``) shapes are added one at a time; the shape that would
overshoot the target is shrunk by bisection on its size, so the achieved
coverage is normally within ``tolerance`` (default 0.01) of the target. The
achieved coverage is always measured and recorded; it is never assumed.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

MASK_FAMILIES: Tuple[str, ...] = (
    "center_square",
    "random_rectangles",
    "irregular_brush",
    "multiple_holes",
    "stripes_horizontal",
    "stripes_vertical",
    "large_rectangle",
    "random_blocks",
    "border_frame",
)
DEFAULT_TARGET_RATIOS: Tuple[float, ...] = (0.10, 0.25, 0.40, 0.60)
DEFAULT_NOT_APPLICABLE: Dict[str, Tuple[float, ...]] = {"large_rectangle": (0.10,)}
STOCHASTIC_FAMILIES = ("random_rectangles", "multiple_holes", "irregular_brush")


def mask_seed(family: str, target_ratio: float, image_index: int, base_seed: int) -> int:
    """Stable 31-bit seed from (family, ratio, image index, base seed).

    Uses CRC32 (not Python's salted ``hash``) so it is identical across runs,
    machines and Python versions. The dataset name is deliberately excluded.
    """
    key = f"{family}|{float(target_ratio):.4f}|{int(image_index)}|{int(base_seed)}"
    return zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF


def is_applicable(family: str, target_ratio: float,
                  not_applicable: Optional[Dict[str, Sequence[float]]] = None) -> bool:
    table = DEFAULT_NOT_APPLICABLE if not_applicable is None else not_applicable
    return not any(abs(float(target_ratio) - float(r)) < 1e-9 for r in table.get(family, ()))


def mask_conditions(families: Sequence[str] = MASK_FAMILIES,
                    ratios: Sequence[float] = DEFAULT_TARGET_RATIOS,
                    not_applicable: Optional[Dict[str, Sequence[float]]] = None) -> List[Tuple[str, float]]:
    """All applicable (family, target_ratio) pairs, in a fixed order."""
    unknown = [f for f in families if f not in MASK_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown mask families {unknown}; choose from {MASK_FAMILIES}")
    return [(f, float(r)) for f in families for r in ratios if is_applicable(f, r, not_applicable)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))[None, None]


def _check(size: int, target: float) -> None:
    if int(size) != size or size < 16:
        raise ValueError("size must be an integer >= 16")
    if not 0.0 < float(target) < 1.0:
        raise ValueError("target ratio must be in (0, 1)")


def _rasterize(size: int, draw_fn: Callable[[ImageDraw.ImageDraw], None]) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    draw_fn(ImageDraw.Draw(image))
    return np.asarray(image) > 127


def _fill_to_target(size: int, target: float, rng: np.random.Generator,
                    sample_params: Callable[[np.random.Generator], dict],
                    render: Callable[[dict, float], np.ndarray],
                    tolerance: float, max_shapes: int = 2000) -> np.ndarray:
    """Add random shapes until coverage reaches ``target`` (+/- tolerance).

    A shape that would overshoot ``target + tolerance`` is shrunk by bisection
    on a scale factor in (0, 1] (largest scale that does not exceed target).
    """
    canvas = np.zeros((size, size), dtype=bool)
    for _ in range(max_shapes):
        if canvas.mean() >= target - tolerance:
            break
        params = sample_params(rng)
        candidate = canvas | render(params, 1.0)
        if candidate.mean() <= target + tolerance:
            canvas = candidate
            continue
        lo, hi, best = 0.0, 1.0, canvas
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            trial = canvas | render(params, mid)
            if trial.mean() > target:
                hi = mid
            else:
                lo, best = mid, trial
        canvas = best
    return canvas


# ---------------------------------------------------------------------------
# Deterministic families
# ---------------------------------------------------------------------------
def center_square(size: int, target: float, rng: np.random.Generator) -> np.ndarray:
    side = int(round(size * math.sqrt(target)))
    side = min(max(side, 1), size)
    start = (size - side) // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[start:start + side, start:start + side] = True
    return mask


def large_rectangle(size: int, target: float, rng: np.random.Generator) -> np.ndarray:
    area = target * size * size
    aspect = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))  # width / height
    height = int(round(math.sqrt(area / aspect)))
    height = min(max(height, 1), size)
    width = int(round(area / height))
    if width > size:
        width = size
        height = min(size, int(round(area / size)))
    top = int(rng.integers(0, size - height + 1))
    left = int(rng.integers(0, size - width + 1))
    mask = np.zeros((size, size), dtype=bool)
    mask[top:top + height, left:left + width] = True
    return mask


def _stripes(size: int, target: float, rng: np.random.Generator, axis: int) -> np.ndarray:
    band = max(1, size // 64)
    n_bands = int(math.ceil(size / band))
    k = int(round(target * n_bands))           # exactly k bands are missing
    rate = k / n_bands
    phase = float(rng.uniform(0.0, 1.0))
    chosen = [int(math.floor((i + 1) * rate + phase)) - int(math.floor(i * rate + phase)) == 1
              for i in range(n_bands)]
    lines = np.zeros(size, dtype=bool)
    for i, on in enumerate(chosen):
        if on:
            lines[i * band:(i + 1) * band] = True
    mask = np.zeros((size, size), dtype=bool)
    if axis == 0:
        mask[lines, :] = True
    else:
        mask[:, lines] = True
    return mask


def stripes_horizontal(size: int, target: float, rng: np.random.Generator) -> np.ndarray:
    return _stripes(size, target, rng, axis=0)


def stripes_vertical(size: int, target: float, rng: np.random.Generator) -> np.ndarray:
    return _stripes(size, target, rng, axis=1)


def random_blocks(size: int, target: float, rng: np.random.Generator, grid: int = 16) -> np.ndarray:
    cells = grid * grid
    k = int(round(target * cells))
    chosen = rng.choice(cells, size=k, replace=False)
    cell_mask = np.zeros(cells, dtype=bool)
    cell_mask[chosen] = True
    cell_mask = cell_mask.reshape(grid, grid)
    edges = np.linspace(0, size, grid + 1).round().astype(int)
    mask = np.zeros((size, size), dtype=bool)
    for r in range(grid):
        for c in range(grid):
            if cell_mask[r, c]:
                mask[edges[r]:edges[r + 1], edges[c]:edges[c + 1]] = True
    return mask


def border_frame(size: int, target: float, rng: np.random.Generator) -> np.ndarray:
    keep = int(round(size * math.sqrt(1.0 - target)))
    keep = min(max(keep, 1), size)
    start = (size - keep) // 2
    mask = np.ones((size, size), dtype=bool)
    mask[start:start + keep, start:start + keep] = False
    return mask


# ---------------------------------------------------------------------------
# Stochastic families
# ---------------------------------------------------------------------------
def random_rectangles(size: int, target: float, rng: np.random.Generator, tolerance: float = 0.01) -> np.ndarray:
    def sample(g):
        return dict(cx=g.uniform(0, size), cy=g.uniform(0, size),
                    w=g.uniform(0.08, 0.35) * size, h=g.uniform(0.08, 0.35) * size)

    def render(p, scale):
        w, h = max(p["w"] * scale, 1.0), max(p["h"] * scale, 1.0)
        box = (p["cx"] - w / 2, p["cy"] - h / 2, p["cx"] + w / 2, p["cy"] + h / 2)
        return _rasterize(size, lambda d: d.rectangle(box, fill=255))

    return _fill_to_target(size, target, rng, sample, render, tolerance)


def multiple_holes(size: int, target: float, rng: np.random.Generator, tolerance: float = 0.01) -> np.ndarray:
    def sample(g):
        return dict(cx=g.uniform(0, size), cy=g.uniform(0, size),
                    rx=g.uniform(0.03, 0.10) * size, ry=g.uniform(0.03, 0.10) * size)

    def render(p, scale):
        rx, ry = max(p["rx"] * scale, 0.5), max(p["ry"] * scale, 0.5)
        box = (p["cx"] - rx, p["cy"] - ry, p["cx"] + rx, p["cy"] + ry)
        return _rasterize(size, lambda d: d.ellipse(box, fill=255))

    return _fill_to_target(size, target, rng, sample, render, tolerance)


def irregular_brush(size: int, target: float, rng: np.random.Generator, tolerance: float = 0.01) -> np.ndarray:
    def sample(g):
        n = int(g.integers(4, 11))
        x, y = g.uniform(0, size), g.uniform(0, size)
        points = [(x, y)]
        angle = g.uniform(0, 2 * math.pi)
        for _ in range(n):
            angle += g.uniform(-math.pi / 2, math.pi / 2)
            length = g.uniform(0.05, 0.25) * size
            x = float(np.clip(x + length * math.cos(angle), 0, size - 1))
            y = float(np.clip(y + length * math.sin(angle), 0, size - 1))
            points.append((x, y))
        return dict(points=points, width=g.uniform(0.03, 0.08) * size)

    def render(p, scale):
        width = max(p["width"] * scale, 1.0)

        def draw(d):
            d.line(p["points"], fill=255, width=int(round(width)))
            r = width / 2
            for (px, py) in p["points"]:
                d.ellipse((px - r, py - r, px + r, py + r), fill=255)

        return _rasterize(size, draw)

    return _fill_to_target(size, target, rng, sample, render, tolerance)


_GENERATORS = {
    "center_square": center_square,
    "random_rectangles": random_rectangles,
    "irregular_brush": irregular_brush,
    "multiple_holes": multiple_holes,
    "stripes_horizontal": stripes_horizontal,
    "stripes_vertical": stripes_vertical,
    "large_rectangle": large_rectangle,
    "random_blocks": random_blocks,
    "border_frame": border_frame,
}


@dataclass
class MaskSpec:
    family: str
    target_ratio: float
    image_index: int
    base_seed: int
    seed: int
    size: int
    actual_ratio: float
    mask: torch.Tensor  # [1, 1, S, S], 1 = missing

    def metadata(self) -> dict:
        return {
            "mask_family": self.family,
            "mask_target_ratio": self.target_ratio,
            "mask_actual_ratio": self.actual_ratio,
            "mask_coverage_error": self.actual_ratio - self.target_ratio,
            "mask_seed": self.seed,
            "mask_base_seed": self.base_seed,
            "mask_image_index": self.image_index,
            "mask_size": self.size,
        }


def generate_mask(family: str, target_ratio: float, size: int = 256, image_index: int = 0,
                  base_seed: int = 0, tolerance: float = 0.01) -> MaskSpec:
    """Generate one reproducible mask and measure its actual coverage."""
    if family not in _GENERATORS:
        raise ValueError(f"Unknown mask family {family!r}; choose from {MASK_FAMILIES}")
    _check(size, target_ratio)
    seed = mask_seed(family, target_ratio, image_index, base_seed)
    rng = np.random.default_rng(seed)
    generator = _GENERATORS[family]
    if family in STOCHASTIC_FAMILIES:
        array = generator(int(size), float(target_ratio), rng, tolerance=tolerance)
    else:
        array = generator(int(size), float(target_ratio), rng)
    tensor = _to_tensor(array)
    return MaskSpec(family=family, target_ratio=float(target_ratio), image_index=int(image_index),
                    base_seed=int(base_seed), seed=seed, size=int(size),
                    actual_ratio=float(tensor.mean().item()), mask=tensor)


def coverage_table(families: Sequence[str] = MASK_FAMILIES, ratios: Sequence[float] = DEFAULT_TARGET_RATIOS,
                   size: int = 256, image_indices: Sequence[int] = (0,), base_seed: int = 0,
                   not_applicable: Optional[Dict[str, Sequence[float]]] = None) -> List[dict]:
    """Target vs achieved coverage for every applicable condition (no model needed)."""
    rows = []
    for family, ratio in mask_conditions(families, ratios, not_applicable):
        for index in image_indices:
            spec = generate_mask(family, ratio, size, index, base_seed)
            rows.append({k: v for k, v in spec.metadata().items()})
    return rows
