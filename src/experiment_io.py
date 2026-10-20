"""Saving paired-comparison results so figures can be rebuilt without re-sampling.

Each call to :func:`save_comparison` writes one folder per (image, mask, seed):

    images/original.png, mask_missing_white.png, masked_input.png,
    images/<method>_raw.png, images/<method>_composite.png, pilot_raw.png
    metrics.csv      one row per method (metrics + NFE + timing + seeds)
    scores.csv       pilot instability scores per anchor
    allocation.csv   extra cycles per anchor for fixed / uniform / adaptive
    run.json         all metadata, the budget plan and the environment

Raw sampler outputs are kept separate from composites (known pixels restored).
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from src.dataset import tensor_to_pil
from src.metrics import compute_inpainting_metrics


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except ImportError:
        info["diffusers"] = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_total_memory_bytes"] = int(props.total_memory)
    return info


def _clean(value):
    """Make values JSON/CSV safe (no NaN/inf, tensors -> lists)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _write_csv(path: Path, rows) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: _clean(row.get(k)) for k in fields} for row in rows])


def save_mask_png(mask: torch.Tensor, path: Path) -> None:
    array = (mask[0, 0].detach().cpu().clamp(0, 1) * 255).round().byte().numpy()
    Image.fromarray(array, mode="L").save(path)


def save_comparison(
    result,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    out_dir: Path,
    metadata: Dict,
    compute_ssim: bool = True,
    lpips_fn=None,
    boundary_width: int = 8,
) -> Dict[str, str]:
    """Write images, metrics, scores, allocations and run metadata to ``out_dir``.

    ``metadata`` should at least contain ``model_identifier``, ``image_id`` and
    ``mask_type``; it is copied verbatim into run.json and every CSV row.
    """
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask = mask.to(device=x_0.device, dtype=x_0.dtype)
    paths: Dict[str, str] = {}

    tensor_to_pil(x_0).save(image_dir / "original.png")
    save_mask_png(mask, image_dir / "mask_missing_white.png")
    tensor_to_pil(x_0 * (1.0 - mask) - mask).save(image_dir / "masked_input.png")
    if result.pilot is not None:
        tensor_to_pil(result.pilot.output).save(image_dir / "pilot_raw.png")

    base = {k: metadata.get(k) for k in ("model_identifier", "image_id", "mask_type")}
    metric_rows = []
    method_summaries = {}
    for name, method in result.methods.items():
        raw = method.raw
        composite = (1.0 - mask) * x_0 + mask * raw
        raw_path = image_dir / f"{name}_raw.png"
        comp_path = image_dir / f"{name}_composite.png"
        tensor_to_pil(raw).save(raw_path)
        tensor_to_pil(composite).save(comp_path)
        metrics = compute_inpainting_metrics(raw, x_0, mask, boundary_width=boundary_width,
                                             compute_ssim=compute_ssim, lpips_fn=lpips_fn)
        summary = method.summary()
        row = dict(base)
        row.update({
            "method": name,
            "latent_seed": result.settings["latent_seed"],
            "noise_seed": method.noise_seed,
            "pilot_nfe": method.pilot_nfe,
            "final_nfe": method.final_nfe,
            "total_nfe": method.total_nfe,
            "expected_total_nfe": method.expected_total_nfe,
            "extra_cycles_total": method.extra_cycles_total,
            "pilot_wall_time_s": method.pilot_wall_time_s,
            "final_wall_time_s": method.final_wall_time_s,
            "total_wall_time_s": method.total_wall_time_s,
            "peak_gpu_memory_bytes": method.peak_gpu_memory_bytes,
        })
        row.update(metrics)
        row["raw_png"] = str(raw_path)
        row["composite_png"] = str(comp_path)
        metric_rows.append(row)
        summary["metrics"] = metrics
        summary["raw_png"] = str(raw_path)
        summary["composite_png"] = str(comp_path)
        method_summaries[name] = summary
    paths["metrics_csv"] = str(out_dir / "metrics.csv")
    _write_csv(Path(paths["metrics_csv"]), metric_rows)

    anchors = result.settings["anchors"]
    alloc_rows = []
    for i, anchor in enumerate(anchors):
        row = dict(base)
        row["anchor"] = anchor
        if result.scores is not None:
            row["native_timestep"] = result.scores["native_timestep"][i]
        for name in ("fixed", "uniform", "adaptive"):
            if name in result.methods:
                row[f"{name}_extra_cycles"] = result.methods[name].allocation[anchor]
        alloc_rows.append(row)
    paths["allocation_csv"] = str(out_dir / "allocation.csv")
    _write_csv(Path(paths["allocation_csv"]), alloc_rows)

    if result.scores is not None:
        per_anchor_keys = [k for k, v in result.scores.items() if isinstance(v, list) and len(v) == len(anchors)]
        score_rows = []
        for i in range(len(anchors)):
            row = dict(base)
            row.update({k: result.scores[k][i] for k in per_anchor_keys})
            score_rows.append(row)
        paths["scores_csv"] = str(out_dir / "scores.csv")
        _write_csv(Path(paths["scores_csv"]), score_rows)

    run = dict(metadata)
    run.update({
        "plan": result.plan.as_dict(),
        "settings": result.settings,
        "score_settings": {k: v for k, v in (result.scores or {}).items() if not isinstance(v, list)},
        "pilot": None if result.pilot is None else {
            "nfe": result.pilot.nfe, "wall_time_s": result.pilot.wall_time_s,
            "schedule_length": result.pilot.schedule_length,
            "recorded_levels": sorted(result.pilot.x0_estimates),
        },
        "methods": method_summaries,
        "mask_shape": list(mask.shape),
        "mask_missing_fraction": float(mask.mean().item()),
        "mask_convention": "1=missing/repaint, 0=known",
        "image_shape": list(x_0.shape),
        "environment": environment_info(),
        "outputs": dict(paths),
    })
    paths["run_json"] = str(out_dir / "run.json")
    Path(paths["run_json"]).write_text(json.dumps(_clean(run), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return paths
