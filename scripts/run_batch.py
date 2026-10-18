"""Run a small, deterministic RePaint benchmark over a dataset subset."""
import csv
import json
from pathlib import Path
import sys

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import load_image, tensor_to_pil
from src.mask import create_center_mask
from src.metrics import compute_metrics
from src.model import get_device, load_pretrained_ddpm
from src.repaint import repaint_pipeline

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    image_size = int(cfg["data"]["image_size"])
    rcfg = cfg.get("repaint", {})
    steps = int(rcfg.get("num_inference_steps", rcfg.get("num_inferencing_steps", 250)))
    jump_length = int(rcfg.get("jump_length", 10))
    jump_n_sample = int(rcfg.get("jump_n_sample", 10))
    mask_ratio = float(rcfg.get("mask_ratio", 0.35))
    seed = int(rcfg.get("seed", 42))
    max_images = cfg.get("evaluation", {}).get("max_images", 5)

    data_dir = PROJECT_ROOT / "data" / "celeba"
    image_paths = sorted(
        (p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if max_images is not None:
        image_paths = image_paths[: int(max_images)]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {data_dir}")

    device = get_device()
    model, scheduler, device = load_pretrained_ddpm(cfg["model"]["pretrained_model_name"], device=device)
    mask = create_center_mask(image_size=image_size, mask_ratio=mask_ratio).to(device)
    result_root = PROJECT_ROOT / "results_benchmarks" / "batch_eval"
    raw_dir = result_root / "raw"
    grid_dir = result_root / "grids"
    raw_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for index, image_path in enumerate(image_paths):
        rel = image_path.relative_to(data_dir)
        print(f"[{index + 1}/{len(image_paths)}] {rel}")
        x_0 = load_image(str(image_path), image_size=image_size).to(device)
        generator = torch.Generator(device=device).manual_seed(seed + index)
        output = repaint_pipeline(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            mask=mask,
            num_inference_steps=steps,
            jump_length=jump_length,
            jump_n_sample=jump_n_sample,
            generator=generator,
        )

        out_rel = rel.with_suffix(".png")
        raw_path = raw_dir / out_rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        tensor_to_pil(output).save(raw_path)

        gt_pil = tensor_to_pil(x_0)
        masked_display = tensor_to_pil(x_0 * (1.0 - mask) - mask)
        out_pil = tensor_to_pil(output)
        grid = Image.new("RGB", (image_size * 3, image_size))
        grid.paste(gt_pil, (0, 0))
        grid.paste(masked_display, (image_size, 0))
        grid.paste(out_pil, (image_size * 2, 0))
        grid_path = grid_dir / out_rel
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(grid_path)

        metric_row = {"image": str(rel), "seed": seed + index}
        metric_row.update(compute_metrics(output, x_0, mask, compute_ssim=True))
        metric_rows.append(metric_row)

    metric_path = result_root / "per_image_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {"num_images": len(metric_rows), "run_seed": seed, "num_inference_steps": steps,
               "jump_length": jump_length, "jump_n_sample": jump_n_sample}
    for key in ("full_mse", "full_psnr_db", "full_ssim", "missing_mse", "missing_mae",
                "missing_psnr_db", "known_mse", "known_mae", "known_psnr_db"):
        vals = [row[key] for row in metric_rows if isinstance(row.get(key), (int, float))]
        summary[f"mean_{key}"] = sum(vals) / len(vals) if vals else None
    (result_root / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Processed {len(metric_rows)} images. Metrics: {metric_path}")


if __name__ == "__main__":
    main()
