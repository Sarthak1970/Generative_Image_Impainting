"""Re-evaluate saved batch outputs against the corresponding ground-truth images."""
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import load_image
from src.mask import create_center_mask
from src.metrics import compute_metrics

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    image_size = int(cfg["data"]["image_size"])
    mask_ratio = float(cfg.get("repaint", {}).get("mask_ratio", 0.35))
    data_dir = PROJECT_ROOT / "data" / "celeba"
    result_root = PROJECT_ROOT / "results_benchmarks" / "batch_eval"
    raw_dir = result_root / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(f"No batch output directory found: {raw_dir}; run scripts/run_batch.py first")

    mask = create_center_mask(image_size=image_size, mask_ratio=mask_ratio)
    rows = []
    image_paths = sorted(
        (p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    for image_path in image_paths:
        rel = image_path.relative_to(data_dir)
        pred_path = raw_dir / rel.with_suffix(".png")
        if not pred_path.exists():
            continue
        target = load_image(str(image_path), image_size=image_size)
        prediction = load_image(str(pred_path), image_size=image_size)
        row = {"image": str(rel)}
        row.update(compute_metrics(prediction, target, mask, compute_ssim=True))
        rows.append(row)

    if not rows:
        raise FileNotFoundError("No matching target/output pairs were found for evaluation")

    metric_path = result_root / "per_image_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"num_images": len(rows)}
    metric_keys = [key for key in rows[0] if key != "image"]
    for key in metric_keys:
        vals = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        summary[f"mean_{key}"] = sum(vals) / len(vals) if vals else None
    (result_root / "summary_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved per-image metrics to {metric_path}")


if __name__ == "__main__":
    main()
