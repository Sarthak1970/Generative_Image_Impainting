import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# When running as a script, this points to the project root if this file is in scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_and_transform(img_path, size=256):
    """Load an image as RGB and return a [1, 3, H, W] float tensor in [0, 1]."""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return TF.to_tensor(img).unsqueeze(0)

def main():
    data_dir = PROJECT_ROOT / "data" / "celeba"
    results_dir = PROJECT_ROOT / "results" / "batch_eval"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    gt_paths = sorted(data_dir.glob("*.png"))

    if not gt_paths:
        print(f"[-] No ground-truth PNG images found in: {data_dir}")
        return

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    image_names = []
    mse_values = []
    psnr_values = []
    ssim_values = []

    print("[+] Running structural evaluation metrics...")

    for gt_path in gt_paths:
        out_path = results_dir / f"out_{gt_path.name}"
        if not out_path.exists():
            print(f"[skip] Missing output: {out_path.name}")
            continue

        try:
            gt_tensor = load_and_transform(gt_path)
            out_tensor = load_and_transform(out_path)

            # Compute MSE directly; metric modules are used for PSNR and SSIM.
            mse = torch.mean((out_tensor - gt_tensor) ** 2).item()
            psnr = psnr_metric(out_tensor, gt_tensor).item()
            ssim = ssim_metric(out_tensor, gt_tensor).item()

            image_names.append(gt_path.stem)
            mse_values.append(mse)
            psnr_values.append(psnr)
            ssim_values.append(ssim)

        except Exception as exc:
            print(f"[skip] Could not evaluate {gt_path.name}: {exc}")

    if not image_names:
        print(f"[-] No matching output images found in: {results_dir}")
        return

    # Reset metric state between runs / avoid accumulated state affecting later images.
    # TorchMetrics may accumulate state internally, so instantiate fresh metrics per image
    # in the loop above if changing this script to use distributed/epoch-style evaluation.

    avg_mse = float(np.mean(mse_values))
    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))

    print(f"\n--- Evaluation Results ({len(image_names)} Images) ---")
    print(f"Average MSE:  {avg_mse:.6f} (Lower is better)")
    print(f"Average PSNR: {avg_psnr:.4f} dB (Higher is better)")
    print(f"Average SSIM: {avg_ssim:.4f} (Closer to 1.0 is better)")

    # 1) Per-image metric plots: one PNG per metric.
    plot_specs = [
        ("MSE", mse_values, "MSE (lower is better)", "mse_per_image.png"),
        ("PSNR (dB)", psnr_values, "PSNR in dB (higher is better)", "psnr_per_image.png"),
        ("SSIM", ssim_values, "SSIM (closer to 1 is better)", "ssim_per_image.png"),
    ]

    x = np.arange(len(image_names))
    for ylabel, values, title, filename in plot_specs:
        fig_width = max(9, min(18, len(image_names) * 0.45))
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        ax.bar(x, values)
        ax.set_title(title)
        ax.set_xlabel("Image")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, rotation=60, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        fig.tight_layout()
        output_path = plots_dir / filename
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {output_path}")

    # 2) Average metrics summary chart (separate panels because metric scales differ).
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    averages = [avg_mse, avg_psnr, avg_ssim]
    labels = ["MSE", "PSNR (dB)", "SSIM"]
    descriptions = ["Lower is better", "Higher is better", "Closer to 1 is better"]

    for ax, value, label, description in zip(axes, averages, labels, descriptions):
        ax.bar([label], [value])
        ax.set_title(f"{label}\n{description}")
        ax.set_ylabel("Average value")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.text(0, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"RePaint Evaluation — {len(image_names)} images")
    fig.tight_layout()
    summary_path = plots_dir / "average_metrics.png"
    fig.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {summary_path}")

    # 3) Save the per-image results as CSV for later analysis/reporting.
    csv_path = plots_dir / "metrics_per_image.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,mse,psnr_db,ssim\n")
        for name, mse, psnr, ssim in zip(image_names, mse_values, psnr_values, ssim_values):
            f.write(f'"{name}",{mse:.8f},{psnr:.6f},{ssim:.6f}\n')
        f.write(f'"AVERAGE",{avg_mse:.8f},{avg_psnr:.6f},{avg_ssim:.6f}\n')
    print(f"[saved] {csv_path}")

if __name__ == "__main__":
    main()
