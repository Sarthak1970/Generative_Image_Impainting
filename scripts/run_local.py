"""Run one deterministic, paper-schedule RePaint benchmark."""
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import load_image, tensor_to_pil
from src.mask import create_center_mask
from src.metrics import compute_metrics
from src.model import get_device, load_pretrained_ddpm
from src.repaint import build_repaint_schedule, repaint_pipeline

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def get_test_image(data_dir: Path) -> Path:
    images = sorted(
        (path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    if not images:
        raise FileNotFoundError(
            f"No supported image files found under {data_dir}. "
            "Place test images in data/celeba/ (nested folders are supported)."
        )
    return images[0]


def _write_json(path: Path, content) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    cfg = load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    image_size = int(cfg["data"]["image_size"])
    repaint_cfg = cfg.get("repaint", {})
    steps = int(repaint_cfg.get("num_inference_steps", repaint_cfg.get("num_inferencing_steps", 250)))
    jump_length = int(repaint_cfg.get("jump_length", 10))
    jump_n_sample = int(repaint_cfg.get("jump_n_sample", 10))
    mask_ratio = float(repaint_cfg.get("mask_ratio", 0.35))
    seed = int(repaint_cfg.get("seed", 42))

    device = get_device()
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Device             : {device}")
    print(f"Image size         : {image_size}")

    image_path = get_test_image(PROJECT_ROOT / "data" / "celeba")
    print(f"Deterministic input : {image_path}")
    x_0 = load_image(str(image_path), image_size=image_size).to(device)

    model_name = cfg["model"]["pretrained_model_name"]
    model, scheduler, device = load_pretrained_ddpm(model_name, device=device)
    print(f"Pretrained model   : {model_name}")
    print(f"Training timesteps : {int(scheduler.config.num_train_timesteps)}")

    mask = create_center_mask(image_size=image_size, mask_ratio=mask_ratio).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    schedule = build_repaint_schedule(steps, jump_length, jump_n_sample)
    reverse_calls = sum(a > b for a, b in zip(schedule[:-1], schedule[1:]))
    forward_transitions = sum(max(0, b - a) for a, b in zip(schedule[:-1], schedule[1:]))

    print(
        f"RePaint schedule    : T={steps}, jump_length={jump_length}, "
        f"jump_n_sample={jump_n_sample}"
    )
    print(f"Reverse model calls : {reverse_calls}")
    print(f"Forward respaced transitions: {forward_transitions}")
    print("Running sampler; the paper schedule intentionally revisits timesteps.")

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

    run_name = f"seed{seed}_T{steps}_j{jump_length}_r{jump_n_sample}_mask{mask_ratio:g}"
    result_dir = PROJECT_ROOT / "results_benchmarks" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    known_mask = 1.0 - mask
    masked_display = x_0 * known_mask + (-1.0) * mask  # black hole in [-1, 1] space
    composite = known_mask * x_0 + mask * output
    tensor_to_pil(x_0).save(result_dir / "original_input.png")
    tensor_to_pil(masked_display).save(result_dir / "masked_input.png")
    tensor_to_pil(output).save(result_dir / "generated_output.png")
    tensor_to_pil(composite).save(result_dir / "composited_output.png")
    mask_img = (mask[0, 0].detach().cpu().clamp(0, 1) * 255).round().byte().numpy()
    from PIL import Image
    Image.fromarray(mask_img, mode="L").save(result_dir / "mask_missing_white.png")

    metrics = compute_metrics(output, x_0, mask, compute_ssim=True)
    run_config = {
        "model_name": model_name,
        "input_image": str(image_path.relative_to(PROJECT_ROOT)),
        "image_size": image_size,
        "num_train_timesteps": int(scheduler.config.num_train_timesteps),
        "num_inference_steps": steps,
        "respacing_stride": int(scheduler.config.num_train_timesteps) // steps,
        "jump_length": jump_length,
        "jump_n_sample": jump_n_sample,
        "seed": seed,
        "mask_convention": "1=missing/repaint, 0=known",
        "mask_shape": list(mask.shape),
        "mask_missing_pixel_fraction": float(mask.mean().item()),
        "schedule_entries": len(schedule),
        "reverse_model_calls": reverse_calls,
        "forward_respaced_transitions": forward_transitions,
        "raw_known_region_final_max_abs_error": float(((output - x_0).abs() * known_mask).max().item()),
    }
    _write_json(result_dir / "config_used.json", run_config)
    _write_json(result_dir / "metrics.json", metrics)

    print(f"Known-region MAE   : {metrics['known_mae']:.8f}")
    print(f"Missing-region MAE : {metrics['missing_mae']:.6f}")
    print(f"Full-image PSNR    : {metrics['full_psnr_db']:.3f} dB")
    print(f"Full-image SSIM    : {metrics['full_ssim']}")
    print(f"Saved results      : {result_dir}")
    if metrics["full_ssim"] is None:
        print("Note: install requirements.txt to enable SSIM (torchmetrics).")


if __name__ == "__main__":
    main()
