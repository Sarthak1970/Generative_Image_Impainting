
import json
import random
import time
from pathlib import Path

import torch

from src.config import load_config
from src.dataset import load_image, tensor_to_pil
from src.mask import create_center_mask
from src.model import load_pretrained_ddpm, get_device
from src.repaint import repaint_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_ROOT / "results_benchmarks"

SEED = 42
MASK_RATIO = 0.35

# Start with no extra jumps for the baseline.
# After validating the sampler, compare fixed resampling budgets.
JUMP_LENGTH = 10
JUMP_N_SAMPLE = 0

# Set to a filename if you want a specific image.
# Example: "000001.png"
FIXED_IMAGE_NAME = None


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_test_image() -> Path:
    """Select a deterministic image from data/celeba."""
    data_dir = PROJECT_ROOT / "data" / "celeba"

    extensions = {".png", ".jpg", ".jpeg"}
    images = sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise FileNotFoundError(
            f"No PNG/JPG/JPEG images found in:\n{data_dir}\n"
            "Place your test images inside data/celeba/."
        )

    if FIXED_IMAGE_NAME is not None:
        matches = [
            p for p in images if p.name == FIXED_IMAGE_NAME
        ]
        if not matches:
            raise FileNotFoundError(
                f"Could not find {FIXED_IMAGE_NAME} in {data_dir}"
            )
        image_path = matches[0]
    else:
        image_path = images[0]

    print(f"Selected image: {image_path}")
    return image_path


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a [1, C, H, W] image tensor using the project's converter."""
    path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_pil(tensor.detach().cpu())
    image.save(path)


def main() -> None:
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Separate output folder for this configuration.
    run_name = (
        f"seed{SEED}_mask{MASK_RATIO:.2f}_"
        f"jump{JUMP_LENGTH}_n{JUMP_N_SAMPLE}"
    )
    run_dir = RESULT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = load_config(str(config_path))

    image_size = int(cfg["data"]["image_size"])
    model_name = cfg["model"]["pretrained_model_name"]

    device = get_device()

    print("=" * 64)
    print("RePaint single-image experiment")
    print("=" * 64)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {model_name}")
    print(f"Device       : {device}")
    print(f"Image size   : {image_size}")
    print(f"Seed         : {SEED}")
    print(f"Mask ratio   : {MASK_RATIO}")
    print(f"Jump length  : {JUMP_LENGTH}")
    print(f"Jump samples : {JUMP_N_SAMPLE}")
    print(f"Output dir   : {run_dir}")
    print("=" * 64)


    model, scheduler, device = load_pretrained_ddpm(
        model_name,
        device=device,
    )
    model.eval()

    print("Pretrained model loaded.")

    image_path = get_test_image()
    x_0 = load_image(str(image_path)).to(device)

    if x_0.ndim != 4 or x_0.shape[0] != 1:
        raise ValueError(
            f"Expected one image with shape [1, C, H, W], got {x_0.shape}"
        )

    if x_0.shape[1] != 3:
        raise ValueError(
            f"Expected an RGB image with 3 channels, got {x_0.shape[1]}"
        )

    if x_0.shape[-2:] != (image_size, image_size):
        print(
            f"Resizing input from {tuple(x_0.shape[-2:])} "
            f"to {(image_size, image_size)}"
        )
        x_0 = torch.nn.functional.interpolate(
            x_0,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

    x_0 = x_0.clamp(-1.0, 1.0)

    print(f"Input shape  : {tuple(x_0.shape)}")
    print(
        f"Input range  : "
        f"[{x_0.min().item():.4f}, {x_0.max().item():.4f}]"
    )

    # -----------------------------------------------------
    # Create mask
    # Convention: 1 = missing, 0 = known
    # -----------------------------------------------------

    mask = create_center_mask(
        image_size=image_size,
        mask_ratio=MASK_RATIO,
    ).to(device=device, dtype=x_0.dtype)

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            f"Expected mask shape [B, 1, H, W], got {mask.shape}"
        )

    if mask.shape[0] not in (1, x_0.shape[0]):
        raise ValueError("Mask batch dimension does not match the input.")

    if mask.shape[-2:] != x_0.shape[-2:]:
        raise ValueError("Mask and image dimensions do not match.")

    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary, with values 0 and 1.")

    missing_fraction = mask.float().mean().item()

    print(f"Mask shape   : {tuple(mask.shape)}")
    print(f"Missing area : {missing_fraction:.4f}")

    # -----------------------------------------------------
    # Save original and masked input before inference
    # -----------------------------------------------------

    save_tensor_image(x_0, run_dir / "original_input.png")

    masked_input = x_0 * (1.0 - mask)
    save_tensor_image(masked_input, run_dir / "masked_input.png")

    save_tensor_image(
        mask.expand(-1, 3, -1, -1),
        run_dir / "mask.png",
    )

    generator = torch.Generator(device=device).manual_seed(SEED)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = repaint_pipeline(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            mask=mask,
            jump_length=JUMP_LENGTH,
            jump_n_sample=JUMP_N_SAMPLE,
            generator=generator,
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    if output.shape != x_0.shape:
        raise ValueError(
            f"Output shape {output.shape} does not match input {x_0.shape}"
        )

    if not torch.isfinite(output).all():
        raise ValueError("Output contains NaN or infinite values.")

    output = output.clamp(-1.0, 1.0)

    save_tensor_image(output, run_dir / "generated_output.png")

    # Explicitly composite the original known pixels with the
    # generated missing region for a preservation reference.
    composite = (1.0 - mask) * x_0 + mask * output
    save_tensor_image(composite, run_dir / "composited_output.png")

    known_region = 1.0 - mask

    known_abs_error = (
        (output - x_0).abs() * known_region
    ).sum() / (
        known_region.sum() * x_0.shape[1]
    ).clamp_min(1.0)

    metrics = {
        "input_image": str(image_path.relative_to(PROJECT_ROOT)),
        "model": model_name,
        "device": str(device),
        "seed": SEED,
        "image_size": image_size,
        "mask_ratio_parameter": MASK_RATIO,
        "actual_missing_fraction": missing_fraction,
        "jump_length": JUMP_LENGTH,
        "jump_n_sample": JUMP_N_SAMPLE,
        "runtime_seconds": elapsed_seconds,
        "known_region_mean_absolute_error": known_abs_error.item(),
        "output_min": output.min().item(),
        "output_max": output.max().item(),
    }

    if device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "image_path": str(image_path),
                "seed": SEED,
                "mask_ratio": MASK_RATIO,
                "jump_length": JUMP_LENGTH,
                "jump_n_sample": JUMP_N_SAMPLE,
                "image_size": image_size,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 64)
    print("Experiment completed")
    print("=" * 64)
    print(f"Runtime                  : {elapsed_seconds:.2f} seconds")
    print(f"Known-region MAE         : {known_abs_error.item():.6f}")
    print(f"Output directory         : {run_dir}")
    print(f"Metrics                  : {run_dir / 'metrics.json'}")
    print(f"Generated image          : {run_dir / 'generated_output.png'}")
    print(f"Composited image         : {run_dir / 'composited_output.png'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
