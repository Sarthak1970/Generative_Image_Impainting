import random
from pathlib import Path

import torch

from src.config import load_config
from src.dataset import load_image, tensor_to_pil
from src.mask import create_center_mask
from src.model import load_pretrained_ddpm, get_device
from src.repaint import repaint_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_test_image() -> Path:

    data_dir = PROJECT_ROOT / "data" / "celeba"

    # Search recursively because the downloaded dataset may
    # contain an inner directory such as img_align_celeba/
    images = list(data_dir.rglob("*.png"))

    # Also allow JPG/JPEG images
    images += list(data_dir.rglob("*.jpg"))
    images += list(data_dir.rglob("*.jpeg"))

    if not images:
        raise FileNotFoundError(
            f"\nNo CelebA images found in:\n"
            f"{data_dir}\n\n"
            f"Put some CelebA images inside this directory."
        )

    image_path = random.choice(images)

    print(f"Using test image: {image_path}")

    return image_path

def main():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    cfg = load_config(str(config_path))

    img_size = cfg["data"]["image_size"]

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Image size   : {img_size}")

    device = get_device()

    model_name = cfg["model"]["pretrained_model_name"]

    print(
        f"Using Device : {device} | "
        f"Model : {model_name}"
    )
    model, scheduler, device = load_pretrained_ddpm(
        model_name,
        device=device
    )

    print("Pretrained DDPM loaded.")

    test_img_path = get_test_image()

    x_0 = load_image(str(test_img_path))

    x_0 = x_0.to(device)

    print(f"Loaded original image.")
    print(f"Input tensor shape : {x_0.shape}")
    print(f"Input tensor range : [{x_0.min().item():.3f}, "
          f"{x_0.max().item():.3f}]")

    if x_0.shape[-1] != img_size or x_0.shape[-2] != img_size:

        print(
            f"Resizing image from "
            f"{x_0.shape[-2]}x{x_0.shape[-1]} "
            f"to {img_size}x{img_size}"
        )

        x_0 = torch.nn.functional.interpolate(
            x_0,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False
        )

    mask = create_center_mask(
        image_size=img_size,
        mask_ratio=0.35
    ).to(device)

    print(f"Mask shape : {mask.shape}")
    print("Created center mask.")
    print()
    print("=" * 60)
    print("Running RePaint")
    print("=" * 60)
    print("This may take several minutes...")


    out_tensor = repaint_pipeline(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        mask=mask,

        # RePaint resampling parameters
        jump_length=5,
        jump_n_sample=2
    )

    result_dir = PROJECT_ROOT / "results"

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    out_img = tensor_to_pil(out_tensor)

    output_filepath = result_dir / "repaint_result.png"

    out_img.save(output_filepath)

    print()
    print("RePaint finished!")
    print(f"Saved result to:")
    print(output_filepath)



if __name__ == "__main__":
    main()