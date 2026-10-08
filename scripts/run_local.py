import os
import torch
import urllib.request
from pathlib import Path
import random

from src.config import load_config
from src.dataset import load_image,tensor_to_pil
from src.mask import create_center_mask
from src.model import load_pretrained_ddpm,get_device
from src.repaint import repaint_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# def download_celeba_test_face(target_path:Path):
#     url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint_sample.png"
#     print(f"Downloading aligned CelebA test face to {target_path}")
#     target_path.parent.mkdir(parents=True,exist_ok=True)
#     urllib.request.urlretrieve(url,target_path)

def get_random_image(data_dir:Path)->Path:
    images=list(data_dir.glob("*.jpg"))

    if not images:
        raise FileNotFoundError(
            f"Image not found at {images}"
        )

    selected=random.choice(images)
    print(f"Selected test image:{selected}")
    return selected


def main():
    config_path=PROJECT_ROOT/"configs"/"config.yaml"
    cfg=load_config(str(config_path))

    device=get_device()
    model_name=cfg["model"]["pretrained_model_name"]
    print(f"Using Device : {device} | Model : {model_name}")

    model,scheduler,device=load_pretrained_ddpm(model_name,device=device)

    img_size=cfg["data"]["image_size"]
    celeba_dir = PROJECT_ROOT / "data" / "celeba"
    test_img_path = get_random_image(celeba_dir)

    x_0=load_image(str(test_img_path),img_size=img_size).to(device)
    print(f"Loaded Original Image")

    mask=create_center_mask(image_size=img_size,mask_ratio=0.35).to(device)

    print(f"Running Repaint pipeline (this may take few minutes)")
    out_tensor=repaint_pipeline(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        mask=mask,
        jump_length=10,
        jump_n_sample=2
    )

    result_dir=PROJECT_ROOT/"results"
    result_dir.mkdir(parents=True,exits_ok=True)

    out_img=tensor_to_pil(out_tensor)
    output_filepath=result_dir/"repaint_result.png"
    out_img.save(output_filepath)
    print(f"Saved Repaint Result to {output_filepath}")

if __name__ == "__main__":
    main()