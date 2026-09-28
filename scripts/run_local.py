import os
import torch
from pathlib import Path

from src.config import load_config
from src.dataset import load_image,tensor_to_pil
from src.mask import create_center_mask
from src.model import load_pretrained_ddpm,get_device
from src.repaint import repaint_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    config_path="configs/config.yaml"
    cfg=load_config(config_path)
    print(f"Loaded config from {config_path}: {cfg}")

    device=get_device()
    print(f"Using Device:{device}")

    model_name=cfg["model"]["pretrained_model_name"]
    print(f"Loading Pretrained Model:{model_name}")

    model,scheduler,device=load_pretrained_ddpm(model_name,device)

    img_size=cfg["data"]["image_size"]

    sample_img_path="data/sample_image.jpg"

    if os.path.exists(sample_img_path):
        x_0=load_image(sample_img_path,img_size).to(device)
        print(f"Loaded Sample Image from {sample_img_path}")

    else:
        print(f"Sample Image not fount at {sample_img_path}.Generating a synthetic placeholder image for testing.")
        x_0=torch.rand((1,3,img_size,img_size),device=device)*2-1


    mask=create_center_mask(image_size=img_size,mask_ratio=0.4).to(device)

    print("Starting Repaint Inference--")

    out_tensor=repaint_pipeline(
        model=model,
        scheduler=scheduler,
        x_0=x_0,
        mask=mask,
        jump_length=10,
        jump_n_sample=2
    )

    results_dir=PROJECT_ROOT/"results"
    results_dir.mkdir(parents=True,exist_ok=True)

    out_img=tensor_to_pil(out_tensor)
    out_img_path=results_dir/"repaint_output.jpg"
    out_img.save(out_img_path)
    print(f"Repaint Output saved at {out_img_path}")

if __name__ == "__main__":
    main()