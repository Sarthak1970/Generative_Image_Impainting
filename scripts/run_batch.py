import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path

from src.config import load_config
from src.dataset import load_image, tensor_to_pil
from src.model import load_pretrained_ddpm, get_device
from src.mask import create_center_mask
from src.repaint import repaint_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def create_image_grid(gt_pil, mask_tensor, out_pil):
    """Creates a side-by-side grid of [Ground Truth | Masked Input | Result]."""
    mask_pil = TF.to_pil_image(mask_tensor.squeeze(0))
    
    masked_input = Image.composite(gt_pil, Image.new("RGB", gt_pil.size, "black"), mask_pil)
    
    # Concatenate side-by-side
    w, h = gt_pil.size
    grid = Image.new('RGB', (w * 3, h))
    grid.paste(gt_pil, (0, 0))
    grid.paste(masked_input, (w, 0))
    grid.paste(out_pil, (w * 2, 0))
    return grid

def main():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    cfg = load_config(str(config_path))
    
    device = get_device()
    print(f"[+] Loading RePaint Model onto {device}...")
    model, scheduler, device = load_pretrained_ddpm(cfg["model"]["pretrained_model_name"], device=device)

    data_dir = PROJECT_ROOT / "data" / "celeba"
    results_dir = PROJECT_ROOT / "results" / "batch_eval"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    img_size = cfg["data"]["image_size"]
    
    # Generate the standard evaluation mask (1.0 = known, 0.0 = repaint)
    mask = create_center_mask(image_size=img_size, mask_ratio=0.35).to(device)

    image_paths = sorted(list(data_dir.glob("*.png")))
    print(f"[+] Found {len(image_paths)} images for batch inference.")

    for img_path in image_paths:
        print(f"[*] Processing {img_path.name}...")
        
        # Load and normalize ground truth [-1, 1]
        x_0 = load_image(str(img_path), image_size=img_size).to(device)
        
        # Run diffusion
        out_tensor = repaint_pipeline(
            model=model,
            scheduler=scheduler,
            x_0=x_0,
            mask=mask,
            jump_length=10,       # Can be increased to 250 for paper-quality results
            jump_n_sample=2       # Can be increased to 10 for paper-quality results
        )
        
        # Save isolated output
        out_pil = tensor_to_pil(out_tensor)
        out_pil.save(results_dir / f"out_{img_path.name}")
        
        # Save side-by-side visual comparison grid
        gt_pil = tensor_to_pil(x_0)
        grid_pil = create_image_grid(gt_pil, mask, out_pil)
        grid_pil.save(results_dir / f"grid_{img_path.name}")

    print(f"[+] Batch inference complete. Outputs saved to {results_dir}")

if __name__ == "__main__":
    main()