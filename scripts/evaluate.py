import torch
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_and_transform(img_path, size=256):
    img = Image.open(img_path).convert("RGB").resize((size, size))
    return TF.to_tensor(img).unsqueeze(0)

def main():
    data_dir = PROJECT_ROOT / "data" / "celeba"
    results_dir = PROJECT_ROOT / "results" / "batch_eval"
    
    gt_paths = sorted(list(data_dir.glob("*.png")))
    
    if not gt_paths:
        print("[-] No ground truth images found. Run the data download script first.")
        return
        
    mse_metric = MeanSquaredError()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    total_mse, total_psnr, total_ssim = 0, 0, 0
    count = 0

    print("[+] Running structural evaluation metrics...")
    
    for gt_path in gt_paths:
        out_path = results_dir / f"out_{gt_path.name}"
        if not out_path.exists():
            continue
            
        gt_tensor = load_and_transform(gt_path)
        out_tensor = load_and_transform(out_path)
        
        total_mse += mse_metric(out_tensor, gt_tensor).item()
        total_psnr += psnr_metric(out_tensor, gt_tensor).item()
        total_ssim += ssim_metric(out_tensor, gt_tensor).item()
        count += 1
        
    if count == 0:
        print("[-] No matching outputs found in results/batch_eval/")
        return
        
    print(f"\n--- Evaluation Results ({count} Images) ---")
    print(f"Average MSE:  {total_mse / count:.4f} (Lower is better)")
    print(f"Average PSNR: {total_psnr / count:.4f} dB (Higher is better)")
    print(f"Average SSIM: {total_ssim / count:.4f} (Closer to 1.0 is better)")

if __name__ == "__main__":
    main()