#load model
import torch
from diffusers import UNet2DModel, DDPMScheduler

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def load_pretrained_ddpm(model_name: str, device: torch.device = None):
    """
    Args:
        model_name: HF model path (e.g. 'google/ddpm-celebahq-256')
        device: torch.device (defaults to auto-detected best device)
    """
    if device is None:
        device = get_device()
        
    model = UNet2DModel.from_pretrained(model_name)
    scheduler = DDPMScheduler.from_pretrained(model_name)
    
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, scheduler, device