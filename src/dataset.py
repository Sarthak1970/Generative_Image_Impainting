import torch
import torchvision.transforms as T
from PIL import Image

def get_transforms(image_size:int=256):
    return T.Compose([
        T.resize((image_size,image_size)),
        T.ToTensor(), #scales to [0,1]
        T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    ])

def load_image(path:str, image_size:int=256)->torch.Tensor:
    """Load an image from the given path and return it as a tensor."""
    img=Image.open(path).convert("RGB")
    transform=get_transforms(image_size)
    return transform(img).unsqueeze(0)

def load_mask(path:str, image_size:int=256)->torch.Tensor:
    """Load mask image and convert it to a binary tensor of shape(1,1,H,W).
    pixel>0.5 = 1(region to repain) others are 0.0(known region)"""

    mask=Image.open(path).convert("L")
    transform=T.compose([
        T.Resize((image_size,image_size)),
        T.ToTensor()
    ])
    mask_tensor=transform(mask).unsqueeze(0)
    return (mask_tensor>0.5).float()

def tensor_to_pil(tensor:torch.Tensor)->Image.Image:
    """Tensor to PIL Image conversion"""
    tensor=(tensor.clamp(-1,1)+1.0)/2.0
    tensor=tensor.squeeze(0).cpu().permute(1,2,0)
    array=(tensor.numpy()*255).astype("uint8")
    return Image.fromarray(array)