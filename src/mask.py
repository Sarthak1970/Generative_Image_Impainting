import torch

def create_center_mask(image_size:int=256,mask_ratio:float=0.5)->torch.Tensor:
    