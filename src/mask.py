import torch

def create_center_mask(image_size:int=256,mask_ratio:float=0.5)->torch.Tensor:
    """"Return a mask tensor where 1.0=region to repaint, 0.0=ground truth"""
    mask=torch.zeros((1,1,image_size,image_size),dtype=torch.float32)

    box_size=int(image_size*mask_ratio)
    start=(image_size-box_size)//2
    end=start+box_size

    mask[:,:,start:end,start:end]=1.0
    return mask

def create_half_mask(image_size:int=256,mode:str="bottom")->torch.Tensor:
    #mode-['left','right','top','bottom']

    mask=torch.zeros((1,1,image_size,image_size),dtype=torch.float32)
    half=image_size//2

    if mode=="bottom":
        mask[:,:,half:,:]=1.0
    elif mode=="top":
        mask[:,:,:half,:]=1.0
    elif mode=="left":
        mask[:,:,:,half:]=1.0
    elif mode=='right':
        mask[:,:,:,:half]=1.0
    else:
        raise ValueError(f"Unknown Value{mode}")

    return mask