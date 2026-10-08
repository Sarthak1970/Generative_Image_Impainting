import torch
from diffusers import DDPMScheduler

def repaint_step(
    model:torch.nn.Module,
    scheduler:DDPMScheduler,
    x_t:torch.Tensor, #noisy latent image at timestep t
    x_0:torch.Tensor, #original image
    mask:torch.Tensor,
    t:int,
    generator:torch.Generator=None
)->torch.Tensor: #returns x{t-1}
    timestep=torch.tensor([t],device=x_t.device,dtype=torch.long) #integer to tensor convertsion (needed by diffuser)

    with torch.no_grad():
        model_output=model(x_t,timestep).sample
        x_t_minus_1_unknown=scheduler.step(
            model_output,t,x_t,generator=generator
        ).prev_sample #reverse DDPM step for unknown region

    if t>0:
        noise=torch.randn(x_0.shape,device=x_0.device,generator=generator)
        t_prev=torch.tensor([t-1],device=x_0.device,dtype=torch.long)
        x_t_minus_1_known=scheduler.add_noise(x_0,noise,t_prev) #schduler.add_noise implements xt*sqrt(alpha) eqn

    else:
        x_t_minus_1_known=x_0#final step-no noise addition

    x_t_minus_1 = (1.0 - mask) * x_t_minus_1_known + mask * x_t_minus_1_unknown
    return x_t_minus_1

def add_noise_forward_single_step(
        scheduler:DDPMScheduler,
        x_t_minus_1:torch.Tensor,
        t:int,
        generator:torch.Generator=None
)->torch.Tensor:
    '''Single Step transition noise for resampling jump'''
    beta_t=scheduler.betas[t].to(x_t_minus_1.device)
    noise=torch.randn(x_t_minus_1.shape,device=x_t_minus_1.device,generator=generator)

    x_t=torch.sqrt(1.0-beta_t)*x_t_minus_1+torch.sqrt(beta_t)*noise
    return x_t

def repaint_pipeline(
        model:torch.nn.Module,
        scheduler:DDPMScheduler,
        x_0:torch.Tensor,
        mask:torch.Tensor,
        jump_length:int=10,
        jump_n_sample:int=2,
        generator:torch.Generator=None
)->torch.Tensor:
    '''Repaint Pipeline'''

    scheduler.set_timesteps(1000)
    timesteps=scheduler.timesteps

    x_t=torch.randn(x_0.shape,device=x_0.device,generator=generator) #initial noise

    num_timesteps=len(timesteps)
    i=0

    while i<num_timesteps:
        t=int(timesteps[i])

        x_t=repaint_step(model,scheduler,x_t,x_0,mask,t,generator=generator) #standard reverse diffusion step

        if(i+1)%jump_length==0 and (i+1)<num_timesteps: #resampling jump condition
            for r in range(jump_n_sample):
                for j in range(jump_length):
                    t_curr=int(timesteps[i-jump_length+1+j])
                    x_t=add_noise_forward_single_step(scheduler,x_t,t_curr,generator)

                #backward diffusion
                for j in range(jump_length):
                    t_curr=int(timesteps[i-jump_length+1+j])
                    x_t=repaint_step(model,scheduler,x_t,x_0,mask,t_curr,generator)

        i+=1

    return x_t