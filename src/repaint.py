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
    timestep=torch.tensor([t],device=x_t.device,d_type=torch.long) #integer to tensor convertsion (needed by diffuser)

    with torch.no_grad():
        model_output=model(x_t,timestep).sampe
        x_t_minus_1_unknown=scheduler.step(
            model_output,t,x_t,generator=generator
        ).prev_sample #reverse DDPM step for unknown region

    if t>0:
        noise=torch.randn(x_0.shape,device=x_0.device,generator=generator)
        t_prev=torch.tensor([t-1],device=x_0.device,dtype=torch.long)
        x_t_minus_1_known=scheduler.add_noise(x_0,noise,t_prev) #schduler.add_noise implements xt*sqrt(alpha) eqn

    else:
        x_t_minus_1_known=x_0#final step-no noise addition

    x_t_minus_1=mask*x_t_minus_1_unknown+(1-mask)*x_t_minus_1_known

    return x_t_minus_1

def add_noise_forward(
    scheduler:DDPMScheduler,
    x_t_minus_1:torch.tensor,
    t_prev=int,
    generator:torch.Generator=None
)->torch.Tensor:
    """Adds noise to the known region of the image at timestep t-1 to get x_t"""

    timestep=torch.tensor([t_prev],device=x_t_minus_1.device,dtype=torch.long)
    beta_t=scheduler.betas[t_prev].to(x_t_minus_1.device)

    noise=torch.randn(x_t_minus_1.shape,device=x_t_minus_1.device,generator=generator)
    x_t=torch.sqrt(1-beta_t)*x_t_minus_1+torch.sqrt(beta_t)*noise
    return x_t

def repaint_pipeline(  ## Repaint Loop
    model:torch.nn.Module,
    scheduler:DDPMScheduler,
    x_0:torch.Tensor,
    mask:torch.Tensor,
    jump_length:int=10,
    jump_n_sample:int=10,
    generator:torch.Generator=None
)->torch.Tensor:

    scheduler.set_timesteps(len(scheduler.config.get("beta_schedule",[])) or 1000)
    timesteps=scheduler.timesteps

    #pure Gaussian Noise at max timestep=1000
    x_t=torch.randn(x_0.shape,device=x_0.device,generator=generator)

    i=0
    total_steps=len(timesteps)

    while i<total_steps:
        t=int(timesteps[i].item())

        #t->t-1
        x_t=repaint_step(model,scheduler,x_t,x_0,mask,t,generator)

        if(i+1)%jump_length==0 and (i+1)<total_steps:
            for r in range(jump_n_sample):
                for j in range(jump_length):
                    t_curr=int(timesteps[i-j].item())

                    x_t=add_noise_forward(scheduler,x_t,t_curr,generator)

                for j in reversed(range(jump_length)):
                    t_curr=int(timesteps[i-j])
                    x_t=repaint_step(model,scheduler,x_t,x_0,mask,t_curr,generator)

        i+=1
        
    return x_t
