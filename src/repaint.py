import torch
from diffusers import DDPMScheduler


def repaint_step(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    t: int,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    One conditioned reverse DDPM step.

    mask = 1: missing region to generate
    mask = 0: known region to preserve

    x_t: current noisy image at timestep t.
    Returns the conditioned image at timestep t-1.
    """
    batch_size = x_t.shape[0]

    timestep = torch.full(
        (batch_size,),
        int(t),
        device=x_t.device,
        dtype=torch.long,
    )

    with torch.no_grad():
        model_output = model(x_t, timestep).sample

        x_prev_unknown = scheduler.step(
            model_output,
            int(t),
            x_t,
            generator=generator,
        ).prev_sample

    if t > 0:
        noise = torch.randn(
            x_0.shape,
            device=x_0.device,
            dtype=x_0.dtype,
            generator=generator,
        )

        previous_timestep = torch.full(
            (batch_size,),
            int(t) - 1,
            device=x_0.device,
            dtype=torch.long,
        )

        x_prev_known = scheduler.add_noise(
            x_0,
            noise,
            previous_timestep,
        )
    else:
        # At the final step, the known region is clean.
        x_prev_known = x_0

    mask = mask.to(device=x_t.device, dtype=x_t.dtype)

    return (
        (1.0 - mask) * x_prev_known
        + mask * x_prev_unknown
    )


def add_noise_forward_single_step(
    scheduler: DDPMScheduler,
    x_t_minus_1: torch.Tensor,
    t: int,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    Forward DDPM transition:
        x_(t-1) -> x_t

    Uses the original scheduler's beta_t coefficient.
    """
    if not 0 <= t < scheduler.config.num_train_timesteps:
        raise ValueError(f"Invalid forward timestep: {t}")

    beta_t = scheduler.betas[t].to(
        device=x_t_minus_1.device,
        dtype=x_t_minus_1.dtype,
    )

    noise = torch.randn(
        x_t_minus_1.shape,
        device=x_t_minus_1.device,
        dtype=x_t_minus_1.dtype,
        generator=generator,
    )

    return (
        torch.sqrt(1.0 - beta_t) * x_t_minus_1
        + torch.sqrt(beta_t) * noise
    )


def repaint_pipeline(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    jump_length: int = 10,
    jump_n_sample: int = 2,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """
    RePaint-style sampling with explicit forward/reverse jump cycles.

    jump_length:
        Number of consecutive forward transitions per jump cycle.

    jump_n_sample:
        Number of additional forward/reverse cycles at eligible timesteps.
        Set to 0 for the no-extra-resampling baseline.

    This implementation uses the full, contiguous training timestep schedule.
    """
    if jump_length < 1:
        raise ValueError("jump_length must be at least 1.")

    if jump_n_sample < 0:
        raise ValueError("jump_n_sample cannot be negative.")

    num_train_steps = int(scheduler.config.num_train_timesteps)

    scheduler.set_timesteps(num_train_steps)

    timesteps = [int(t) for t in scheduler.timesteps]

    expected = list(range(num_train_steps - 1, -1, -1))

    if timesteps != expected:
        raise RuntimeError(
            "Expected a contiguous descending DDPM timestep schedule. "
            "Subsampled inference schedules are not supported by this "
            "forward-jump implementation."
        )

    if x_0.shape != (x_0.shape[0], 3, x_0.shape[-2], x_0.shape[-1]):
        raise ValueError(
            f"Expected an RGB image tensor [B, 3, H, W]; got {x_0.shape}."
        )

    if mask.shape[-2:] != x_0.shape[-2:]:
        raise ValueError("Mask and image spatial dimensions must match.")

    if mask.shape[0] not in (1, x_0.shape[0]):
        raise ValueError("Mask batch dimension must be 1 or match image batch.")

    if mask.shape[1] != 1:
        raise ValueError("Mask must have one channel.")

    mask = mask.to(device=x_0.device, dtype=x_0.dtype)

    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Mask must be binary: 0 for known, 1 for missing.")

    # Initial state x_T.
    x_t = torch.randn(
        x_0.shape,
        device=x_0.device,
        dtype=x_0.dtype,
        generator=generator,
    )

    # Traverse the ordinary reverse diffusion trajectory.
    for t in timesteps:
        # x_t -> x_(t-1), with known-region conditioning.
        x_t = repaint_step(
            model=model,
            scheduler=scheduler,
            x_t=x_t,
            x_0=x_0,
            mask=mask,
            t=t,
            generator=generator,
        )

        # The current state is now at t-1.
        # A jump is legal only if every forward timestep exists.
        can_jump = (
            t > 0
            and t % jump_length == 0
            and t + jump_length - 1 < num_train_steps
        )

        if not can_jump or jump_n_sample == 0:
            continue

        for _ in range(jump_n_sample):
            # Forward: x_(t-1) -> x_t -> ... -> x_(t+L-1).
            for forward_t in range(t, t + jump_length):
                x_t = add_noise_forward_single_step(
                    scheduler=scheduler,
                    x_t_minus_1=x_t,
                    t=forward_t,
                    generator=generator,
                )

            # Reverse: x_(t+L-1) -> ... -> x_t -> x_(t-1).
            for reverse_t in range(
                t + jump_length - 1,
                t - 1,
                -1,
            ):
                x_t = repaint_step(
                    model=model,
                    scheduler=scheduler,
                    x_t=x_t,
                    x_0=x_0,
                    mask=mask,
                    t=reverse_t,
                    generator=generator,
                )

    return x_t