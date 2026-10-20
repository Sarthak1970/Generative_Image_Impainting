# Non-monotonic schedule logic adapted from RePaint (CC BY-NC-SA 4.0); see Docs/.
"""RePaint sampling for a pretrained epsilon-prediction DDPM.

The resampling schedule follows the paper-facing schedule in the authors'
repository (andreas128/RePaint, guided_diffusion/scheduler.py). Conditioning
is injected before reverse denoising, matching the authors' p_sample path.

Mask convention used by this project: 1 = missing / repaint, 0 = known.
Input images are expected to be normalized to [-1, 1].
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch


def eligible_jump_anchors(num_inference_steps: int, jump_length: int) -> List[int]:
    """Respaced levels at which the official schedule may start a jump.

    These are exactly the keys of the ``jumps`` dictionary built by the
    official ``get_schedule_jump`` / ``get_schedule_jump_paper``:
    ``range(0, t_T - jump_length, jump_length)``. A jump from anchor ``a``
    climbs to ``a + jump_length``, which is therefore always a valid level.
    For T=250, j=10 this gives the 24 anchors 0, 10, ..., 230.
    """
    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be >= 2")
    if jump_length < 1:
        raise ValueError("jump_length must be >= 1")
    return list(range(0, num_inference_steps - jump_length, jump_length))


def build_repaint_schedule_from_counts(
    num_inference_steps: int,
    jump_length: int,
    extra_cycles: Mapping[int, int],
) -> List[int]:
    """Official RePaint jump schedule with a per-anchor number of extra cycles.

    ``extra_cycles[a]`` is the number of *additional* revisits (a forward jump
    of ``jump_length`` levels followed by ``jump_length`` reverse steps) made
    at anchor ``a``. The official schedule is the special case
    ``extra_cycles[a] = jump_n_sample - 1`` for every eligible anchor.
    Anchors missing from the mapping receive zero extra cycles. Only anchors
    returned by :func:`eligible_jump_anchors` are legal.
    """
    anchors = eligible_jump_anchors(num_inference_steps, jump_length)
    legal = set(anchors)
    jumps: Dict[int, int] = {a: 0 for a in anchors}
    for anchor, count in extra_cycles.items():
        if int(anchor) != anchor or int(anchor) not in legal:
            raise ValueError(
                f"Anchor {anchor!r} is not a legal RePaint jump anchor for "
                f"T={num_inference_steps}, jump_length={jump_length}; legal anchors: {anchors}"
            )
        if int(count) != count or count < 0:
            raise ValueError(f"Extra-cycle count for anchor {anchor} must be a non-negative integer")
        jumps[int(anchor)] = int(count)

    # The control flow below is the official get_schedule_jump_paper() loop.
    t = num_inference_steps
    times: List[int] = []
    while t >= 1:
        t -= 1
        times.append(t)
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                t += 1
                times.append(t)
    times.append(-1)

    # Equivalent to the important invariants checked in the official scheduler.
    if times[0] != num_inference_steps - 1 or times[-1] != -1:
        raise RuntimeError("Internal error constructing RePaint schedule")
    if any(abs(a - b) != 1 for a, b in zip(times, times[1:])):
        raise RuntimeError("RePaint schedule has a non-unit transition")
    if any(t < -1 or t >= num_inference_steps for t in times):
        raise RuntimeError("RePaint schedule contains an out-of-range timestep")
    if any(count != 0 for count in jumps.values()):
        raise RuntimeError("Not every requested resampling cycle was scheduled")
    return times


def count_schedule_transitions(schedule: List[int]) -> Tuple[int, int]:
    """Return ``(reverse_model_calls, forward_noising_steps)`` for a schedule.

    Every decreasing pair is one denoiser evaluation; every increasing pair is
    one forward noising step that does not call the network.
    """
    reverse = sum(b < a for a, b in zip(schedule[:-1], schedule[1:]))
    forward = sum(b > a for a, b in zip(schedule[:-1], schedule[1:]))
    return int(reverse), int(forward)


def build_repaint_schedule(
    num_inference_steps: int = 250,
    jump_length: int = 10,
    jump_n_sample: int = 10,
) -> List[int]:
    """Return RePaint's non-monotonic respaced-time schedule.

    This implements the simple schedule used for the paper configuration:
    every ``jump_length`` respaced steps, revisit the interval ``jump_n_sample``
    times in total. The result begins at ``num_inference_steps - 1`` and ends
    at -1. Adjacent schedule entries must differ by exactly one.

    The schedule logic is adapted from the authors' official implementation:
    https://github.com/andreas128/RePaint/blob/main/guided_diffusion/scheduler.py
    """
    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be >= 2")
    if jump_length < 1:
        raise ValueError("jump_length must be >= 1")
    if jump_n_sample < 1:
        raise ValueError("jump_n_sample must be >= 1")

    # Official RePaint jump schedule: jump_n_sample=10 means nine additional
    # revisits at each eligible jump point, not ten extra revisits.
    jumps = {
        j: jump_n_sample - 1
        for j in eligible_jump_anchors(num_inference_steps, jump_length)
    }
    return build_repaint_schedule_from_counts(num_inference_steps, jump_length, jumps)


def _randn_like(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.randn(
        x.shape,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )


def _reverse_ddpm_respaced_step(
    model,
    scheduler,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    missing_mask: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    stride: int,
    generator: Optional[torch.Generator] = None,
    alpha_bar_values: Optional[torch.Tensor] = None,
    condition_known: bool = True,
    return_pred_x0: bool = False,
):
    """One reverse transition in the respaced DDPM chain.

    Returns the next latent, or ``(next_latent, pred_x0)`` when
    ``return_pred_x0=True``. ``pred_x0`` is the clipped clean-image estimate
    the posterior mean is built from; returning it changes no computation.

    Following the RePaint implementation, the known part is noised to the
    *current* timestep and inserted into the model input before prediction.
    The model then predicts the previous state for the whole image. In the
    official code the very first reverse call skips this injection because no
    previous ``pred_xstart`` exists yet; ``condition_known=False`` reproduces
    that behavior for the pipeline's first reverse call.

    This function uses epsilon prediction and fixed-small posterior variance,
    which matches this project's ``google/ddpm-celebahq-256`` checkpoint, not
    the authors' separate six-channel learned-variance checkpoint.
    """
    del stride  # documents the fact this is a respaced transition
    batch = x_t.shape[0]
    t_batch = torch.full((batch,), int(timestep), device=x_t.device, dtype=torch.long)

    alpha_bar = alpha_bar_values
    if alpha_bar is None:
        alpha_bar = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
    alpha_bar_t = alpha_bar[int(timestep)]
    alpha_bar_prev = (
        torch.ones((), device=x_t.device, dtype=x_t.dtype)
        if prev_timestep < 0
        else alpha_bar[int(prev_timestep)]
    )

    # The official p_sample draws its reverse noise first, then the independent
    # conditioning noise, then performs the model call. Keep this order stable
    # so the seeded random stream mirrors that control flow.
    reverse_noise = _randn_like(x_t, generator)

    # RePaint's conditioning injection occurs before calling the denoiser:
    # q(x_t | x_0) on the known region, current latent on the missing region.
    model_input = x_t
    if condition_known:
        known_noise = _randn_like(x_0, generator)
        known_at_t = alpha_bar_t.sqrt() * x_0 + (1.0 - alpha_bar_t).clamp(min=0.0).sqrt() * known_noise
        model_input = (1.0 - missing_mask) * known_at_t + missing_mask * x_t

    with torch.no_grad():
        model_output = model(model_input, t_batch).sample

    if model_output.shape != x_t.shape:
        raise ValueError(
            "Expected an epsilon-prediction model with output shape equal to input; "
            f"got input {tuple(x_t.shape)} and output {tuple(model_output.shape)}. "
            "The official RePaint checkpoint predicts mean and variance in 6 channels, "
            "whereas this project uses a 3-channel Hugging Face DDPM checkpoint."
        )

    prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")
    if prediction_type == "epsilon":
        pred_x0 = (model_input - (1.0 - alpha_bar_t).sqrt() * model_output) / alpha_bar_t.sqrt()
    elif prediction_type == "sample":
        pred_x0 = model_output
    elif prediction_type == "v_prediction":
        pred_x0 = alpha_bar_t.sqrt() * model_input - (1.0 - alpha_bar_t).sqrt() * model_output
    else:
        raise ValueError(f"Unsupported scheduler prediction_type: {prediction_type}")

    if getattr(getattr(scheduler, "config", None), "clip_sample", True):
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

    # Effective beta/posterior for the jump between selected original DDPM
    # timesteps. This is the standard SpacedDiffusion posterior for respacing.
    beta_eff = (1.0 - alpha_bar_t / alpha_bar_prev).clamp(min=0.0, max=1.0)
    denom = (1.0 - alpha_bar_t).clamp(min=1e-20)
    coef_x0 = alpha_bar_prev.sqrt() * beta_eff / denom
    coef_xt = (alpha_bar_t / alpha_bar_prev).clamp(min=0.0).sqrt() * (1.0 - alpha_bar_prev) / denom
    posterior_mean = coef_x0 * pred_x0 + coef_xt * model_input
    posterior_variance = ((1.0 - alpha_bar_prev) / denom * beta_eff).clamp(min=0.0)

    if prev_timestep < 0:
        # At the final transition alpha_bar_prev=1, so posterior_mean is the
        # clipped x0 prediction and no stochastic term is added.
        next_latent = posterior_mean
    else:
        next_latent = posterior_mean + posterior_variance.sqrt() * reverse_noise
    if return_pred_x0:
        return next_latent, pred_x0
    return next_latent

def repaint_step(
    model: torch.nn.Module,
    scheduler,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    t: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Compatibility helper for a one-training-step reverse transition.

    ``mask`` follows the project's convention (1 = missing, 0 = known).
    The full pipeline below uses respaced transitions and should be preferred.
    """
    if t < 0:
        raise ValueError("t must be non-negative")
    return _reverse_ddpm_respaced_step(
        model=model,
        scheduler=scheduler,
        x_t=x_t,
        x_0=x_0,
        missing_mask=mask,
        timestep=int(t),
        prev_timestep=int(t) - 1,
        stride=1,
        generator=generator,
    )


def add_noise_forward_single_step(
    scheduler,
    x_t_minus_1: torch.Tensor,
    t: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply q(x_t | x_{t-1}) for one native training timestep ``t``."""
    if t <= 0:
        raise ValueError("Forward transition timestep t must be >= 1")
    beta_t = scheduler.betas[int(t)].to(device=x_t_minus_1.device, dtype=x_t_minus_1.dtype)
    noise = _randn_like(x_t_minus_1, generator)
    return (1.0 - beta_t).sqrt() * x_t_minus_1 + beta_t.sqrt() * noise


RepaintStepCallback = Callable[[int, int, torch.Tensor], None]


def _prepare_sampling_inputs(scheduler, x_0: torch.Tensor, mask: torch.Tensor, num_inference_steps: int):
    """Validate inputs and return ``(expanded_mask, stride, alpha_bar_values)``.

    Shared by the fixed RePaint pipeline and the budget-aware extension so
    both use identical validation, respacing and cached coefficients.
    """
    if x_0.ndim != 4 or x_0.shape[1] != 3:
        raise ValueError(f"x_0 must have shape [B, 3, H, W], got {tuple(x_0.shape)}")
    if not torch.is_floating_point(x_0):
        raise TypeError("x_0 must be a floating point tensor")
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[-2:] != x_0.shape[-2:]:
        raise ValueError(
            "mask must have shape [1 or B, 1, H, W] matching x_0 spatial dimensions"
        )
    if mask.shape[0] not in (1, x_0.shape[0]):
        raise ValueError("mask batch dimension must be 1 or equal to x_0 batch size")
    if not torch.isfinite(x_0).all():
        raise ValueError("x_0 contains NaN or infinite values")

    mask = mask.to(device=x_0.device, dtype=x_0.dtype).expand(x_0.shape[0], -1, -1, -1)
    if torch.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError("mask values must be in [0, 1]")

    scheduler_config = getattr(scheduler, "config", None)
    configured_train_steps = getattr(scheduler_config, "num_train_timesteps", None)
    if configured_train_steps is None:
        scheduler_betas = getattr(scheduler, "betas", None)
        if scheduler_betas is None:
            raise ValueError("scheduler must expose config.num_train_timesteps or betas")
        configured_train_steps = len(scheduler_betas)
    num_train_timesteps = int(configured_train_steps)
    if num_inference_steps > num_train_timesteps:
        raise ValueError(
            f"num_inference_steps ({num_inference_steps}) cannot exceed model training steps "
            f"({num_train_timesteps})"
        )
    if num_train_timesteps % num_inference_steps != 0:
        raise ValueError(
            "num_train_timesteps must be divisible by num_inference_steps for this respaced DDPM "
            "implementation. The paper setup is 1000 training steps / 250 inference levels."
        )
    stride = num_train_timesteps // num_inference_steps
    # Cache coefficients on the sampling device once; avoid thousands of tiny
    # CPU-to-GPU scheduler tensor copies in the repeated RePaint loop.
    alpha_bar_values = scheduler.alphas_cumprod.to(device=x_0.device, dtype=x_0.dtype)
    return mask, stride, alpha_bar_values


def _validate_schedule(schedule: List[int], num_inference_steps: int) -> None:
    if len(schedule) < 2 or schedule[0] != num_inference_steps - 1 or schedule[-1] != -1:
        raise ValueError(
            f"schedule must start at {num_inference_steps - 1} and end at -1"
        )
    if any(abs(a - b) != 1 for a, b in zip(schedule[:-1], schedule[1:])):
        raise ValueError("schedule transitions must change the level by exactly one")
    if any(t < -1 or t >= num_inference_steps for t in schedule):
        raise ValueError("schedule contains an out-of-range level")
    if any(t < 0 for t in schedule[:-1]):
        raise ValueError("only the terminal schedule entry may be -1")


def _execute_repaint_schedule(
    model,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    schedule: List[int],
    stride: int,
    alpha_bar_values: torch.Tensor,
    x_t: torch.Tensor,
    generator: Optional[torch.Generator],
    on_reverse_step: Optional[RepaintStepCallback] = None,
) -> torch.Tensor:
    """Run the RePaint transitions listed in ``schedule`` starting from ``x_t``.

    This is the single transition loop used by every sampler in the project.
    ``on_reverse_step(level, next_level, pred_x0)`` is an optional read-only
    observer called after each reverse (model-calling) transition.
    """
    # Mirrors the official p_sample_loop: conditioning injection is guarded by
    # whether a previous x_start prediction exists, so the very first reverse
    # call is unconditioned and subsequent reverse calls inject q(x_t | x_0).
    has_previous_prediction = False
    for current_level, next_level in zip(schedule[:-1], schedule[1:]):
        if next_level < current_level:
            current_timestep = current_level * stride
            prev_timestep = -1 if next_level < 0 else next_level * stride
            x_t, pred_x0 = _reverse_ddpm_respaced_step(
                model=model,
                scheduler=scheduler,
                x_t=x_t,
                x_0=x_0,
                missing_mask=mask,
                timestep=current_timestep,
                prev_timestep=prev_timestep,
                stride=stride,
                generator=generator,
                alpha_bar_values=alpha_bar_values,
                condition_known=has_previous_prediction,
                return_pred_x0=True,
            )
            has_previous_prediction = True
            if on_reverse_step is not None:
                on_reverse_step(int(current_level), int(next_level), pred_x0)
        elif next_level > current_level:
            # One effective forward step in the respaced chain. Its beta is
            # beta'_s = 1 - alpha_bar[t_high] / alpha_bar[t_low], equivalent
            # in distribution to composing all native DDPM transitions between
            # the two retained timesteps (official RePaint ``undo``).
            low_timestep = current_level * stride
            high_timestep = next_level * stride
            alpha_bar_low = alpha_bar_values[low_timestep]
            alpha_bar_high = alpha_bar_values[high_timestep]
            alpha_effective = (alpha_bar_high / alpha_bar_low).clamp(min=0.0, max=1.0)
            beta_effective = (1.0 - alpha_effective).clamp(min=0.0, max=1.0)
            x_t = alpha_effective.sqrt() * x_t + beta_effective.sqrt() * _randn_like(x_t, generator)
        else:  # defensive check; schedule builders forbid this
            raise RuntimeError("Duplicate adjacent entries in RePaint schedule")

    return x_t


def _initial_latent(x_0: torch.Tensor, x_T: Optional[torch.Tensor], generator: Optional[torch.Generator]):
    if x_T is None:
        return torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)
    if tuple(x_T.shape) != tuple(x_0.shape):
        raise ValueError(f"x_T must have shape {tuple(x_0.shape)}, got {tuple(x_T.shape)}")
    x_T = x_T.to(device=x_0.device, dtype=x_0.dtype)
    if not torch.isfinite(x_T).all():
        raise ValueError("x_T contains NaN or infinite values")
    return x_T.clone()


def run_repaint_schedule(
    model,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    schedule: List[int],
    num_inference_steps: int,
    generator: Optional[torch.Generator] = None,
    x_T: Optional[torch.Tensor] = None,
    on_reverse_step: Optional[RepaintStepCallback] = None,
) -> torch.Tensor:
    """Run an explicit RePaint level schedule with the shared transitions.

    ``schedule`` must be a list of respaced levels from ``T-1`` down to -1
    with unit steps, e.g. from :func:`build_repaint_schedule` or
    :func:`build_repaint_schedule_from_counts`. ``x_T`` optionally fixes the
    initial latent; otherwise it is drawn from ``generator`` exactly as in
    :func:`repaint_pipeline`.
    """
    mask, stride, alpha_bar_values = _prepare_sampling_inputs(scheduler, x_0, mask, num_inference_steps)
    _validate_schedule(list(schedule), num_inference_steps)
    x_t = _initial_latent(x_0, x_T, generator)
    return _execute_repaint_schedule(
        model, scheduler, x_0, mask, list(schedule), stride, alpha_bar_values,
        x_t, generator, on_reverse_step,
    )


def repaint_pipeline(
    model: torch.nn.Module,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    jump_length: int = 10,
    jump_n_sample: int = 10,
    generator: Optional[torch.Generator] = None,
    num_inference_steps: int = 250,
    x_T: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the paper-style RePaint schedule with respaced DDPM transitions.

    Args:
        model: Pretrained unconditional epsilon-prediction DDPM.
        scheduler: Its matching ``DDPMScheduler`` (same beta schedule).
        x_0: Original RGB image tensor [B, 3, H, W], normalized to [-1, 1].
        mask: Binary or soft mask [1 or B, 1, H, W], 1 = missing, 0 = known.
        jump_length: Number of respaced levels traversed in each forward jump.
        jump_n_sample: Total number of visits per eligible jump point, including
            the first visit. The paper configuration is 10.
        generator: Optional seeded torch.Generator on the same device as x_0.
        num_inference_steps: Number of respaced model-time levels; paper default 250.
        x_T: Optional initial latent with the shape of ``x_0``. When omitted
            (the default, and the previous behaviour) it is drawn from
            ``generator`` before any sampler noise.

    Returns:
        Raw RePaint sample. Known pixels are injected as correctly noised values
        before reverse denoising calls, but the raw final sample is not guaranteed
        to equal the input pixels exactly. For display/export, composite the raw
        result with x_0 using ``(1 - mask) * x_0 + mask * result``.
    """
    mask, stride, alpha_bar_values = _prepare_sampling_inputs(scheduler, x_0, mask, num_inference_steps)
    schedule = build_repaint_schedule(num_inference_steps, jump_length, jump_n_sample)
    # Respaced level k is evaluated at native timestep k * stride (0, 4, ...,
    # 996 for 1000/250). The official SpacedDiffusion with
    # timestep_respacing='250' instead keeps round(k * 999 / 249)
    # (0, 4, ..., 999); the two maps differ by at most 3 native steps. See
    # Docs/budget_aware_repaint_methods.md.
    x_t = _initial_latent(x_0, x_T, generator)
    return _execute_repaint_schedule(
        model, scheduler, x_0, mask, schedule, stride, alpha_bar_values, x_t, generator,
    )
