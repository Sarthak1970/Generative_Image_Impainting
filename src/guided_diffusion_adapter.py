"""Load official RePaint / guided-diffusion checkpoints for the shared sampler.

Why this exists: RePaint's Places2 model (``places256_300000.pt``, listed in
the official ``download.sh`` and used by ``confs/test_p256_*.yml``) is an
OpenAI guided-diffusion ``UNetModel``, not a diffusers ``UNet2DModel``. The
project's sampler (``src.repaint``) calls ``model(x, t).sample`` and expects a
3-channel epsilon prediction. This module isolates the adaptation:

* the network definition is imported from a local checkout of the official
  RePaint repository (``guided_diffusion/``); nothing is vendored or copied;
* :class:`GuidedDiffusionEpsAdapter` calls the network with the native
  timestep (the sampler already passes ``level * stride``) and returns the
  first 3 output channels (epsilon) in ``.sample``;
* the model's learned-variance channels (``learn_sigma: true``) are DISCARDED.
  The shared sampler uses the fixed posterior variance for every model. The
  fixed posterior variance is the lower end of guided-diffusion's
  learned-range interpolation, so this is a documented approximation of how
  the checkpoint was sampled in the paper, not a reproduction of it;
* :func:`make_linear_ddpm_scheduler` builds a diffusers ``DDPMScheduler`` with
  the same linear beta schedule (1e-4 .. 0.02 over 1000 steps) as
  ``get_named_beta_schedule("linear", 1000)`` (float32 here vs float64
  officially).

Architecture of the Places2 checkpoint (from ``confs/test_p256_thick.yml``):
image_size 256, num_channels 256, num_res_blocks 2, channel_mult "" (= 1,1,2,2,4,4),
attention_resolutions 32,16,8, num_heads 4, num_head_channels 64,
num_heads_upsample -1, use_scale_shift_norm true, resblock_updown true,
learn_sigma true, class_cond false, dropout 0, diffusion_steps 1000,
noise_schedule linear, predict_xstart false, rescale_timesteps false,
timestep_respacing '250'. The same architecture is used by the official
CelebA-HQ ``test_c256_*`` configs.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union

import torch

REPAINT_P256_MODEL_CONFIG = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="32,16,8",
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=False,
    use_new_attention_order=False,
)


class GuidedDiffusionEpsAdapter:
    """Make a guided-diffusion UNet look like ``diffusers.UNet2DModel`` to the sampler."""

    def __init__(self, net, learn_sigma: bool = True, in_channels: int = 3):
        self.net = net
        self.learn_sigma = bool(learn_sigma)
        self.in_channels = int(in_channels)
        self.learned_variance_discarded = self.learn_sigma

    def __call__(self, sample: torch.Tensor, timesteps: torch.Tensor):
        out = self.net(sample, timesteps.long())
        expected = self.in_channels * (2 if self.learn_sigma else 1)
        if out.shape[1] != expected:
            raise ValueError(f"guided-diffusion output has {out.shape[1]} channels, expected {expected}")
        eps = out[:, : self.in_channels]
        return SimpleNamespace(sample=eps.to(sample.dtype))

    def parameters(self):
        return self.net.parameters()

    def eval(self):
        self.net.eval()
        return self


def import_official_guided_diffusion(code_dir: Union[str, Path]):
    """Import ``guided_diffusion.script_util`` from a local official RePaint checkout."""
    code_dir = Path(code_dir).expanduser().resolve()
    if not (code_dir / "guided_diffusion" / "unet.py").is_file():
        raise FileNotFoundError(
            f"{code_dir} is not an official RePaint checkout (guided_diffusion/unet.py missing). "
            "Clone https://github.com/andreas128/RePaint and set its path in configs/faces_places2_study.yaml."
        )
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    module = importlib.import_module("guided_diffusion.script_util")
    loaded_from = Path(module.__file__).resolve()
    if code_dir not in loaded_from.parents:
        raise ImportError(f"guided_diffusion was imported from {loaded_from}, not from {code_dir}")
    return module


def build_guided_diffusion_unet(code_dir: Union[str, Path], model_config: dict, diffusion_steps: int = 1000):
    script_util = import_official_guided_diffusion(code_dir)
    conf = SimpleNamespace(diffusion_steps=int(diffusion_steps), use_value_logger=False)
    return script_util.create_model(**model_config, conf=conf)


def make_linear_ddpm_scheduler(num_train_timesteps: int = 1000):
    """diffusers DDPMScheduler with guided-diffusion's 'linear' schedule (use_scale=True)."""
    from diffusers import DDPMScheduler

    scale = 1000.0 / num_train_timesteps
    return DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        beta_start=scale * 1e-4,
        beta_end=scale * 0.02,
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


def load_repaint_checkpoint(checkpoint_path: Union[str, Path], code_dir: Union[str, Path],
                            model_config: Optional[dict] = None, device="cpu",
                            diffusion_steps: int = 1000, use_fp16: bool = False):
    """Return ``(adapter, scheduler, info)`` for an official RePaint checkpoint.

    The state dict is loaded with ``strict=True``; any key mismatch raises.
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Obtain places256_300000.pt with the official RePaint "
            "download.sh (Google Drive via gdown) and attach/copy it to that path."
        )
    config = dict(REPAINT_P256_MODEL_CONFIG if model_config is None else model_config)
    config["use_fp16"] = bool(use_fp16)
    net = build_guided_diffusion_unet(code_dir, config, diffusion_steps)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=True)
    if use_fp16:
        net.convert_to_fp16()
    device = torch.device(device)
    net = net.eval().requires_grad_(False).to(device)
    adapter = GuidedDiffusionEpsAdapter(net, learn_sigma=config["learn_sigma"])
    scheduler = make_linear_ddpm_scheduler(diffusion_steps)
    info = {
        "backend": "repaint_guided_diffusion",
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "official_code_dir": str(Path(code_dir).resolve()),
        "model_config": config,
        "parameters_million": sum(p.numel() for p in net.parameters()) / 1e6,
        "learned_variance_discarded": adapter.learned_variance_discarded,
        "use_fp16": bool(use_fp16),
    }
    return adapter, scheduler, info


def _official_linear_alphas_cumprod(num_train_timesteps: int) -> torch.Tensor:
    scale = 1000.0 / num_train_timesteps
    betas = torch.linspace(scale * 1e-4, scale * 0.02, num_train_timesteps, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


@torch.no_grad()
def check_model_compatibility(model, scheduler, image_size: int, num_inference_steps: int,
                              device="cpu", n_timing_calls: int = 2) -> dict:
    """Run the checks the shared sampler relies on and return a report dict.

    Checks: epsilon prediction, clipping flag, train-steps divisibility by the
    respaced T, linear beta schedule matching guided-diffusion's, output shape
    3 x S x S and finiteness at t = max and t = 0, and the time per call.
    """
    device = torch.device(device)
    cfg = scheduler.config
    n_train = int(cfg.num_train_timesteps)
    report = {
        "num_train_timesteps": n_train,
        "prediction_type": cfg.prediction_type,
        "clip_sample": bool(cfg.clip_sample),
        "beta_schedule": cfg.beta_schedule,
        "beta_start": float(cfg.beta_start),
        "beta_end": float(cfg.beta_end),
        "scheduler_variance_type": getattr(cfg, "variance_type", None),
        "sampler_variance": "fixed posterior variance (src.repaint)",
        "num_inference_steps": int(num_inference_steps),
        "respacing_divisible": n_train % int(num_inference_steps) == 0,
    }
    reference = _official_linear_alphas_cumprod(n_train)
    report["alphas_cumprod_max_abs_diff_vs_guided_diffusion_linear"] = float(
        (scheduler.alphas_cumprod.double() - reference).abs().max().item())
    x = torch.randn(1, 3, image_size, image_size, generator=torch.Generator().manual_seed(0)).to(device)
    shapes, finite = [], True
    for t in (n_train - 1, 0):
        out = model(x, torch.full((1,), t, device=device, dtype=torch.long)).sample
        shapes.append(list(out.shape))
        finite = finite and bool(torch.isfinite(out).all().item())
    report["output_shapes"] = shapes
    report["output_finite"] = finite
    report["output_is_3_channel_epsilon_shape"] = all(s == [1, 3, image_size, image_size] for s in shapes)
    times = []
    for _ in range(max(0, n_timing_calls)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        model(x, torch.full((1,), n_train // 2, device=device, dtype=torch.long))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
    report["seconds_per_call"] = min(times) if times else None
    report["compatible"] = bool(
        report["prediction_type"] == "epsilon" and report["respacing_divisible"]
        and report["output_is_3_channel_epsilon_shape"] and report["output_finite"]
        and report["alphas_cumprod_max_abs_diff_vs_guided_diffusion_linear"] < 1e-5
    )
    return report
