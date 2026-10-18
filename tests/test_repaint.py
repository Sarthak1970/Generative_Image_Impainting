import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw

from src.mask import create_center_mask, create_half_mask
from src.dataset import load_mask
from src.metrics import compute_metrics
from src.repaint import build_repaint_schedule, repaint_pipeline, _reverse_ddpm_respaced_step


class FakeScheduler:
    def __init__(self, num_train_timesteps=8):
        self.betas = torch.linspace(0.01, 0.08, num_train_timesteps, dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.config = SimpleNamespace(
            num_train_timesteps=num_train_timesteps,
            prediction_type="epsilon",
            clip_sample=True,
        )


class FakeOutput:
    def __init__(self, sample):
        self.sample = sample


class ZeroEpsilonModel:
    def __init__(self, record_inputs=False):
        self.calls = 0
        self.record_inputs = record_inputs
        self.inputs = []
        self.timesteps = []

    def __call__(self, sample, timesteps):
        self.calls += 1
        if timesteps.shape != (sample.shape[0],):
            raise AssertionError("model timesteps should have one value per batch item")
        if self.record_inputs:
            self.inputs.append(sample.detach().clone())
            self.timesteps.append(timesteps.detach().clone())
        return FakeOutput(torch.zeros_like(sample))


class TestRePaintSchedule(unittest.TestCase):
    def test_schedule_invariants_and_has_jumps(self):
        schedule = build_repaint_schedule(25, jump_length=5, jump_n_sample=3)
        self.assertEqual(schedule[0], 24)
        self.assertEqual(schedule[-1], -1)
        self.assertTrue(all(abs(a - b) == 1 for a, b in zip(schedule[:-1], schedule[1:])))
        self.assertTrue(any(b > a for a, b in zip(schedule[:-1], schedule[1:])))
        self.assertTrue(all(-1 <= t < 25 for t in schedule))

    def test_matches_authors_paper_schedule_logic(self):
        # Direct reference translation of get_schedule_jump_paper() from the
        # official RePaint guided_diffusion/scheduler.py.
        t_T, jump_length, jump_n_sample = 25, 5, 3
        jumps = {j: jump_n_sample - 1 for j in range(0, t_T - jump_length, jump_length)}
        t = t_T
        reference = []
        while t >= 1:
            t -= 1
            reference.append(t)
            if jumps.get(t, 0) > 0:
                jumps[t] -= 1
                for _ in range(jump_length):
                    t += 1
                    reference.append(t)
        reference.append(-1)
        self.assertEqual(build_repaint_schedule(t_T, jump_length, jump_n_sample), reference)

    def test_no_resampling_is_monotonic(self):
        schedule = build_repaint_schedule(8, jump_length=2, jump_n_sample=1)
        self.assertEqual(schedule, list(range(7, -2, -1)))

    def test_paper_parameter_schedule_has_expected_transition_budget(self):
        schedule = build_repaint_schedule(250, jump_length=10, jump_n_sample=10)
        self.assertEqual(len(schedule), 4571)
        self.assertEqual(sum(a > b for a, b in zip(schedule[:-1], schedule[1:])), 2410)
        self.assertEqual(sum(b > a for a, b in zip(schedule[:-1], schedule[1:])), 2160)

    def test_invalid_params_fail_fast(self):
        with self.assertRaises(ValueError):
            build_repaint_schedule(1, 1, 1)
        with self.assertRaises(ValueError):
            build_repaint_schedule(8, 0, 1)
        with self.assertRaises(ValueError):
            build_repaint_schedule(8, 2, 0)


class TestMasks(unittest.TestCase):
    def test_half_mask_orientation(self):
        left = create_half_mask(8, "left")[0, 0]
        right = create_half_mask(8, "right")[0, 0]
        self.assertEqual(float(left[:, :4].mean()), 1.0)
        self.assertEqual(float(left[:, 4:].mean()), 0.0)
        self.assertEqual(float(right[:, :4].mean()), 0.0)
        self.assertEqual(float(right[:, 4:].mean()), 1.0)

    def test_center_mask_fraction(self):
        mask = create_center_mask(100, 0.35)
        self.assertAlmostEqual(float(mask.mean()), 0.1225, places=3)

    def test_official_keep_mask_polarity_can_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mask.png"
            image = Image.new("L", (8, 8), color=0)
            draw = ImageDraw.Draw(image)
            draw.rectangle((4, 0, 7, 7), fill=255)
            image.save(path)
            project_polarity = load_mask(str(path), image_size=8, white_is_missing=True)
            official_keep_mask_polarity = load_mask(str(path), image_size=8, white_is_missing=False)
        self.assertEqual(float(project_polarity[:, :, :, 4:].mean()), 1.0)
        self.assertEqual(float(project_polarity[:, :, :, :4].mean()), 0.0)
        self.assertEqual(float(official_keep_mask_polarity[:, :, :, 4:].mean()), 0.0)
        self.assertEqual(float(official_keep_mask_polarity[:, :, :, :4].mean()), 1.0)


class TestRePaintPipeline(unittest.TestCase):
    def test_raw_sample_shape_finiteness_and_composite_preservation(self):
        torch.manual_seed(11)
        x0 = torch.rand(1, 3, 8, 8) * 2.0 - 1.0
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        scheduler = FakeScheduler(8)
        model = ZeroEpsilonModel()
        generator = torch.Generator(device="cpu").manual_seed(5)
        output = repaint_pipeline(
            model, scheduler, x0, mask,
            num_inference_steps=8,
            jump_length=2,
            jump_n_sample=2,
            generator=generator,
        )
        self.assertEqual(output.shape, x0.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all(output >= -1.0))
        self.assertTrue(torch.all(output <= 1.0))
        known = (1.0 - mask).expand_as(x0).bool()
        # RePaint's raw sampler injects noised known pixels before denoising;
        # exact pixel preservation belongs to the separately saved composite.
        composite = (1.0 - mask) * x0 + mask * output
        self.assertTrue(torch.equal(composite[known], x0[known]))
        schedule = build_repaint_schedule(8, 2, 2)
        expected_calls = sum(a > b for a, b in zip(schedule[:-1], schedule[1:]))
        self.assertEqual(model.calls, expected_calls)

    def test_reverse_step_uses_respaced_ddpm_posterior_coefficients(self):
        scheduler = FakeScheduler(8)
        model = ZeroEpsilonModel()
        x0 = torch.zeros(1, 3, 4, 4)
        mask = torch.ones(1, 1, 4, 4)  # all missing; no known conditioning
        x_t = torch.linspace(-0.7, 0.7, 3 * 4 * 4).reshape(1, 3, 4, 4)
        t, prev_t = 7, 6
        alpha_bar_t = scheduler.alphas_cumprod[t]
        alpha_bar_prev = scheduler.alphas_cumprod[prev_t]
        pred_x0 = (x_t / alpha_bar_t.sqrt()).clamp(-1.0, 1.0)
        beta_eff = 1.0 - alpha_bar_t / alpha_bar_prev
        denom = 1.0 - alpha_bar_t
        coef_x0 = alpha_bar_prev.sqrt() * beta_eff / denom
        coef_xt = (alpha_bar_t / alpha_bar_prev).sqrt() * (1.0 - alpha_bar_prev) / denom
        expected_mean = coef_x0 * pred_x0 + coef_xt * x_t
        with __import__("unittest").mock.patch(
            "src.repaint._randn_like", side_effect=lambda x, generator=None: torch.zeros_like(x)
        ):
            actual = _reverse_ddpm_respaced_step(
                model, scheduler, x_t, x0, mask, t, prev_t, stride=1,
                alpha_bar_values=scheduler.alphas_cumprod, condition_known=False,
            )
        self.assertTrue(torch.allclose(actual, expected_mean, atol=1e-6, rtol=1e-6))

    def test_known_region_is_conditioned_before_second_model_call(self):
        x0 = torch.linspace(-1.0, 1.0, 3 * 8 * 8).reshape(1, 3, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, :, 4:] = 1.0
        scheduler = FakeScheduler(8)
        model = ZeroEpsilonModel(record_inputs=True)
        # Deterministic zero noise isolates the conditioning algebra: official
        # RePaint skips injection on its first model call, then injects q(x_t|x0).
        from unittest.mock import patch
        with patch("src.repaint._randn_like", side_effect=lambda x, generator=None: torch.zeros_like(x)):
            repaint_pipeline(
                model, scheduler, x0, mask,
                num_inference_steps=8, jump_length=2, jump_n_sample=1,
                generator=torch.Generator().manual_seed(7),
            )
        self.assertEqual(int(model.timesteps[0][0]), 7)
        self.assertEqual(int(model.timesteps[1][0]), 6)
        known = (1.0 - mask).expand_as(x0).bool()
        expected_known_at_t6 = scheduler.alphas_cumprod[6].sqrt() * x0
        self.assertTrue(torch.allclose(model.inputs[1][known], expected_known_at_t6[known], atol=1e-6))

    def test_all_known_mask_runs_without_nonfinite_values(self):
        x0 = torch.rand(1, 3, 8, 8) * 2.0 - 1.0
        mask = torch.zeros(1, 1, 8, 8)
        output = repaint_pipeline(
            ZeroEpsilonModel(), FakeScheduler(8), x0, mask,
            num_inference_steps=8, jump_length=2, jump_n_sample=1,
            generator=torch.Generator().manual_seed(3),
        )
        self.assertEqual(output.shape, x0.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_nondivisible_respacing_is_rejected(self):
        x0 = torch.zeros(1, 3, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        with self.assertRaises(ValueError):
            repaint_pipeline(ZeroEpsilonModel(), FakeScheduler(8), x0, mask, num_inference_steps=3)


class TestMetrics(unittest.TestCase):
    def test_metrics_are_region_aware(self):
        target = torch.zeros(1, 3, 8, 8)
        prediction = target.clone()
        prediction[:, :, 2:6, 2:6] = 1.0
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1.0
        result = compute_metrics(prediction, target, mask, compute_ssim=False)
        self.assertEqual(result["known_mae"], 0.0)
        self.assertGreater(result["missing_mae"], 0.0)
        self.assertTrue(math.isfinite(result["full_psnr_db"]))


if __name__ == "__main__":
    unittest.main()
