"""Tests for the faces / Places2 mask and LPIPS study (additions only).

Run from the repository root:  python -m pytest tests -q
The official-guided-diffusion test runs only if a RePaint checkout is found
(env REPAINT_OFFICIAL_CODE_DIR, or external/RePaint); otherwise it is skipped.
The real-LPIPS test uses a RANDOM backbone (no download) and only checks the
plumbing; its values are not valid LPIPS.
"""
import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from src.adaptive_repaint import plan_budget
from src.config import find_project_root
from src.faces_places2_study import (
    budget_settings, check_budget, collect_results, estimate_study_cost, executed_nfe_per_run,
    load_study_config, mask_difficulty_ranking, paired_differences, run_study, write_study_summaries,
)
from src.guided_diffusion_adapter import (
    GuidedDiffusionEpsAdapter, _official_linear_alphas_cumprod, check_model_compatibility,
    make_linear_ddpm_scheduler,
)
from src.lpips_metric import LPIPSEvaluator, LPIPSUnavailableError, load_lpips, lpips_status
from src.mask_families import (
    MASK_FAMILIES, coverage_table, generate_mask, is_applicable, mask_conditions, mask_seed,
)
from src.study_datasets import (
    center_crop_arr, dataset_images, list_images, load_study_image, resolve_data_dir, select_images,
)

PROJECT_ROOT = find_project_root(Path(__file__).parent)


class FakeOutput:
    def __init__(self, sample):
        self.sample = sample


class TinyDenoiser:
    def __call__(self, sample, timesteps):
        t = timesteps.to(sample.dtype).view(-1, 1, 1, 1) / 1000.0
        return FakeOutput(0.5 * torch.tanh(sample) + 0.1 * t * sample.roll(1, dims=-1))

    def parameters(self):
        return iter(())


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
class TestMaskFamilies(unittest.TestCase):
    def test_all_conditions_hit_target_coverage(self):
        rows = coverage_table(size=256, image_indices=range(3), base_seed=7)
        self.assertEqual(len(rows), 35 * 3)  # 9 families x 4 ratios - 1 not applicable
        for r in rows:
            self.assertLessEqual(abs(r["mask_coverage_error"]), 0.0105, r)

    def test_binary_shape_polarity(self):
        for family in MASK_FAMILIES:
            spec = generate_mask(family, 0.4, 64, 0, 1)
            m = spec.mask
            self.assertEqual(tuple(m.shape), (1, 1, 64, 64))
            self.assertEqual(m.dtype, torch.float32)
            self.assertTrue(torch.all((m == 0) | (m == 1)))
            self.assertAlmostEqual(spec.actual_ratio, float(m.mean()))
        center = generate_mask("center_square", 0.25, 64).mask[0, 0]
        self.assertEqual(float(center[32, 32]), 1.0)   # 1 = missing at the centre
        self.assertEqual(float(center[0, 0]), 0.0)
        border = generate_mask("border_frame", 0.25, 64).mask[0, 0]
        self.assertEqual(float(border[0, 0]), 1.0)
        self.assertEqual(float(border[32, 32]), 0.0)

    def test_stripe_orientation_and_blocks_grid(self):
        h = generate_mask("stripes_horizontal", 0.25, 64).mask[0, 0]
        v = generate_mask("stripes_vertical", 0.25, 64).mask[0, 0]
        self.assertTrue(torch.all(h == h[:, :1]))   # rows constant
        self.assertTrue(torch.all(v == v[:1, :]))   # columns constant
        b = generate_mask("random_blocks", 0.25, 64).mask[0, 0]
        cells = b.reshape(16, 4, 16, 4).permute(0, 2, 1, 3).reshape(256, 16)
        self.assertTrue(torch.all((cells.sum(1) == 0) | (cells.sum(1) == 16)))
        self.assertEqual(int(cells.sum(1).gt(0).sum()), 64)

    def test_reproducible_and_dataset_independent(self):
        a = generate_mask("irregular_brush", 0.4, 128, image_index=3, base_seed=2026)
        b = generate_mask("irregular_brush", 0.4, 128, image_index=3, base_seed=2026)
        c = generate_mask("irregular_brush", 0.4, 128, image_index=4, base_seed=2026)
        self.assertTrue(torch.equal(a.mask, b.mask))
        self.assertFalse(torch.equal(a.mask, c.mask))
        self.assertEqual(mask_seed("center_square", 0.1, 0, 0), mask_seed("center_square", 0.1, 0, 0))
        self.assertEqual(mask_seed("irregular_brush", 0.4, 3, 2026), a.seed)

    def test_applicability_and_errors(self):
        self.assertFalse(is_applicable("large_rectangle", 0.10))
        self.assertTrue(is_applicable("large_rectangle", 0.25))
        self.assertEqual(len(mask_conditions()), 35)
        with self.assertRaises(ValueError):
            generate_mask("not_a_mask", 0.1)
        with self.assertRaises(ValueError):
            generate_mask("center_square", 1.0)
        with self.assertRaises(ValueError):
            mask_conditions(["oops"])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class TestStudyDatasets(unittest.TestCase):
    def test_center_crop_no_stretch_and_identity(self):
        arr = np.zeros((200, 300, 3), np.uint8)
        arr[:, 150:] = 255
        out = center_crop_arr(Image.fromarray(arr), 128)
        self.assertEqual(out.shape, (128, 128, 3))
        self.assertGreater(out[:, :60].mean(), -1)  # crop exists
        with tempfile.TemporaryDirectory() as tmp:
            img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
            Image.fromarray(img).save(Path(tmp) / "a.png")
            x = load_study_image(Path(tmp) / "a.png", 64, "center_crop")
            self.assertEqual(tuple(x.shape), (1, 3, 64, 64))
            self.assertTrue(torch.allclose((x[0].permute(1, 2, 0) + 1) * 127.5, torch.from_numpy(img).float(), atol=1e-3))
            y = load_study_image(Path(tmp) / "a.png", 32, "project_resize")
            self.assertEqual(tuple(y.shape), (1, 3, 32, 32))
            self.assertTrue(float(y.min()) >= -1 and float(y.max()) <= 1)
            with self.assertRaises(ValueError):
                load_study_image(Path(tmp) / "a.png", 32, "bogus")

    def test_listing_selection_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "b").mkdir()
            for name in ("z.png", "a.jpg", "b/c.png", "notes.txt"):
                p = root / name
                if p.suffix == ".txt":
                    p.write_text("x")
                else:
                    Image.new("RGB", (8, 8)).save(p)
            paths = list_images(root)
            self.assertEqual([str(p.relative_to(root)) for p in paths], ["a.jpg", "b/c.png", "z.png"])
            (root / "list.txt").write_text("z.png\nb/c.png\n")
            self.assertEqual([p.name for p in list_images(root, file_list=root / "list.txt")], ["z.png", "c.png"])
            self.assertEqual(select_images(paths, 2), paths[:2])
            s1 = select_images(paths, 2, "seeded_random", 3)
            self.assertEqual(s1, select_images(paths, 2, "seeded_random", 3))
            self.assertEqual(resolve_data_dir(["missing", tmp], Path("/")), root.resolve())
            with self.assertRaises(FileNotFoundError):
                dataset_images({"name": "x", "data_dirs": ["/no/such/dir"]}, Path("/"))
            with self.assertRaises(FileNotFoundError):
                list_images(root / "nope")


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------
class TestLPIPS(unittest.TestCase):
    def test_unavailable_package_is_reported(self):
        with mock.patch.dict(sys.modules, {"lpips": None}):
            with self.assertRaises(LPIPSUnavailableError) as ctx:
                load_lpips()
        self.assertIn("lpips", str(ctx.exception))
        status = lpips_status(None, ctx.exception)
        self.assertEqual(status["lpips_status"], "unavailable")
        self.assertFalse(status["lpips_valid"])
        self.assertIn("LPIPSUnavailableError", status["lpips_error"])

    def test_masked_average_math(self):
        def spatial(a, b):
            m = torch.zeros(1, 1, 4, 4)
            m[..., :2, :] = 2.0
            return m

        ev = LPIPSEvaluator(full_fn=lambda a, b: torch.tensor([[[[0.5]]]]), spatial_fn=spatial, info={})
        x = torch.zeros(1, 3, 4, 4)
        top = torch.zeros(1, 1, 4, 4)
        top[..., :2, :] = 1
        self.assertAlmostEqual(ev.full(x, x), 0.5)
        self.assertAlmostEqual(ev.masked(x, x, top), 2.0)
        self.assertAlmostEqual(ev.masked(x, x, 1 - top), 0.0)
        self.assertIsNone(ev.masked(x, x, torch.zeros(1, 1, 4, 4)))

    @unittest.skipIf(__import__("importlib").util.find_spec("lpips") is None, "lpips not installed")
    def test_real_lpips_network_plumbing_random_backbone(self):
        ev = load_lpips(random_backbone=True)
        self.assertEqual(lpips_status(ev)["lpips_status"], "invalid_test_backbone")
        g = torch.Generator().manual_seed(0)
        x = torch.rand(1, 3, 64, 64, generator=g) * 2 - 1
        y = x.clone()
        y[..., 16:48, 16:48] = 0.0
        self.assertAlmostEqual(ev.full(x, x), 0.0, places=6)
        full = ev.full(x, y)
        spatial_mean = float(ev.spatial_fn(x, y).mean())
        self.assertAlmostEqual(full, spatial_mean, places=4)  # spatial map averages to the scalar
        hole = torch.zeros(1, 1, 64, 64)
        hole[..., 16:48, 16:48] = 1
        self.assertGreater(ev.masked(x, y, hole), ev.masked(x, y, 1 - hole))


# ---------------------------------------------------------------------------
# Guided-diffusion adapter
# ---------------------------------------------------------------------------
def _official_code_dir():
    for candidate in (os.environ.get("REPAINT_OFFICIAL_CODE_DIR"), PROJECT_ROOT / "external" / "RePaint"):
        if candidate and (Path(candidate) / "guided_diffusion" / "unet.py").is_file():
            return Path(candidate)
    return None


class TestGuidedDiffusionAdapter(unittest.TestCase):
    def test_adapter_returns_epsilon_channels(self):
        net = lambda x, t: torch.cat([x * 2, x * 0 - 7], dim=1)  # [eps, variance]
        ad = GuidedDiffusionEpsAdapter(net, learn_sigma=True)
        x = torch.randn(1, 3, 8, 8)
        self.assertTrue(torch.equal(ad(x, torch.tensor([5])).sample, x * 2))
        self.assertTrue(ad.learned_variance_discarded)
        bad = GuidedDiffusionEpsAdapter(lambda x, t: x, learn_sigma=True)
        with self.assertRaises(ValueError):
            bad(x, torch.tensor([5]))

    def test_scheduler_matches_guided_diffusion_linear(self):
        s = make_linear_ddpm_scheduler(1000)
        diff = (s.alphas_cumprod.double() - _official_linear_alphas_cumprod(1000)).abs().max()
        self.assertLess(float(diff), 1e-5)
        self.assertEqual(s.config.prediction_type, "epsilon")
        self.assertTrue(s.config.clip_sample)

    @unittest.skipIf(_official_code_dir() is None, "official RePaint checkout not available")
    def test_official_unet_checkpoint_roundtrip_and_sampler(self):
        from src.adaptive_repaint import CountingModel, run_comparison
        from src.guided_diffusion_adapter import build_guided_diffusion_unet, load_repaint_checkpoint

        tiny = dict(image_size=64, num_channels=32, num_res_blocks=1, channel_mult="1,2", learn_sigma=True,
                    class_cond=False, use_checkpoint=False, attention_resolutions="16", num_heads=1,
                    num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
                    resblock_updown=True, use_fp16=False, use_new_attention_order=False)
        torch.manual_seed(0)
        net = build_guided_diffusion_unet(_official_code_dir(), tiny)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "tiny.pt"
            torch.save(net.state_dict(), ckpt)
            model, sched, info = load_repaint_checkpoint(ckpt, _official_code_dir(), tiny)
        self.assertTrue(info["learned_variance_discarded"])
        report = check_model_compatibility(model, sched, 64, 20, n_timing_calls=0)
        self.assertTrue(report["compatible"], report)
        x0 = torch.rand(1, 3, 64, 64) * 2 - 1
        mask = generate_mask("center_square", 0.25, 64).mask
        res = run_comparison(CountingModel(model), sched, x0, mask, 20, 5, 3, latent_seed=1, noise_seed=2,
                             min_extra_cycles=0, max_extra_cycles=2, verbose=False)
        self.assertEqual(res.methods["fixed"].total_nfe, 50)
        self.assertEqual(res.methods["uniform"].total_nfe, res.methods["adaptive"].total_nfe)
        self.assertTrue(all(torch.isfinite(m.raw).all() for m in res.methods.values()))


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------
def _study_cfg(image_size=32):
    return {
        "study": {"image_size": image_size, "mask_base_seed": 5, "latent_seed": 42, "noise_seed": 43,
                  "boundary_band_px": 2, "require_lpips": True,
                  "methods": ["no_resampling", "fixed", "uniform", "adaptive"]},
        "masks": {"families": ["center_square", "irregular_brush"], "target_ratios": [0.25, 0.4],
                  "not_applicable": {}, "coverage_tolerance": 0.02},
        "adaptive": {"lambda_boundary": 0.5, "boundary_band_px": 2, "normalization": "known_ratio", "ratio_eps": 1e-3},
    }


def _fake_lpips():
    def full(a, b):
        return (a - b).abs().mean().view(1, 1, 1, 1)

    def spatial(a, b):
        return (a - b).abs().mean(1, keepdim=True)

    ev = LPIPSEvaluator(full_fn=full, spatial_fn=spatial, info={"lpips_net": "fake", "lpips_valid": True})
    return ev, {"lpips_status": "ok", "lpips_valid": True, "lpips_net": "fake"}


class TestDatasetModelLoading(unittest.TestCase):
    @unittest.skipIf(_official_code_dir() is None, "official RePaint checkout not available")
    def test_places2_config_path_loads_checkpoint_and_runs(self):
        from src.faces_places2_study import load_dataset_model
        from src.guided_diffusion_adapter import build_guided_diffusion_unet

        tiny = dict(image_size=64, num_channels=32, num_res_blocks=1, channel_mult="1,2", learn_sigma=True,
                    class_cond=False, use_checkpoint=False, attention_resolutions="16", num_heads=1,
                    num_head_channels=-1, num_heads_upsample=-1, use_scale_shift_norm=True, dropout=0.0,
                    resblock_updown=True, use_fp16=False, use_new_attention_order=False)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.manual_seed(1)
            torch.save(build_guided_diffusion_unet(_official_code_dir(), tiny).state_dict(), tmp / "places_tiny.pt")
            img_dir = tmp / "p256"
            img_dir.mkdir()
            for i in range(2):
                Image.fromarray((np.random.default_rng(i).random((80, 96, 3)) * 255).astype(np.uint8)).save(img_dir / f"{i}.jpg")
            dcfg = {"name": "places2", "backend": "repaint_guided_diffusion",
                    "checkpoint_candidates": ["missing.pt", str(tmp / "places_tiny.pt")],
                    "official_code_dirs": ["missing_dir", str(_official_code_dir())], "model_config": tiny,
                    "data_dirs": ["missing", str(img_dir)], "preprocessing": "center_crop", "max_images": 1}
            model, scheduler, info = load_dataset_model(dcfg, tmp, "cpu")
            self.assertEqual(info["checkpoint"], str((tmp / "places_tiny.pt").resolve()))
            self.assertTrue(info["learned_variance_discarded"])
            images = dataset_images(dcfg, tmp)
            self.assertEqual([p.name for p in images], ["0.jpg"])
            cfg = _study_cfg(64)
            cfg["masks"]["families"] = ["random_blocks"]
            cfg["masks"]["target_ratios"] = [0.25]
            ev, meta = _fake_lpips()
            datasets = {"places2": {"model": model, "scheduler": scheduler, "model_info": info, "config": dcfg,
                                    "images": images, "image_root": img_dir}}
            budget = {"profile": "t", "num_inference_steps": 20, "jump_length": 5, "jump_n_sample": 3,
                      "min_extra_cycles": 0, "max_extra_cycles": 2}
            run_study(datasets, cfg, budget, tmp / "out", "cpu", ev, meta, compute_ssim=False, log=lambda *a: None)
            df = collect_results(tmp / "out")
            self.assertEqual(set(df.method), {"no_resampling", "fixed", "uniform", "adaptive"})
            self.assertTrue((df.model_backend == "repaint_guided_diffusion").all())
            self.assertTrue(df.learned_variance_discarded.all())


class TestStudyRunner(unittest.TestCase):
    def test_config_budgets_feasible_and_costs(self):
        cfg = load_study_config(PROJECT_ROOT)
        for profile in ("smoke", "reduced", "paper"):
            check_budget(budget_settings(cfg, profile))
        paper = plan_budget(250, 10, 10)
        self.assertEqual(executed_nfe_per_run(paper, ["no_resampling", "fixed", "uniform", "adaptive"]),
                         2410 + 250 + 250 + 2 * 2160)
        rows = estimate_study_cost({"faces": 5}, 35, budget_settings(cfg, "paper"),
                                   ["no_resampling", "fixed", "uniform", "adaptive"], {"faces": 0.1})
        self.assertEqual(rows[0]["total_nfe"], 5 * 35 * 7230)
        self.assertAlmostEqual(rows[0]["estimated_hours"], 5 * 35 * 7230 * 0.1 / 3600)
        with self.assertRaises(ValueError):
            check_budget({"profile": "bad", "num_inference_steps": 250, "jump_length": 10, "jump_n_sample": 10,
                          "min_extra_cycles": 9, "max_extra_cycles": 14})

    def _datasets(self, tmp: Path, image_size: int):
        datasets = {}
        for d, name in enumerate(("faces", "places2")):
            root = tmp / name
            root.mkdir()
            images = []
            for i in range(2):
                arr = (np.random.default_rng(10 * d + i).random((image_size, image_size, 3)) * 255).astype(np.uint8)
                p = root / f"img_{i}.png"
                Image.fromarray(arr).save(p)
                images.append(p)
            datasets[name] = {
                "model": TinyDenoiser(), "scheduler": make_linear_ddpm_scheduler(1000),
                "model_info": {"backend": "fake", "model_identifier": f"fake-{name}", "learned_variance_discarded": False},
                "config": {"preprocessing": "center_crop" if name == "places2" else "project_resize"},
                "images": images, "image_root": root,
            }
        return datasets

    def test_end_to_end_resume_and_summaries(self):
        cfg = _study_cfg(32)
        budget = {"profile": "test", "num_inference_steps": 20, "jump_length": 5, "jump_n_sample": 3,
                  "min_extra_cycles": 0, "max_extra_cycles": 2}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            datasets = self._datasets(tmp, 32)
            out = tmp / "results"
            with self.assertRaises(RuntimeError):  # LPIPS required but missing
                run_study(datasets, cfg, budget, out, "cpu", None, lpips_status(None), log=lambda *a: None)
            ev, meta = _fake_lpips()
            manifest = run_study(datasets, cfg, budget, out, "cpu", ev, meta, compute_ssim=False, log=lambda *a: None)
            self.assertEqual(manifest["completed_this_call"], 2 * 4 * 2)
            again = run_study(datasets, cfg, budget, out, "cpu", ev, meta, compute_ssim=False, log=lambda *a: None)
            self.assertEqual(again["completed_this_call"], 0)
            self.assertEqual(again["skipped_existing"], 16)
            with self.assertRaises(RuntimeError):  # different settings into the same folder
                run_study(datasets, cfg, dict(budget, max_extra_cycles=3), out, "cpu", ev, meta,
                          compute_ssim=False, log=lambda *a: None)
            df = collect_results(out)
            self.assertEqual(len(df), 16 * 4)
            for (_, _, _, _), g in df.groupby(["dataset", "mask_family", "mask_target_ratio", "image_id"]):
                nfe = dict(zip(g.method, g.total_nfe))
                self.assertEqual(nfe["uniform"], nfe["adaptive"])
                self.assertEqual(nfe["fixed"], 50)
            self.assertTrue(df["lpips_full"].notna().all())
            self.assertTrue(df["lpips_missing_masked"].notna().all())
            for col in ("mask_actual_ratio", "mask_seed", "latent_seed", "noise_seed", "total_wall_time_s",
                        "model_identifier", "lpips_status", "preprocessing"):
                self.assertIn(col, df.columns)
            # Same mask for the same image index in both datasets.
            for family in ("center_square", "irregular_brush"):
                a = np.asarray(Image.open(out / "faces" / family / "ratio_0.40" / "img_1" / "images" / "mask_missing_white.png"))
                b = np.asarray(Image.open(out / "places2" / family / "ratio_0.40" / "img_1" / "images" / "mask_missing_white.png"))
                self.assertTrue(np.array_equal(a, b))
            paths = write_study_summaries(out)
            for name in ("per_image_results.csv", "summary_by_condition.csv", "paired_adaptive_vs_uniform.csv",
                         "paired_adaptive_vs_fixed.csv", "mask_difficulty_ranking.csv", "relative_to_no_resampling.csv"):
                self.assertIn(name, paths)
                self.assertTrue(Path(paths[name]).exists())
            paired = paired_differences(df, "adaptive", "uniform", ["lpips_full"])
            self.assertTrue((paired.n_pairs == 2).all())
            ranks = mask_difficulty_ranking(df, "fixed", "lpips_full")
            self.assertEqual(set(ranks.dataset), {"faces", "places2"})
            self.assertEqual(sorted(ranks[ranks.dataset == "faces"].rank_most_challenging), [1, 2])

    def test_max_runs_stops_cleanly(self):
        cfg = _study_cfg(32)
        budget = {"profile": "test", "num_inference_steps": 20, "jump_length": 5, "jump_n_sample": 3,
                  "min_extra_cycles": 0, "max_extra_cycles": 2}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ev, meta = _fake_lpips()
            m = run_study(self._datasets(tmp, 32), cfg, budget, tmp / "r", "cpu", ev, meta, max_runs=3,
                          compute_ssim=False, log=lambda *a: None)
            self.assertEqual(m["completed_this_call"], 3)
            self.assertIn("max_runs", m["stopped_reason"])


if __name__ == "__main__":
    unittest.main()
