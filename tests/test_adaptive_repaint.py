"""Tests for the budget-aware RePaint extension and baseline regression.

Run from the repository root:
    python -m unittest discover -s tests -v
or
    python -m pytest tests -q
"""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.adaptive_repaint import (
    CountingModel,
    allocate_cycles,
    anchor_comparison_levels,
    budgeted_repaint,
    compute_instability_scores,
    make_generator,
    make_initial_latent,
    plan_budget,
    region_rms,
    run_comparison,
    run_pilot,
    uniform_allocation,
)
from src.config import find_project_root, load_project_config, resolve_project_path
from src.experiment_io import save_comparison
from src.mask import create_boundary_band, create_center_mask
from src.metrics import compute_inpainting_metrics
from src.repaint import (
    build_repaint_schedule,
    build_repaint_schedule_from_counts,
    count_schedule_transitions,
    eligible_jump_anchors,
    repaint_pipeline,
    run_repaint_schedule,
)


# ---------------------------------------------------------------------------
# Test doubles (no pretrained weights needed)
# ---------------------------------------------------------------------------
class FakeScheduler:
    """Linear-beta DDPM schedule with the attributes the sampler reads."""

    def __init__(self, num_train_timesteps=1000):
        self.betas = torch.linspace(1e-4, 0.02, num_train_timesteps, dtype=torch.float32)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps,
                                      prediction_type="epsilon", clip_sample=True)


class FakeOutput:
    def __init__(self, sample):
        self.sample = sample


class TinyDenoiser:
    """Deterministic, input-dependent fake epsilon model (never all-zero)."""

    def __init__(self):
        self.timesteps = []

    def __call__(self, sample, timesteps):
        self.timesteps.append(int(timesteps[0]))
        t = timesteps.to(sample.dtype).view(-1, 1, 1, 1) / 1000.0
        return FakeOutput(0.5 * torch.tanh(sample) + 0.1 * t * sample.roll(1, dims=-1))


def official_get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                               jump2_length=1, jump2_n_sample=1,
                               jump3_length=1, jump3_n_sample=1,
                               start_resampling=100000000):
    """Line-by-line translation of guided_diffusion/scheduler.py:get_schedule_jump
    (andreas128/RePaint @ 8e2a055), used by confs/face_example.yml."""
    jumps = {j: jump_n_sample - 1 for j in range(0, t_T - jump_length, jump_length)}
    jumps2 = {j: jump2_n_sample - 1 for j in range(0, t_T - jump2_length, jump2_length)}
    jumps3 = {j: jump3_n_sample - 1 for j in range(0, t_T - jump3_length, jump3_length)}
    t = t_T
    ts = []
    while t >= 1:
        t = t - 1
        ts.append(t)
        if t + 1 < t_T - 1 and t <= start_resampling:
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)
                if t >= 0:
                    t = t - 1
                    ts.append(t)
        if jumps3.get(t, 0) > 0 and t <= start_resampling - jump3_length:
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)
        if jumps2.get(t, 0) > 0 and t <= start_resampling - jump2_length:
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {j: jump3_n_sample - 1 for j in range(0, t_T - jump3_length, jump3_length)}
        if jumps.get(t, 0) > 0 and t <= start_resampling - jump_length:
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {j: jump2_n_sample - 1 for j in range(0, t_T - jump2_length, jump2_length)}
            jumps3 = {j: jump3_n_sample - 1 for j in range(0, t_T - jump3_length, jump3_length)}
    ts.append(-1)
    return ts


def small_problem(size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x0 = torch.rand(1, 3, size, size, generator=g) * 2 - 1
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, size // 4: 3 * size // 4, size // 4: 3 * size // 4] = 1.0
    return x0, mask


# tests/golden_fixed_repaint_309b3da.json holds values produced by the
# *unmodified* src/repaint.py at commit 309b3da for the call in
# test_golden_fixed_output_unchanged. They pin the fixed baseline's numerical
# output for the default call path (x_T drawn from the generator).


# ---------------------------------------------------------------------------
# A. Baseline regression
# ---------------------------------------------------------------------------
class TestBaselineRegression(unittest.TestCase):
    def test_reference_schedule_matches_official_face_config(self):
        ours = build_repaint_schedule(250, 10, 10)
        self.assertEqual(ours, official_get_schedule_jump(250, 1, 10, 10))
        self.assertEqual(count_schedule_transitions(ours), (2410, 2160))
        self.assertEqual(len(ours), 4571)

    def test_small_schedules_match_official(self):
        for t_T, j, r in [(25, 5, 3), (20, 5, 2), (12, 5, 4), (11, 5, 2), (30, 1, 3), (8, 2, 1)]:
            with self.subTest(t_T=t_T, j=j, r=r):
                self.assertEqual(build_repaint_schedule(t_T, j, r), official_get_schedule_jump(t_T, 1, j, r))

    def test_schedule_transitions_are_legal(self):
        s = build_repaint_schedule(250, 10, 10)
        self.assertEqual(s[0], 249)
        self.assertEqual(s[-1], -1)
        self.assertTrue(all(abs(a - b) == 1 for a, b in zip(s, s[1:])))
        self.assertTrue(all(-1 <= t <= 249 for t in s))
        self.assertTrue(all(t >= 0 for t in s[:-1]))

    def test_eligible_anchors_are_official_jump_keys(self):
        self.assertEqual(eligible_jump_anchors(250, 10), list(range(0, 240, 10)))
        self.assertEqual(len(eligible_jump_anchors(250, 10)), 24)

    def test_from_counts_generalises_fixed_schedule(self):
        anchors = eligible_jump_anchors(250, 10)
        self.assertEqual(build_repaint_schedule_from_counts(250, 10, {a: 9 for a in anchors}),
                         build_repaint_schedule(250, 10, 10))
        self.assertEqual(build_repaint_schedule_from_counts(250, 10, {}), list(range(249, -2, -1)))

    def test_golden_fixed_output_unchanged(self):
        x0, mask = small_problem(8, seed=1)
        out = repaint_pipeline(TinyDenoiser(), FakeScheduler(1000), x0, mask, jump_length=5, jump_n_sample=3,
                               generator=torch.Generator().manual_seed(123), num_inference_steps=20)
        golden = _load_golden()
        self.assertEqual(list(out.shape), golden["shape"])
        self.assertAlmostEqual(float(out.double().sum()), golden["sum"], places=4)
        self.assertAlmostEqual(float(out.double().abs().sum()), golden["abs_sum"], places=4)
        flat = out.flatten()[:8].tolist()
        for a, b in zip(flat, golden["first8"]):
            self.assertAlmostEqual(a, b, places=5)

    def test_explicit_x_T_reproduces_default_path(self):
        x0, mask = small_problem(8, seed=2)
        sched = FakeScheduler(1000)
        g = torch.Generator().manual_seed(9)
        default = repaint_pipeline(TinyDenoiser(), sched, x0, mask, 5, 2, generator=g, num_inference_steps=20)
        g2 = torch.Generator().manual_seed(9)
        x_T = torch.randn(x0.shape, generator=g2)
        explicit = repaint_pipeline(TinyDenoiser(), sched, x0, mask, 5, 2, generator=g2, num_inference_steps=20, x_T=x_T)
        self.assertTrue(torch.equal(default, explicit))

    def test_run_repaint_schedule_equals_pipeline(self):
        x0, mask = small_problem(8, seed=3)
        sched = FakeScheduler(1000)
        a = repaint_pipeline(TinyDenoiser(), sched, x0, mask, 5, 3, generator=torch.Generator().manual_seed(4),
                             num_inference_steps=20)
        b = run_repaint_schedule(TinyDenoiser(), sched, x0, mask, build_repaint_schedule(20, 5, 3), 20,
                                 generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(a, b))

    def test_invalid_schedule_rejected(self):
        x0, mask = small_problem(8)
        with self.assertRaises(ValueError):
            run_repaint_schedule(TinyDenoiser(), FakeScheduler(1000), x0, mask, [19, 17, -1], 20)
        with self.assertRaises(ValueError):
            build_repaint_schedule_from_counts(20, 5, {3: 1})  # 3 is not an anchor
        with self.assertRaises(ValueError):
            build_repaint_schedule_from_counts(20, 5, {15: 1})  # 15 + 5 would exceed T-1
        with self.assertRaises(ValueError):
            build_repaint_schedule_from_counts(20, 5, {5: -1})


def _load_golden():
    path = Path(__file__).with_name("golden_fixed_repaint_309b3da.json")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# B. Instability score
# ---------------------------------------------------------------------------
class TestInstabilityScore(unittest.TestCase):
    T, J = 20, 5

    def _estimates(self, change_fn):
        levels = sorted({l for p in anchor_comparison_levels(self.T, self.J).values() for l in p})
        return {lvl: change_fn(lvl) for lvl in levels}

    def test_region_rms_formula(self):
        diff = torch.zeros(1, 3, 4, 4)
        diff[:, :, :2] = 2.0
        region = torch.zeros(1, 1, 4, 4)
        region[:, :, :2] = 1.0
        self.assertAlmostEqual(region_rms(diff, region), 2.0)
        self.assertEqual(region_rms(diff, torch.zeros(1, 1, 4, 4)), 0.0)

    def test_comparison_levels(self):
        pairs = anchor_comparison_levels(250, 10)
        self.assertEqual(pairs[0], (1, 11))
        self.assertEqual(pairs[230], (231, 241))
        # Edge case where a + j + 1 would exceed T - 1.
        self.assertEqual(anchor_comparison_levels(11, 5)[5], (6, 10))

    def test_mask_polarity_and_known_pixels_ignored(self):
        size = 16
        mask = create_center_mask(size, 0.5)  # 1 = missing
        hole = mask.bool().expand(1, 3, size, size)

        def est(level):
            x = torch.zeros(1, 3, size, size)
            x[~hole] = 100.0 * level  # huge changes in KNOWN pixels only
            return x

        scores = compute_instability_scores(self._estimates(est), mask, self.T, self.J,
                                            lambda_boundary=0.0, normalization="none")
        self.assertTrue(all(v == 0.0 for v in scores["d_hole"]))
        self.assertTrue(all(v > 0 for v in scores["d_known"]))

        def est_hole(level):
            x = torch.zeros(1, 3, size, size)
            x[hole] = float(level)
            return x

        scores = compute_instability_scores(self._estimates(est_hole), mask, self.T, self.J,
                                            lambda_boundary=0.0, normalization="none")
        for after, before, d in zip(scores["after_level"], scores["before_level"], scores["d_hole"]):
            self.assertAlmostEqual(d, abs(after - before))
        self.assertTrue(all(v == 0.0 for v in scores["d_known"]))

    def test_inverted_mask_changes_scores(self):
        size = 16
        mask = create_center_mask(size, 0.5)
        hole = mask.bool().expand(1, 3, size, size)

        def est(level):
            x = torch.zeros(1, 3, size, size)
            x[hole] = float(level) * (level % 3 + 1)
            return x

        a = compute_instability_scores(self._estimates(est), mask, self.T, self.J, normalization="none")
        b = compute_instability_scores(self._estimates(est), 1 - mask, self.T, self.J, normalization="none")
        self.assertNotEqual(a["d_hole"], b["d_hole"])

    def test_boundary_band(self):
        mask = torch.zeros(1, 1, 12, 12)
        mask[:, :, 4:8, 4:8] = 1.0
        inner = create_boundary_band(mask, 1, "inner")
        outer = create_boundary_band(mask, 1, "outer")
        both = create_boundary_band(mask, 1, "both")
        self.assertEqual(int(inner.sum()), 16 - 4)        # 4x4 hole minus its 2x2 core
        self.assertEqual(int(outer.sum()), 36 - 16)       # 6x6 dilation minus hole
        self.assertTrue(torch.equal(both, inner + outer))
        self.assertEqual(float((inner * (1 - mask)).sum()), 0.0)
        self.assertEqual(float((outer * mask).sum()), 0.0)
        self.assertEqual(float(create_boundary_band(torch.zeros(1, 1, 8, 8), 2).sum()), 0.0)
        self.assertEqual(float(create_boundary_band(torch.ones(1, 1, 8, 8), 2).sum()), 0.0)

    def test_boundary_term_reacts_only_near_seam(self):
        size = 24
        mask = torch.zeros(1, 1, size, size)
        mask[:, :, 6:18, 6:18] = 1.0
        band = create_boundary_band(mask, 2, "both").bool().expand(1, 3, size, size)

        def est(level):
            x = torch.zeros(1, 3, size, size)
            x[band] = float(level)
            return x

        s = compute_instability_scores(self._estimates(est), mask, self.T, self.J, boundary_band_px=2,
                                       normalization="none")
        self.assertTrue(all(v > 0 for v in s["d_band"]))
        far = (1 - create_boundary_band(mask, 2, "both")).bool().expand(1, 3, size, size) & ~mask.bool().expand(1, 3, size, size)

        def est_far(level):
            x = torch.zeros(1, 3, size, size)
            x[far] = float(level)
            return x

        s2 = compute_instability_scores(self._estimates(est_far), mask, self.T, self.J, boundary_band_px=2,
                                        normalization="none")
        self.assertTrue(all(v == 0 for v in s2["d_band"]))

    def test_degenerate_masks_are_finite(self):
        rand = lambda level: torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(level))
        for mask in (torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)):
            for norm in ("known_ratio", "none"):
                s = compute_instability_scores(self._estimates(rand), mask, self.T, self.J, normalization=norm)
                for key in ("score", "hole_share", "band_share", "n_hole", "n_band"):
                    self.assertTrue(all(math.isfinite(v) and v >= 0 for v in s[key]), (key, norm))
        s = compute_instability_scores(self._estimates(rand), torch.zeros(1, 1, 8, 8), self.T, self.J)
        self.assertTrue(all(v == 0 for v in s["score"]))  # empty hole -> no information
        s = compute_instability_scores(self._estimates(rand), torch.ones(1, 1, 8, 8), self.T, self.J)
        self.assertFalse(s["normalization_applied"])      # no known pixels -> raw fallback

    def test_known_ratio_normalisation_no_nan_when_known_static(self):
        size = 16
        mask = create_center_mask(size, 0.5)
        hole = mask.bool().expand(1, 3, size, size)

        def est(level):
            x = torch.zeros(1, 3, size, size)
            x[hole] = 0.01 * level
            return x  # known region never changes -> d_known = 0

        s = compute_instability_scores(self._estimates(est), mask, self.T, self.J, normalization="known_ratio")
        self.assertTrue(s["normalization_applied"])
        self.assertTrue(all(math.isfinite(v) for v in s["n_hole"]))
        self.assertAlmostEqual(sum(s["hole_share"]), 1.0)

    def test_known_ratio_removes_shared_timestep_scale(self):
        size = 16
        mask = create_center_mask(size, 0.5)
        hole = mask.bool().expand(1, 3, size, size)

        def est(level):  # both regions drift by an amount growing with level
            x = torch.zeros(1, 3, size, size)
            x[hole] = 3.0 * level ** 2
            x[~hole] = 1.0 * level ** 2
            return x

        raw = compute_instability_scores(self._estimates(est), mask, self.T, self.J, lambda_boundary=0,
                                         normalization="none")
        norm = compute_instability_scores(self._estimates(est), mask, self.T, self.J, lambda_boundary=0,
                                          normalization="known_ratio", ratio_eps=1e-9)
        self.assertGreater(max(raw["hole_share"]) - min(raw["hole_share"]), 0.1)
        self.assertLess(max(norm["hole_share"]) - min(norm["hole_share"]), 1e-6)


# ---------------------------------------------------------------------------
# C. Allocation
# ---------------------------------------------------------------------------
class TestAllocation(unittest.TestCase):
    def test_equal_scores_deterministic(self):
        a = allocate_cycles([1, 1, 1, 1], 10)
        self.assertEqual(a, allocate_cycles([1, 1, 1, 1], 10))
        self.assertEqual(a, [3, 3, 2, 2])  # ties -> lower position first
        self.assertEqual(uniform_allocation(24, 191), [8] * 23 + [7])

    def test_higher_scores_get_more(self):
        a = allocate_cycles([1.0, 2.0, 7.0], 20)
        self.assertLessEqual(a[0], a[1])
        self.assertLessEqual(a[1], a[2])
        self.assertEqual(sum(a), 20)

    def test_integers_limits_and_exact_sum(self):
        g = torch.Generator().manual_seed(0)
        for trial in range(200):
            n = int(torch.randint(1, 30, (1,), generator=g))
            lo = int(torch.randint(0, 4, (1,), generator=g))
            hi = lo + int(torch.randint(0, 10, (1,), generator=g))
            total = int(torch.randint(n * lo, n * hi + 1, (1,), generator=g))
            scores = (torch.rand(n, generator=g) * (torch.rand(n, generator=g) > 0.3)).tolist()
            a = allocate_cycles(scores, total, lo, hi)
            self.assertEqual(sum(a), total)
            self.assertTrue(all(isinstance(v, int) for v in a))
            self.assertTrue(all(lo <= v <= hi for v in a))

    def test_cap_and_redistribution(self):
        a = allocate_cycles([100.0, 0.0, 0.0], 9, 0, 4)
        self.assertEqual(a[0], 4)
        self.assertEqual(sum(a), 9)
        self.assertEqual(a[1:], [3, 2])  # zero-score anchors share the rest equally

    def test_zero_scores_policy(self):
        self.assertEqual(allocate_cycles([0, 0, 0], 7), allocate_cycles([5, 5, 5], 7))
        self.assertEqual(allocate_cycles([0.0, 1.0, 0.0], 5, 1, None), [1, 3, 1])

    def test_infeasible(self):
        with self.assertRaises(ValueError):
            allocate_cycles([1, 1], 1, min_cycles=1)
        with self.assertRaises(ValueError):
            allocate_cycles([1, 1], 10, 0, 4)
        with self.assertRaises(ValueError):
            allocate_cycles([], 3)
        with self.assertRaises(ValueError):
            allocate_cycles([1, float("nan")], 3)
        with self.assertRaises(ValueError):
            allocate_cycles([1, -1], 3)
        self.assertEqual(allocate_cycles([], 0), [])


# ---------------------------------------------------------------------------
# D. Schedule execution
# ---------------------------------------------------------------------------
class TestScheduleExecution(unittest.TestCase):
    def test_cycles_occur_at_intended_anchor(self):
        T, J = 20, 5
        alloc = {0: 2, 5: 0, 10: 1}
        s = build_repaint_schedule_from_counts(T, J, alloc)
        starts = {}
        for i in range(1, len(s) - 1):
            if s[i + 1] > s[i] and s[i - 1] > s[i]:
                starts[s[i]] = starts.get(s[i], 0) + 1
        self.assertEqual(starts, {0: 2, 10: 1})
        self.assertEqual(count_schedule_transitions(s), (T + J * 3, J * 3))

    def test_fake_model_run_and_counts(self):
        T, J = 20, 5
        x0, mask = small_problem(8)
        alloc = dict(zip(eligible_jump_anchors(T, J), [0, 3, 1]))
        model = CountingModel(TinyDenoiser())
        out, schedule = budgeted_repaint(model, FakeScheduler(1000), x0, mask, alloc, T, J,
                                         generator=torch.Generator().manual_seed(0))
        self.assertEqual(model.evaluations, count_schedule_transitions(schedule)[0])
        self.assertEqual(model.evaluations, T + J * 4)
        self.assertEqual(out.shape, x0.shape)
        self.assertEqual(out.dtype, x0.dtype)
        self.assertTrue(torch.isfinite(out).all())

    def test_uniform_counts_reproduce_fixed_pipeline(self):
        T, J, r = 20, 5, 3
        x0, mask = small_problem(8, seed=5)
        alloc = {a: r - 1 for a in eligible_jump_anchors(T, J)}
        x_T = make_initial_latent(x0.shape, 11)
        a = repaint_pipeline(TinyDenoiser(), FakeScheduler(1000), x0, mask, J, r,
                             generator=make_generator(3, "cpu"), num_inference_steps=T, x_T=x_T)
        b, _ = budgeted_repaint(TinyDenoiser(), FakeScheduler(1000), x0, mask, alloc, T, J,
                                generator=make_generator(3, "cpu"), x_T=x_T)
        self.assertTrue(torch.equal(a, b))

    def test_transitions_identical_to_baseline_step(self):
        """The budgeted sampler calls the same reverse-step function as the baseline."""
        import src.repaint as rp
        x0, mask = small_problem(8)
        calls = []
        original = rp._reverse_ddpm_respaced_step

        def spy(*args, **kwargs):
            calls.append((kwargs["timestep"], kwargs["prev_timestep"], kwargs["condition_known"]))
            return original(*args, **kwargs)

        with mock.patch.object(rp, "_reverse_ddpm_respaced_step", side_effect=spy):
            budgeted_repaint(TinyDenoiser(), FakeScheduler(1000), x0, mask, {5: 1}, 20, 5,
                             generator=torch.Generator().manual_seed(0))
        self.assertEqual(len(calls), 25)
        self.assertFalse(calls[0][2])                       # first reverse call is unconditioned
        self.assertTrue(all(c[2] for c in calls[1:]))
        self.assertEqual(calls[-1][1], -1)                  # terminal step goes to -1
        self.assertTrue(all(t - p == 50 for t, p, _ in calls[:-1]))  # stride 1000/20


# ---------------------------------------------------------------------------
# E. Budget accounting
# ---------------------------------------------------------------------------
class TestBudgetAccounting(unittest.TestCase):
    def test_reference_plan(self):
        plan = plan_budget(250, 10, 10)
        self.assertEqual(plan.fixed_schedule_nfe, 2410)
        self.assertEqual(plan.reference_budget_nfe, 2410)
        self.assertEqual(plan.pilot_nfe, 250)
        self.assertEqual(plan.base_trajectory_nfe, 250)
        self.assertEqual(plan.nfe_per_cycle, 10)
        self.assertEqual(plan.extra_cycles, 191)
        self.assertEqual(plan.unspent_nfe, 0)
        self.assertEqual(plan.budgeted_total_nfe, 2410)

    def test_remainder(self):
        plan = plan_budget(250, 10, 10, reference_budget_nfe=2415)
        self.assertEqual(plan.extra_cycles, 191)
        self.assertEqual(plan.unspent_nfe, 5)
        with self.assertRaises(ValueError):
            plan_budget(250, 10, 10, reference_budget_nfe=400)

    def test_pilot_counts_and_records(self):
        T, J = 20, 5
        x0, mask = small_problem(8)
        model = CountingModel(TinyDenoiser())
        pilot = run_pilot(model, FakeScheduler(1000), x0, mask, make_initial_latent(x0.shape, 1), T, J,
                          make_generator(2, "cpu"))
        self.assertEqual(pilot.nfe, T)
        needed = {l for p in anchor_comparison_levels(T, J).values() for l in p}
        self.assertEqual(set(pilot.x0_estimates), needed)

    def test_forward_steps_do_not_count(self):
        model = CountingModel(TinyDenoiser())
        x0, mask = small_problem(8)
        budgeted_repaint(model, FakeScheduler(1000), x0, mask, {0: 4}, 20, 5,
                         generator=torch.Generator().manual_seed(0))
        self.assertEqual(model.evaluations, 20 + 4 * 5)  # 20 forward steps add nothing

    def test_full_reference_comparison_with_fake_model(self):
        """T=250, j=10, r=10 on a tiny image: every method's measured NFE."""
        x0, mask = small_problem(8, seed=7)
        model = CountingModel(TinyDenoiser())
        res = run_comparison(model, FakeScheduler(1000), x0, mask, 250, 10, 10, latent_seed=1, noise_seed=2,
                             min_extra_cycles=3, max_extra_cycles=14, verbose=False)
        m = res.methods
        self.assertEqual(m["fixed"].total_nfe, 2410)
        self.assertEqual(m["no_resampling"].total_nfe, 250)
        self.assertEqual(m["uniform"].pilot_nfe, 250)
        self.assertEqual(m["adaptive"].pilot_nfe, 250)
        self.assertEqual(m["uniform"].final_nfe, 2160)
        self.assertEqual(m["adaptive"].final_nfe, 2160)
        self.assertEqual(m["uniform"].total_nfe, m["adaptive"].total_nfe)
        self.assertEqual(m["adaptive"].total_nfe, 2410)
        self.assertEqual(m["uniform"].extra_cycles_total, 191)
        self.assertEqual(m["adaptive"].extra_cycles_total, 191)
        self.assertTrue(all(3 <= v <= 14 for v in m["adaptive"].allocation.values()))
        # With pilot_noise_seed == noise_seed the pilot is exactly the r=1 baseline.
        self.assertTrue(torch.equal(res.pilot.output, m["no_resampling"].raw))


# ---------------------------------------------------------------------------
# F. Results and output
# ---------------------------------------------------------------------------
class TestOutputs(unittest.TestCase):
    def test_metrics_raw_vs_composite(self):
        x0, mask = small_problem(16, seed=1)
        raw = x0.clone()
        raw[:, :, :2] += 0.5          # damage KNOWN pixels in the raw output
        raw = raw.clamp(-1, 1)
        m = compute_inpainting_metrics(raw, x0, mask, boundary_width=2, compute_ssim=False)
        self.assertGreater(m["raw_known_mae"], 0)
        self.assertEqual(m["missing_mae"], 0.0)
        self.assertEqual(m["composite_full_psnr_db"], 120.0)
        for k, v in m.items():
            if v is not None:
                self.assertTrue(math.isfinite(v), k)

    def test_save_comparison_writes_metadata(self):
        x0, mask = small_problem(16, seed=4)
        res = run_comparison(CountingModel(TinyDenoiser()), FakeScheduler(1000), x0, mask, 20, 5, 3,
                             latent_seed=1, noise_seed=2, min_extra_cycles=0, max_extra_cycles=4, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            meta = {"model_identifier": "fake", "image_id": "synthetic", "mask_type": "center"}
            paths = save_comparison(res, x0, mask, Path(tmp), meta, compute_ssim=False)
            for p in paths.values():
                self.assertTrue(Path(p).exists(), p)
            run = json.loads(Path(paths["run_json"]).read_text())
            for key in ("model_identifier", "image_id", "mask_type", "plan", "settings", "methods",
                        "mask_missing_fraction", "environment"):
                self.assertIn(key, run)
            with open(paths["metrics_csv"], newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual({r["method"] for r in rows}, {"fixed", "no_resampling", "uniform", "adaptive"})
            for r in rows:
                self.assertIn("total_nfe", r)
                self.assertIn("missing_psnr_db", r)
            with open(paths["allocation_csv"], newline="") as f:
                arows = list(csv.DictReader(f))
            self.assertEqual(sum(int(r["adaptive_extra_cycles"]) for r in arows), res.plan.extra_cycles)


class TestProjectPaths(unittest.TestCase):
    def test_project_root_and_config(self):
        root = find_project_root(Path(__file__).parent)
        self.assertTrue((root / "src" / "repaint.py").exists())
        cfg = load_project_config(project_root=root)
        self.assertIn("budget_aware", cfg)
        self.assertEqual(resolve_project_path("configs/config.yaml", root), root / "configs" / "config.yaml")
        with self.assertRaises(FileNotFoundError):
            resolve_project_path("does/not/exist.yaml", root, must_exist=True)
        with self.assertRaises(FileNotFoundError):
            load_project_config("configs/missing.yaml", project_root=root)


if __name__ == "__main__":
    unittest.main()
