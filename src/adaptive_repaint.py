"""Pilot-Guided, Mask-Aware Budgeted RePaint (research extension).

This module changes only *where* RePaint spends its extra resampling cycles.
Every transition (reverse posterior step, known-region injection, forward
re-noising, first-step and terminal handling) is executed by the shared loop
in :mod:`src.repaint`, so the fixed baseline and the budgeted samplers use the
same mathematics.

Procedure for one image (batch size 1):

1. Pilot: run the monotone schedule (no extra cycles, ``T`` model calls) from
   the shared initial latent and record the model's clean-image estimate
   ``x0_hat`` after selected reverse calls.
2. Score: for each legal jump anchor ``a`` (0, j, 2j, ... as in the official
   scheduler) measure how much ``x0_hat`` changed across the window of levels
   ``(a, a + j]`` that a resampling cycle at ``a`` would revisit, inside the
   hole and in a band around the hole boundary. This is a heuristic
   trajectory-instability proxy, not a calibrated uncertainty estimate.
3. Allocate: distribute an exact integer number of extra cycles over the
   anchors in proportion to the score, within per-anchor limits.
4. Execute: build the official jump schedule with these per-anchor counts and
   run it with the shared transitions.

Compute is accounted in denoiser evaluations (NFE), counted by wrapping the
model. The pilot's NFE are charged to the budget.

Mask convention: 1 = missing, 0 = known. Images in [-1, 1].
See Docs/budget_aware_repaint_methods.md for the rationale and references.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from src.mask import create_boundary_band
from src.repaint import (
    _initial_latent,
    _prepare_sampling_inputs,
    build_repaint_schedule,
    build_repaint_schedule_from_counts,
    count_schedule_transitions,
    eligible_jump_anchors,
    repaint_pipeline,
    run_repaint_schedule,
)

METHOD_NAMES = ("fixed", "no_resampling", "uniform", "adaptive")
NORMALIZATIONS = ("known_ratio", "none")


# ---------------------------------------------------------------------------
# NFE accounting
# ---------------------------------------------------------------------------
class CountingModel:
    """Transparent wrapper that counts denoiser evaluations (NFE).

    One call with a batch of images counts as one evaluation per sample in the
    batch (``count_per_sample=True``, the default) so budgets stay comparable
    if batch sizes ever differ. All budgeted samplers here use batch size 1.
    """

    def __init__(self, model, count_per_sample: bool = True):
        self.model = model
        self.count_per_sample = count_per_sample
        self.calls = 0          # number of forward passes
        self.evaluations = 0    # NFE (per-sample evaluations)

    def __call__(self, sample, timesteps, *args, **kwargs):
        self.calls += 1
        self.evaluations += int(sample.shape[0]) if self.count_per_sample else 1
        return self.model(sample, timesteps, *args, **kwargs)

    def reset(self) -> None:
        self.calls = 0
        self.evaluations = 0

    def __getattr__(self, name):  # expose .config, .dtype, etc. of the wrapped model
        return getattr(self.model, name)


@dataclass
class BudgetPlan:
    """How a total NFE budget is split. All counts come from built schedules."""

    num_inference_steps: int
    jump_length: int
    reference_jump_n_sample: int
    reference_budget_nfe: int
    fixed_schedule_nfe: int
    pilot_nfe: int
    base_trajectory_nfe: int
    nfe_per_cycle: int
    num_anchors: int
    extra_cycles: int
    unspent_nfe: int
    budgeted_total_nfe: int

    def as_dict(self) -> dict:
        return asdict(self)


def schedule_nfe(schedule: Sequence[int]) -> int:
    return count_schedule_transitions(list(schedule))[0]


def plan_budget(
    num_inference_steps: int,
    jump_length: int,
    reference_jump_n_sample: int,
    reference_budget_nfe: Optional[int] = None,
) -> BudgetPlan:
    """Split a total NFE budget into pilot + base trajectory + extra cycles.

    ``reference_budget_nfe=None`` uses the NFE of the fixed RePaint schedule
    with ``reference_jump_n_sample`` (2410 for T=250, j=10, r=10). The cost of
    one extra cycle is measured by building a schedule with a single extra
    cycle, not assumed. A remainder that cannot buy a whole cycle is left
    unspent and reported; both budgeted methods leave the same remainder.
    """
    fixed_nfe = schedule_nfe(build_repaint_schedule(num_inference_steps, jump_length, reference_jump_n_sample))
    reference = fixed_nfe if reference_budget_nfe is None else int(reference_budget_nfe)
    base_nfe = schedule_nfe(build_repaint_schedule_from_counts(num_inference_steps, jump_length, {}))
    pilot_nfe = base_nfe  # the pilot is the monotone schedule
    anchors = eligible_jump_anchors(num_inference_steps, jump_length)
    if not anchors:
        raise ValueError(
            f"T={num_inference_steps}, jump_length={jump_length} has no legal jump anchors; "
            "there are no resampling cycles to allocate."
        )
    per_cycle = {
        a: schedule_nfe(build_repaint_schedule_from_counts(num_inference_steps, jump_length, {a: 1})) - base_nfe
        for a in anchors
    }
    if len(set(per_cycle.values())) != 1:
        raise RuntimeError(f"Extra cycles do not have a uniform cost: {per_cycle}")
    nfe_per_cycle = next(iter(per_cycle.values()))
    available = reference - pilot_nfe - base_nfe
    if available < 0:
        raise ValueError(
            f"Budget {reference} NFE is smaller than pilot ({pilot_nfe}) + base trajectory ({base_nfe})."
        )
    cycles, unspent = divmod(available, nfe_per_cycle)
    return BudgetPlan(
        num_inference_steps=num_inference_steps,
        jump_length=jump_length,
        reference_jump_n_sample=reference_jump_n_sample,
        reference_budget_nfe=reference,
        fixed_schedule_nfe=fixed_nfe,
        pilot_nfe=pilot_nfe,
        base_trajectory_nfe=base_nfe,
        nfe_per_cycle=nfe_per_cycle,
        num_anchors=len(anchors),
        extra_cycles=int(cycles),
        unspent_nfe=int(unspent),
        budgeted_total_nfe=pilot_nfe + base_nfe + int(cycles) * nfe_per_cycle,
    )


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
def make_initial_latent(shape: Sequence[int], seed: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Draw x_T on the CPU so it is identical on every device. Shared by all methods."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=generator, dtype=dtype)


def make_generator(seed: int, device) -> torch.Generator:
    """A fresh sampler-noise generator. Each sampler pass gets its own."""
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


# ---------------------------------------------------------------------------
# Pilot pass
# ---------------------------------------------------------------------------
def anchor_comparison_levels(num_inference_steps: int, jump_length: int) -> Dict[int, Tuple[int, int]]:
    """For each anchor ``a``: the model-call levels whose x0 estimates are compared.

    * ``after`` = ``a + 1``: the reverse call that produces ``x_a`` (arrival at
      the anchor).
    * ``before`` = ``a + j + 1``: the call that produces ``x_{a+j}`` (arrival
      at the top of the window a cycle at ``a`` would revisit). If that level
      does not exist (``a + j + 1 > T - 1``) the highest level ``T - 1`` is used.

    The two calls bracket exactly the ``j`` reverse calls (levels ``a+j`` ...
    ``a+1``) that one extra cycle at ``a`` repeats.
    """
    top = num_inference_steps - 1
    return {
        a: (a + 1, min(a + jump_length + 1, top))
        for a in eligible_jump_anchors(num_inference_steps, jump_length)
    }


@dataclass
class PilotResult:
    output: torch.Tensor                     # raw pilot sample (device of x_0)
    x0_estimates: Dict[int, torch.Tensor]    # model-call level -> x0_hat (CPU, float32)
    nfe: int
    wall_time_s: float
    schedule_length: int


def run_pilot(
    model,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    x_T: torch.Tensor,
    num_inference_steps: int,
    jump_length: int,
    generator: Optional[torch.Generator],
) -> PilotResult:
    """Monotone RePaint pass (no extra cycles) that records x0 estimates.

    ``model`` should be a :class:`CountingModel` if NFE must be measured; the
    measured value is returned in ``PilotResult.nfe`` either way when a
    counter is present (otherwise it is the schedule's reverse count).
    """
    needed = sorted({lvl for pair in anchor_comparison_levels(num_inference_steps, jump_length).values() for lvl in pair})
    needed_set = set(needed)
    estimates: Dict[int, torch.Tensor] = {}

    def record(level: int, next_level: int, pred_x0: torch.Tensor) -> None:
        if level in needed_set:
            if level in estimates:
                raise RuntimeError(f"Level {level} visited twice in a monotone pilot")
            estimates[level] = pred_x0.detach().to(device="cpu", dtype=torch.float32).clone()

    schedule = build_repaint_schedule_from_counts(num_inference_steps, jump_length, {})
    before = getattr(model, "evaluations", None)
    start = _sync_time(x_0.device)
    output = run_repaint_schedule(
        model, scheduler, x_0, mask, schedule, num_inference_steps,
        generator=generator, x_T=x_T, on_reverse_step=record,
    )
    wall = _sync_time(x_0.device) - start
    nfe = (model.evaluations - before) if before is not None else schedule_nfe(schedule)
    missing = needed_set - set(estimates)
    if missing:
        raise RuntimeError(f"Pilot did not record x0 estimates for levels {sorted(missing)}")
    return PilotResult(output=output, x0_estimates=estimates, nfe=int(nfe), wall_time_s=wall,
                       schedule_length=len(schedule))


# ---------------------------------------------------------------------------
# Mask-aware trajectory-instability score
# ---------------------------------------------------------------------------
def region_rms(difference: torch.Tensor, region: torch.Tensor) -> float:
    """sqrt( sum(R * diff^2) / (C * sum(R)) ); 0.0 for an empty region.

    ``difference`` is [1, C, H, W]; ``region`` is a binary [1, 1, H, W] map.
    """
    region = region.to(dtype=difference.dtype, device=difference.device)
    channels = difference.shape[1]
    count = float(region.sum().item()) * channels
    if count == 0.0:
        return 0.0
    value = float((difference.square() * region).sum().item()) / count
    return math.sqrt(max(value, 0.0))


def compute_instability_scores(
    x0_estimates: Mapping[int, torch.Tensor],
    mask: torch.Tensor,
    num_inference_steps: int,
    jump_length: int,
    lambda_boundary: float = 0.5,
    boundary_band_px: int = 8,
    normalization: str = "known_ratio",
    ratio_eps: float = 1e-3,
) -> Dict[str, list]:
    """Per-anchor trajectory-instability scores from pilot x0 estimates.

    For anchor ``a`` with comparison levels ``(after, before)`` and
    ``D = x0_hat[after] - x0_hat[before]``:

    * ``d_hole``  = RMS of D over missing pixels (M = 1);
    * ``d_band``  = RMS of D over the boundary ring (both sides of the seam);
    * ``d_known`` = RMS of D over known pixels.

    ``normalization="known_ratio"`` divides ``d_hole`` and ``d_band`` by
    ``d_known + ratio_eps``. The known region is denoised by the same network
    at the same noise levels, so this removes the shared dependence of the
    drift magnitude on the timestep (early, noisy anchors otherwise dominate).
    With no known pixels the ratio is undefined and the raw values are used
    (``normalization_applied`` is then False). ``"none"`` keeps raw RMS.

    Each component is then divided by its sum over anchors, and
    ``score = hole_share + lambda_boundary * band_share``. Scores are finite
    and non-negative; all-zero scores mean "no information" and lead to a
    uniform allocation.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}")
    if lambda_boundary < 0 or not math.isfinite(lambda_boundary):
        raise ValueError("lambda_boundary must be a finite non-negative number")
    if ratio_eps <= 0:
        raise ValueError("ratio_eps must be positive")
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError("mask must have shape [1, 1, H, W] (budgeted sampling uses batch size 1)")

    missing = (mask.detach().to("cpu", torch.float32) > 0.5).float()
    known = 1.0 - missing
    band = create_boundary_band(missing, boundary_band_px, side="both")
    has_known = float(known.sum().item()) > 0
    apply_ratio = normalization == "known_ratio" and has_known

    pairs = anchor_comparison_levels(num_inference_steps, jump_length)
    rows = {k: [] for k in ("anchor", "after_level", "before_level",
                            "d_hole", "d_band", "d_known", "n_hole", "n_band")}
    for anchor, (after, before) in pairs.items():
        if after not in x0_estimates or before not in x0_estimates:
            raise KeyError(f"Missing pilot x0 estimate for level {after} or {before}")
        diff = x0_estimates[after].to(torch.float32) - x0_estimates[before].to(torch.float32)
        if diff.shape[-2:] != missing.shape[-2:]:
            raise ValueError("x0 estimates and mask must have the same spatial size")
        d_hole = region_rms(diff, missing)
        d_band = region_rms(diff, band)
        d_known = region_rms(diff, known)
        denom = (d_known + ratio_eps) if apply_ratio else 1.0
        rows["anchor"].append(anchor)
        rows["after_level"].append(after)
        rows["before_level"].append(before)
        rows["d_hole"].append(d_hole)
        rows["d_band"].append(d_band)
        rows["d_known"].append(d_known)
        rows["n_hole"].append(d_hole / denom)
        rows["n_band"].append(d_band / denom)

    def shares(values: List[float]) -> List[float]:
        total = sum(values)
        return [v / total for v in values] if total > 0 else [0.0] * len(values)

    hole_share = shares(rows["n_hole"])
    band_share = shares(rows["n_band"])
    rows["hole_share"] = hole_share
    rows["band_share"] = band_share
    rows["score"] = [h + lambda_boundary * b for h, b in zip(hole_share, band_share)]
    for key in ("d_hole", "d_band", "d_known", "n_hole", "n_band", "hole_share", "band_share", "score"):
        if not all(math.isfinite(v) and v >= 0 for v in rows[key]):
            raise FloatingPointError(f"Non-finite or negative values in {key}")
    rows["normalization"] = normalization
    rows["normalization_applied"] = apply_ratio
    rows["lambda_boundary"] = float(lambda_boundary)
    rows["boundary_band_px"] = int(boundary_band_px)
    rows["ratio_eps"] = float(ratio_eps)
    return rows


# ---------------------------------------------------------------------------
# Integer allocation under a fixed cycle budget
# ---------------------------------------------------------------------------
def allocate_cycles(
    scores: Sequence[float],
    total_cycles: int,
    min_cycles: int = 0,
    max_cycles: Optional[int] = None,
) -> List[int]:
    """Constrained proportional allocation of exactly ``total_cycles`` cycles.

    Rule (computed in exact rational arithmetic, hence deterministic):

    1. Every anchor first receives ``min_cycles``.
    2. The remaining ``R`` cycles are split in proportion to the scores
       ("water-filling"): anchors whose proportional share would exceed their
       remaining room ``max_cycles - min_cycles`` are fixed at the cap and the
       rest is re-split among the others until no share exceeds its cap.
       If the anchors still being filled all have score 0 (including the case
       where every score is 0), they share the remainder equally.
    3. Shares are floored; the leftover cycles go one each to the anchors
       with the largest fractional parts. Ties are broken by larger score,
       then by lower position in ``scores``.

    Raises ``ValueError`` if the request is infeasible
    (``len*min > total`` or ``len*max < total``) instead of over- or
    under-spending. Returns a list of ints summing exactly to ``total_cycles``.
    """
    n = len(scores)
    for name, value in (("total_cycles", total_cycles), ("min_cycles", min_cycles)):
        if int(value) != value or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    total_cycles, min_cycles = int(total_cycles), int(min_cycles)
    if max_cycles is not None:
        if int(max_cycles) != max_cycles or max_cycles < min_cycles:
            raise ValueError("max_cycles must be an integer >= min_cycles")
        max_cycles = int(max_cycles)
    values = [float(s) for s in scores]
    if any((not math.isfinite(v)) or v < 0 for v in values):
        raise ValueError("scores must be finite and non-negative")
    if n == 0:
        if total_cycles == 0:
            return []
        raise ValueError("No eligible anchors to receive cycles")
    if n * min_cycles > total_cycles:
        raise ValueError(
            f"Infeasible: {n} anchors x min {min_cycles} = {n * min_cycles} > budget {total_cycles} cycles"
        )
    cap = total_cycles if max_cycles is None else max_cycles
    room = cap - min_cycles
    if n * cap < total_cycles:
        raise ValueError(
            f"Infeasible: {n} anchors x max {cap} = {n * cap} < budget {total_cycles} cycles"
        )

    remaining = total_cycles - n * min_cycles
    weights = [Fraction(v) for v in values]
    capped: set = set()
    while True:
        free = [i for i in range(n) if i not in capped]
        free_budget = remaining - room * len(capped)
        w = [weights[i] for i in free]
        if sum(w) == 0:
            w = [Fraction(1)] * len(free)
        w_total = sum(w)
        shares = {i: Fraction(free_budget) * wi / w_total for i, wi in zip(free, w)} if free else {}
        newly = [i for i in free if shares[i] > room]
        if not newly:
            break
        capped.update(newly)
    real_share = {i: (Fraction(room) if i in capped else shares[i]) for i in range(n)}
    extra = {i: math.floor(real_share[i]) for i in range(n)}
    leftover = remaining - sum(extra.values())
    order = sorted(
        (i for i in range(n) if real_share[i] - extra[i] > 0),
        key=lambda i: (-(real_share[i] - extra[i]), -weights[i], i),
    )
    if leftover > len(order):
        raise RuntimeError("Internal allocation error: not enough fractional remainders")
    for i in order[:leftover]:
        extra[i] += 1
    allocation = [min_cycles + extra[i] for i in range(n)]
    if sum(allocation) != total_cycles or any(a < min_cycles or a > cap for a in allocation):
        raise RuntimeError(f"Internal allocation error: {allocation}")
    return allocation


def uniform_allocation(num_anchors: int, total_cycles: int) -> List[int]:
    """Equal split (floor/ceil) with the same deterministic remainder rule."""
    return allocate_cycles([1.0] * num_anchors, total_cycles, 0, None)


# ---------------------------------------------------------------------------
# Execution with a fixed allocation
# ---------------------------------------------------------------------------
def budgeted_repaint(
    model,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    extra_cycles: Mapping[int, int],
    num_inference_steps: int,
    jump_length: int,
    generator: Optional[torch.Generator] = None,
    x_T: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """RePaint with ``extra_cycles[a]`` additional cycles at each anchor ``a``.

    Only the per-anchor cycle counts differ from :func:`repaint_pipeline`; the
    schedule is built by the official loop and run by the shared executor.
    Returns ``(raw_output, schedule)``.
    """
    schedule = build_repaint_schedule_from_counts(num_inference_steps, jump_length, extra_cycles)
    output = run_repaint_schedule(model, scheduler, x_0, mask, schedule, num_inference_steps,
                                  generator=generator, x_T=x_T)
    return output, schedule


# ---------------------------------------------------------------------------
# Paired comparison of the four methods on one image
# ---------------------------------------------------------------------------
def _sync_time(device) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _reset_peak(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak(device) -> Optional[int]:
    if torch.device(device).type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return None


@dataclass
class MethodResult:
    method: str
    raw: torch.Tensor
    pilot_nfe: int
    final_nfe: int
    total_nfe: int
    expected_total_nfe: int
    extra_cycles_total: int
    allocation: Dict[int, int]
    pilot_wall_time_s: float
    final_wall_time_s: float
    total_wall_time_s: float
    peak_gpu_memory_bytes: Optional[int]
    schedule_length: int
    noise_seed: int

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("raw")
        data["allocation"] = {str(k): v for k, v in self.allocation.items()}
        return data


@dataclass
class ComparisonResult:
    plan: BudgetPlan
    settings: dict
    pilot: Optional[PilotResult]
    scores: Optional[dict]
    methods: Dict[str, MethodResult] = field(default_factory=dict)


def run_comparison(
    model,
    scheduler,
    x_0: torch.Tensor,
    mask: torch.Tensor,
    num_inference_steps: int,
    jump_length: int,
    jump_n_sample: int,
    latent_seed: int,
    noise_seed: int,
    pilot_noise_seed: Optional[int] = None,
    reference_budget_nfe: Optional[int] = None,
    min_extra_cycles: int = 3,
    max_extra_cycles: Optional[int] = 14,
    lambda_boundary: float = 0.5,
    boundary_band_px: int = 8,
    normalization: str = "known_ratio",
    ratio_eps: float = 1e-3,
    methods: Sequence[str] = METHOD_NAMES,
    verbose: bool = True,
) -> ComparisonResult:
    """Run the selected methods on one image/mask with shared inputs.

    Shared across methods: ``x_0``, ``mask``, the model/scheduler, the initial
    latent ``x_T`` (from ``latent_seed``) and the sampler-noise seed. Every
    sampler pass gets a *fresh* generator, so running the pilot cannot shift
    the random numbers of any later pass. Trajectories still diverge after the
    first point where two schedules differ, because the shared noise stream is
    consumed in a different order.

    ``uniform`` and ``adaptive`` are both charged the (single, shared) pilot
    run and therefore have identical total NFE by construction; this is
    asserted from the measured counts.
    """
    unknown = [m for m in methods if m not in METHOD_NAMES]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {METHOD_NAMES}")
    if x_0.shape[0] != 1:
        raise ValueError("run_comparison processes one image at a time (batch size 1)")
    device = x_0.device
    counter = model if isinstance(model, CountingModel) else CountingModel(model)
    pilot_noise_seed = noise_seed if pilot_noise_seed is None else pilot_noise_seed
    # Validate inputs once with the shared validator (raises clear errors).
    _prepare_sampling_inputs(scheduler, x_0, mask, num_inference_steps)
    plan = plan_budget(num_inference_steps, jump_length, jump_n_sample, reference_budget_nfe)
    x_T = make_initial_latent(x_0.shape, latent_seed, dtype=x_0.dtype).to(device)
    anchors = eligible_jump_anchors(num_inference_steps, jump_length)
    settings = dict(
        num_inference_steps=num_inference_steps, jump_length=jump_length, jump_n_sample=jump_n_sample,
        latent_seed=latent_seed, noise_seed=noise_seed, pilot_noise_seed=pilot_noise_seed,
        min_extra_cycles=min_extra_cycles, max_extra_cycles=max_extra_cycles,
        lambda_boundary=lambda_boundary, boundary_band_px=boundary_band_px,
        normalization=normalization, ratio_eps=ratio_eps, anchors=anchors,
        x_T_checksum=float(x_T.double().sum().item()), device=str(device),
    )
    result = ComparisonResult(plan=plan, settings=settings, pilot=None, scores=None)
    log = print if verbose else (lambda *a, **k: None)

    def run_fixed(name: str, r: int) -> None:
        counter.reset()
        _reset_peak(device)
        start = _sync_time(device)
        raw = repaint_pipeline(counter, scheduler, x_0, mask, jump_length=jump_length, jump_n_sample=r,
                               generator=make_generator(noise_seed, device),
                               num_inference_steps=num_inference_steps, x_T=x_T)
        wall = _sync_time(device) - start
        schedule = build_repaint_schedule(num_inference_steps, jump_length, r)
        expected = schedule_nfe(schedule)
        if counter.evaluations != expected:
            raise RuntimeError(f"{name}: measured NFE {counter.evaluations} != schedule NFE {expected}")
        result.methods[name] = MethodResult(
            method=name, raw=raw, pilot_nfe=0, final_nfe=counter.evaluations, total_nfe=counter.evaluations,
            expected_total_nfe=expected, extra_cycles_total=(r - 1) * len(anchors),
            allocation={a: r - 1 for a in anchors}, pilot_wall_time_s=0.0, final_wall_time_s=wall,
            total_wall_time_s=wall, peak_gpu_memory_bytes=_peak(device), schedule_length=len(schedule),
            noise_seed=noise_seed,
        )
        log(f"[{name}] NFE={counter.evaluations} (expected {expected}), wall={wall:.1f}s")

    if "fixed" in methods:
        run_fixed("fixed", jump_n_sample)
    if "no_resampling" in methods:
        run_fixed("no_resampling", 1)

    budgeted = [m for m in ("uniform", "adaptive") if m in methods]
    if budgeted:
        counter.reset()
        _reset_peak(device)
        pilot = run_pilot(counter, scheduler, x_0, mask, x_T, num_inference_steps, jump_length,
                          make_generator(pilot_noise_seed, device))
        pilot_peak = _peak(device)
        if pilot.nfe != plan.pilot_nfe:
            raise RuntimeError(f"Pilot measured NFE {pilot.nfe} != planned {plan.pilot_nfe}")
        result.pilot = pilot
        log(f"[pilot] NFE={pilot.nfe}, wall={pilot.wall_time_s:.1f}s")
        t0 = time.perf_counter()
        scores = compute_instability_scores(
            pilot.x0_estimates, mask[:1], num_inference_steps, jump_length,
            lambda_boundary=lambda_boundary, boundary_band_px=boundary_band_px,
            normalization=normalization, ratio_eps=ratio_eps,
        )
        stride = _prepare_sampling_inputs(scheduler, x_0, mask, num_inference_steps)[1]
        scores["native_timestep"] = [a * stride for a in scores["anchor"]]
        scoring_time = time.perf_counter() - t0
        result.scores = scores
        allocations = {
            "uniform": uniform_allocation(len(anchors), plan.extra_cycles),
            "adaptive": allocate_cycles(scores["score"], plan.extra_cycles, min_extra_cycles, max_extra_cycles),
        }
        for name in budgeted:
            alloc = dict(zip(anchors, allocations[name]))
            counter.reset()
            _reset_peak(device)
            start = _sync_time(device)
            raw, schedule = budgeted_repaint(counter, scheduler, x_0, mask, alloc, num_inference_steps,
                                             jump_length, generator=make_generator(noise_seed, device), x_T=x_T)
            wall = _sync_time(device) - start
            expected_final = schedule_nfe(schedule)
            if counter.evaluations != expected_final:
                raise RuntimeError(f"{name}: measured NFE {counter.evaluations} != schedule NFE {expected_final}")
            total = pilot.nfe + counter.evaluations
            if total != plan.budgeted_total_nfe:
                raise RuntimeError(f"{name}: total NFE {total} != planned {plan.budgeted_total_nfe}")
            peaks = [p for p in (pilot_peak, _peak(device)) if p is not None]
            result.methods[name] = MethodResult(
                method=name, raw=raw, pilot_nfe=pilot.nfe, final_nfe=counter.evaluations, total_nfe=total,
                expected_total_nfe=plan.budgeted_total_nfe, extra_cycles_total=sum(alloc.values()),
                allocation=alloc, pilot_wall_time_s=pilot.wall_time_s + scoring_time,
                final_wall_time_s=wall, total_wall_time_s=pilot.wall_time_s + scoring_time + wall,
                peak_gpu_memory_bytes=max(peaks) if peaks else None, schedule_length=len(schedule),
                noise_seed=noise_seed,
            )
            log(f"[{name}] NFE pilot {pilot.nfe} + final {counter.evaluations} = {total}, wall={wall:.1f}s")
        if {"uniform", "adaptive"} <= set(result.methods):
            if result.methods["uniform"].total_nfe != result.methods["adaptive"].total_nfe:
                raise RuntimeError("Uniform and adaptive totals differ; budget matching failed")
    return result
