"""Faces vs Places2 study: mask geometry x coverage x sampler, with LPIPS.

Orchestration only. Sampling, NFE accounting and per-run saving reuse
``src.adaptive_repaint.run_comparison`` and ``src.experiment_io.save_comparison``
unchanged; metrics reuse ``src.metrics.compute_inpainting_metrics`` and add
LPIPS from ``src.lpips_metric``. No sampler logic lives here.

Output layout (one folder per dataset / mask family / ratio / image):

    <out_root>/<dataset>/<family>/ratio_<r>/<image_stem>/
        images/*.png, metrics.csv, scores.csv, allocation.csv, run.json   (save_comparison)
        study_rows.csv     one row per sampler, all metadata + metrics + LPIPS
        condition.json     settings used; a rerun with different settings is refused

Aggregates written by :func:`write_study_summaries`:

    per_image_results.csv, summary_by_condition.csv, paired_adaptive_vs_uniform.csv,
    paired_adaptive_vs_fixed.csv, mask_difficulty_ranking.csv, relative_to_no_resampling.csv
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from src.adaptive_repaint import METHOD_NAMES, BudgetPlan, CountingModel, plan_budget, run_comparison
from src.config import load_config, resolve_project_path
from src.experiment_io import environment_info, save_comparison
from src.lpips_metric import LPIPSEvaluator
from src.mask_families import MaskSpec, generate_mask, mask_conditions
from src.metrics import compute_inpainting_metrics
from src.repaint import eligible_jump_anchors
from src.study_datasets import load_study_image

STUDY_CONFIG_RELATIVE_PATH = Path("configs") / "faces_places2_study.yaml"
METRIC_COLUMNS = (
    "lpips_full", "lpips_missing_masked",
    "composite_full_psnr_db", "composite_full_ssim", "composite_full_mse",
    "missing_mae", "missing_mse", "missing_psnr_db",
    "boundary_inner_mae", "boundary_inner_mse", "boundary_inner_psnr_db",
    "raw_known_mae", "raw_known_psnr_db",
)
LOWER_IS_BETTER = {"lpips_full", "lpips_missing_masked", "composite_full_mse", "missing_mae", "missing_mse",
                   "boundary_inner_mae", "boundary_inner_mse", "raw_known_mae"}


# ---------------------------------------------------------------------------
# Configuration and budgets
# ---------------------------------------------------------------------------
def load_study_config(project_root: Path, config_path: Optional[Path] = None) -> dict:
    path = resolve_project_path(config_path or STUDY_CONFIG_RELATIVE_PATH, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Study config not found: {path}")
    return load_config(str(path))


def budget_settings(cfg: dict, profile: str) -> dict:
    budgets = cfg.get("budgets", {})
    if profile not in budgets:
        raise KeyError(f"Unknown budget profile {profile!r}; available: {sorted(budgets)}")
    settings = dict(budgets[profile])
    settings["profile"] = profile
    return settings


def check_budget(budget: dict) -> BudgetPlan:
    """Plan the budget and verify the adaptive limits can spend it exactly."""
    plan = plan_budget(budget["num_inference_steps"], budget["jump_length"], budget["jump_n_sample"])
    n = len(eligible_jump_anchors(budget["num_inference_steps"], budget["jump_length"]))
    lo, hi = int(budget["min_extra_cycles"]), budget.get("max_extra_cycles")
    if n * lo > plan.extra_cycles or (hi is not None and n * int(hi) < plan.extra_cycles):
        raise ValueError(
            f"Budget profile {budget.get('profile')!r} is infeasible: {plan.extra_cycles} extra cycles over "
            f"{n} anchors with limits [{lo}, {hi}] (need {n}*{lo} <= {plan.extra_cycles} <= {n}*{hi})."
        )
    return plan


def executed_nfe_per_run(plan: BudgetPlan, methods: Sequence[str]) -> int:
    """Denoiser calls actually executed for one image/mask (the pilot is shared)."""
    total = 0
    if "fixed" in methods:
        total += plan.fixed_schedule_nfe
    if "no_resampling" in methods:
        total += plan.base_trajectory_nfe
    budgeted = [m for m in ("uniform", "adaptive") if m in methods]
    if budgeted:
        total += plan.pilot_nfe + len(budgeted) * (plan.budgeted_total_nfe - plan.pilot_nfe)
    return total


def estimate_study_cost(images_per_dataset: Dict[str, int], n_conditions: int, budget: dict,
                        methods: Sequence[str], seconds_per_call: Optional[Dict[str, float]] = None) -> List[dict]:
    plan = check_budget(budget)
    per_run = executed_nfe_per_run(plan, methods)
    rows = []
    for name, n_images in images_per_dataset.items():
        runs = n_images * n_conditions
        spc = (seconds_per_call or {}).get(name)
        rows.append({
            "dataset": name, "images": n_images, "mask_conditions": n_conditions, "runs": runs,
            "executed_nfe_per_run": per_run, "total_nfe": runs * per_run,
            "seconds_per_call": spc,
            "estimated_hours": None if spc is None else runs * per_run * spc / 3600.0,
        })
    return rows


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def first_existing(candidates: Sequence, project_root: Path, want_dir: bool) -> Optional[Path]:
    for candidate in candidates or ():
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = project_root / path
        if (want_dir and path.is_dir()) or (not want_dir and path.is_file()):
            return path.resolve()
    return None


def load_dataset_model(dataset_cfg: dict, project_root: Path, device):
    """Return ``(model, scheduler, info)`` for one dataset's configured backend."""
    backend = dataset_cfg["backend"]
    if backend == "diffusers_ddpm":
        from src.model import load_pretrained_ddpm

        model, scheduler, device = load_pretrained_ddpm(dataset_cfg["model_id"], device=torch.device(device))
        info = {
            "backend": backend, "model_identifier": dataset_cfg["model_id"],
            "parameters_million": sum(p.numel() for p in model.parameters()) / 1e6,
            "learned_variance_discarded": False,
            "scheduler_config": {k: scheduler.config[k] for k in (
                "num_train_timesteps", "beta_schedule", "beta_start", "beta_end", "variance_type",
                "prediction_type", "clip_sample")},
        }
        return model, scheduler, info
    if backend == "repaint_guided_diffusion":
        from src.guided_diffusion_adapter import load_repaint_checkpoint

        checkpoint = first_existing(dataset_cfg.get("checkpoint_candidates"), project_root, want_dir=False)
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found among {dataset_cfg.get('checkpoint_candidates')}")
        code_dir = first_existing(dataset_cfg.get("official_code_dirs"), project_root, want_dir=True)
        if code_dir is None:
            raise FileNotFoundError(f"No official RePaint checkout found among {dataset_cfg.get('official_code_dirs')}")
        model, scheduler, info = load_repaint_checkpoint(checkpoint, code_dir, dataset_cfg.get("model_config"),
                                                         device=device, use_fp16=bool(dataset_cfg.get("use_fp16", False)))
        info["model_identifier"] = f"RePaint official {checkpoint.name}"
        return model, scheduler, info
    raise ValueError(f"Unknown backend {backend!r}")


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------
def compute_study_metrics(raw: torch.Tensor, x_0: torch.Tensor, mask: torch.Tensor,
                          lpips_eval: Optional[LPIPSEvaluator], boundary_width: int,
                          compute_ssim: bool = True) -> dict:
    metrics = compute_inpainting_metrics(raw, x_0, mask, boundary_width=boundary_width,
                                         compute_ssim=compute_ssim, lpips_fn=None)
    metrics.pop("lpips_composite", None)
    composite = (1.0 - mask) * x_0 + mask * raw
    if lpips_eval is None:
        metrics["lpips_full"] = None
        metrics["lpips_missing_masked"] = None
    else:
        metrics["lpips_full"] = lpips_eval.full(composite, x_0)
        metrics["lpips_missing_masked"] = lpips_eval.masked(composite, x_0, mask)
    return metrics


def condition_dir(out_root: Path, dataset: str, family: str, ratio: float, image_path: Path) -> Path:
    return Path(out_root) / dataset / family / f"ratio_{ratio:.2f}" / image_path.stem


def _write_rows(path: Path, rows: List[dict]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def run_condition(model, scheduler, dataset: str, dataset_cfg: dict, model_info: dict, image_path: Path,
                  image_root: Path, image_index: int, mask_spec: MaskSpec, budget: dict, cfg: dict,
                  lpips_eval: Optional[LPIPSEvaluator], lpips_meta: dict, out_dir: Path, device,
                  compute_ssim: bool = True) -> List[dict]:
    """Run all configured samplers on one image + mask and write ``study_rows.csv``."""
    study = cfg["study"]
    adaptive = cfg.get("adaptive", {})
    methods = tuple(study.get("methods", METHOD_NAMES))
    device = torch.device(device)
    x_0 = load_study_image(image_path, int(study["image_size"]), dataset_cfg["preprocessing"]).to(device)
    mask = mask_spec.mask.to(device)
    latent_seed = int(study["latent_seed"]) + int(image_index)
    noise_seed = int(study["noise_seed"]) + int(image_index)
    result = run_comparison(
        CountingModel(model), scheduler, x_0, mask,
        int(budget["num_inference_steps"]), int(budget["jump_length"]), int(budget["jump_n_sample"]),
        latent_seed=latent_seed, noise_seed=noise_seed, pilot_noise_seed=noise_seed,
        min_extra_cycles=int(budget["min_extra_cycles"]), max_extra_cycles=budget.get("max_extra_cycles"),
        lambda_boundary=float(adaptive.get("lambda_boundary", 0.5)),
        boundary_band_px=int(adaptive.get("boundary_band_px", 8)),
        normalization=adaptive.get("normalization", "known_ratio"),
        ratio_eps=float(adaptive.get("ratio_eps", 1e-3)),
        methods=methods, verbose=False,
    )
    try:
        image_id = str(Path(image_path).relative_to(image_root))
    except ValueError:
        image_id = Path(image_path).name
    base_meta = {
        "dataset": dataset,
        "model_identifier": model_info.get("model_identifier"),
        "model_backend": model_info.get("backend"),
        "learned_variance_discarded": model_info.get("learned_variance_discarded"),
        "image_id": image_id,
        "image_index": int(image_index),
        "preprocessing": dataset_cfg["preprocessing"],
        "budget_profile": budget["profile"],
        "num_inference_steps": int(budget["num_inference_steps"]),
        "jump_length": int(budget["jump_length"]),
        "jump_n_sample": int(budget["jump_n_sample"]),
    }
    base_meta.update(mask_spec.metadata())
    out_dir.mkdir(parents=True, exist_ok=True)
    save_meta = dict(base_meta, mask_type=f"{mask_spec.family}@{mask_spec.target_ratio:.2f}",
                     run_kind="faces_places2_study", lpips=lpips_meta, model_info=model_info)
    save_comparison(result, x_0, mask, out_dir, save_meta, compute_ssim=compute_ssim,
                    lpips_fn=None, boundary_width=int(study.get("boundary_band_px", 8)))
    rows = []
    for name, method in result.methods.items():
        row = dict(base_meta)
        row.update({
            "method": name, "latent_seed": latent_seed, "noise_seed": noise_seed,
            "pilot_nfe": method.pilot_nfe, "final_nfe": method.final_nfe, "total_nfe": method.total_nfe,
            "expected_total_nfe": method.expected_total_nfe, "extra_cycles_total": method.extra_cycles_total,
            "pilot_wall_time_s": method.pilot_wall_time_s, "final_wall_time_s": method.final_wall_time_s,
            "total_wall_time_s": method.total_wall_time_s, "peak_gpu_memory_bytes": method.peak_gpu_memory_bytes,
        })
        row.update(compute_study_metrics(method.raw, x_0, mask, lpips_eval,
                                         int(study.get("boundary_band_px", 8)), compute_ssim))
        row.update({"lpips_status": lpips_meta.get("lpips_status"), "lpips_net": lpips_meta.get("lpips_net"),
                    "lpips_valid": lpips_meta.get("lpips_valid")})
        row["run_dir"] = str(out_dir)
        rows.append(row)
    _write_rows(out_dir / "study_rows.csv", rows)
    return rows


def _condition_key(dataset: str, image_id: str, mask_spec: MaskSpec, budget: dict, cfg: dict,
                   lpips_meta: dict) -> dict:
    study = cfg["study"]
    return {
        "dataset": dataset, "image": image_id, "mask": {k: v for k, v in mask_spec.metadata().items()},
        "budget": {k: budget[k] for k in sorted(budget)}, "latent_seed": study["latent_seed"],
        "noise_seed": study["noise_seed"], "methods": list(study.get("methods", METHOD_NAMES)),
        "adaptive": cfg.get("adaptive", {}), "lpips_status": lpips_meta.get("lpips_status"),
    }


# ---------------------------------------------------------------------------
# Whole study
# ---------------------------------------------------------------------------
def run_study(datasets: Dict[str, dict], cfg: dict, budget: dict, out_root: Path, device,
              lpips_eval: Optional[LPIPSEvaluator], lpips_meta: dict,
              families: Optional[Sequence[str]] = None, ratios: Optional[Sequence[float]] = None,
              require_lpips: Optional[bool] = None, overwrite: bool = False,
              max_runs: Optional[int] = None, time_budget_hours: Optional[float] = None,
              compute_ssim: bool = True, log=print) -> dict:
    """Run every (dataset, mask condition, image). Resumable and budget-checked.

    ``datasets[name]`` must contain ``model``, ``scheduler``, ``model_info``,
    ``config`` (the dataset's config block), ``images`` (list of paths) and
    ``image_root`` (folder that image ids are relative to).
    """
    require = cfg["study"].get("require_lpips", True) if require_lpips is None else require_lpips
    if require and (lpips_eval is None or not lpips_meta.get("lpips_valid", False)):
        raise RuntimeError(
            "LPIPS is required for this study but is not available or not valid: "
            f"{lpips_meta.get('lpips_status')} ({lpips_meta.get('lpips_error')}). "
            "Fix the LPIPS setup (see Docs/faces_places2_mask_lpips_study.md) or run in smoke mode."
        )
    check_budget(budget)
    masks_cfg = cfg["masks"]
    conditions = mask_conditions(families or masks_cfg["families"], ratios or masks_cfg["target_ratios"],
                                 masks_cfg.get("not_applicable"))
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    done = skipped = 0
    stopped_reason = None
    for name, entry in datasets.items():
        for family, ratio in conditions:
            for index, image_path in enumerate(entry["images"]):
                if max_runs is not None and done >= max_runs:
                    stopped_reason = f"max_runs={max_runs} reached"
                    break
                if time_budget_hours is not None and (time.perf_counter() - start) / 3600 > time_budget_hours:
                    stopped_reason = f"time budget {time_budget_hours} h reached"
                    break
                spec = generate_mask(family, ratio, int(cfg["study"]["image_size"]), index,
                                     int(cfg["study"]["mask_base_seed"]), masks_cfg.get("coverage_tolerance", 0.01))
                image_path = Path(image_path)
                try:
                    image_id = str(image_path.relative_to(entry["image_root"]))
                except ValueError:
                    image_id = image_path.name
                key = _condition_key(name, image_id, spec, budget, cfg, lpips_meta)
                run_dir = condition_dir(out_root, name, family, ratio, image_path)
                key_path = run_dir / "condition.json"
                if (run_dir / "study_rows.csv").exists() and not overwrite:
                    previous = json.loads(key_path.read_text()) if key_path.exists() else None
                    if previous != json.loads(json.dumps(key)):
                        raise RuntimeError(
                            f"{run_dir} holds results from different settings; use a new results folder "
                            "or overwrite=True."
                        )
                    skipped += 1
                    continue
                t0 = time.perf_counter()
                run_condition(entry["model"], entry["scheduler"], name, entry["config"], entry["model_info"],
                              image_path, entry["image_root"], index, spec, budget, cfg, lpips_eval,
                              lpips_meta, run_dir, device, compute_ssim)
                key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
                done += 1
                log(f"[{name}] {family} @ {ratio:.2f} (actual {spec.actual_ratio:.3f}) "
                    f"{image_path.name}: {time.perf_counter() - t0:.1f} s")
            if stopped_reason:
                break
        if stopped_reason:
            break
    manifest = {
        "completed_this_call": done, "skipped_existing": skipped, "stopped_reason": stopped_reason,
        "elapsed_s": time.perf_counter() - start, "budget": budget, "conditions": conditions,
        "datasets": {k: {"images": [str(p) for p in v["images"]], "model_info": v["model_info"]}
                     for k, v in datasets.items()},
        "lpips": lpips_meta, "environment": environment_info(),
    }
    (out_root / "study_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    log(f"Completed {done} run(s), skipped {skipped} existing" + (f"; stopped: {stopped_reason}" if stopped_reason else ""))
    return manifest


# ---------------------------------------------------------------------------
# Aggregation (pandas)
# ---------------------------------------------------------------------------
def collect_results(out_root: Path):
    import pandas as pd

    files = sorted(Path(out_root).rglob("study_rows.csv"))
    if not files:
        raise FileNotFoundError(f"No study_rows.csv files under {out_root}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(df):
    """Mean/std/n per dataset x mask family x target ratio x method."""
    keys = ["dataset", "mask_family", "mask_target_ratio", "method"]
    metrics = [m for m in METRIC_COLUMNS if m in df and df[m].notna().any()]
    agg = df.groupby(keys)[metrics + ["mask_actual_ratio", "total_nfe", "total_wall_time_s"]].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def paired_differences(df, method_a: str, method_b: str, metrics: Optional[Sequence[str]] = None):
    """Per dataset x family x ratio: paired (same image, mask, seeds) differences a - b."""
    import numpy as np
    import pandas as pd

    metrics = [m for m in (metrics or METRIC_COLUMNS) if m in df and df[m].notna().any()]
    keys = ["dataset", "mask_family", "mask_target_ratio", "image_id"]
    a = df[df.method == method_a].set_index(keys)
    b = df[df.method == method_b].set_index(keys)
    joined = a[metrics].join(b[metrics], lsuffix="_a", rsuffix="_b", how="inner")
    rows = []
    for (dataset, family, ratio), group in joined.groupby(level=[0, 1, 2]):
        for metric in metrics:
            d = (group[f"{metric}_a"] - group[f"{metric}_b"]).dropna()
            if d.empty:
                continue
            better = (d < 0) if metric in LOWER_IS_BETTER else (d > 0)
            worse = (d > 0) if metric in LOWER_IS_BETTER else (d < 0)
            rows.append({
                "dataset": dataset, "mask_family": family, "mask_target_ratio": ratio, "metric": metric,
                "comparison": f"{method_a} - {method_b}", "n_pairs": int(len(d)), "mean_diff": float(d.mean()),
                "std_diff": float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                "sem_diff": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else np.nan,
                f"n_{method_a}_better": int(better.sum()), f"n_{method_b}_better": int(worse.sum()),
                "n_ties": int((d == 0).sum()),
                "lower_is_better": metric in LOWER_IS_BETTER,
            })
    return pd.DataFrame(rows)


def mask_difficulty_ranking(df, method: str = "fixed", metric: str = "lpips_full"):
    """Within each dataset, rank mask families by mean ``metric`` (rank 1 = most challenging).

    Ranks are computed per dataset and should be compared across datasets only
    as ranks; raw values are not comparable between domains.
    """
    sub = df[(df.method == method) & df[metric].notna()]
    if sub.empty:
        return sub.iloc[0:0]
    table = sub.groupby(["dataset", "mask_family"])[metric].agg(["mean", "std", "count"]).reset_index()
    ascending = metric not in LOWER_IS_BETTER   # higher LPIPS / MAE = harder; lower PSNR = harder
    table["rank_most_challenging"] = table.groupby("dataset")["mean"].rank(ascending=ascending, method="min").astype(int)
    table["metric"] = metric
    table["method"] = method
    return table.sort_values(["dataset", "rank_most_challenging"]).reset_index(drop=True)


def relative_to_no_resampling(df, metric: str = "lpips_full"):
    """Per image, (method - no_resampling) / no_resampling, then averaged.

    A within-dataset, scale-free effect size that can be shown side by side for
    the two domains without treating raw values as comparable.
    """
    import pandas as pd

    keys = ["dataset", "mask_family", "mask_target_ratio", "image_id"]
    base = df[df.method == "no_resampling"].set_index(keys)[metric]
    rows = []
    for method in [m for m in ("fixed", "uniform", "adaptive") if m in set(df.method)]:
        cur = df[df.method == method].set_index(keys)[metric]
        rel = ((cur - base) / base.where(base != 0)).dropna()
        if rel.empty:
            continue
        frame = rel.rename("relative_change").reset_index()
        frame["method"] = method
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby(["dataset", "mask_family", "mask_target_ratio", "method"])["relative_change"].agg(
        ["mean", "std", "count"]).reset_index()
    out["metric"] = metric
    return out


def write_study_summaries(out_root: Path) -> Dict[str, str]:
    out_root = Path(out_root)
    df = collect_results(out_root)
    paths = {}

    def save(frame, name):
        path = out_root / name
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    save(df, "per_image_results.csv")
    save(summarize(df), "summary_by_condition.csv")
    methods = set(df.method)
    if {"adaptive", "uniform"} <= methods:
        save(paired_differences(df, "adaptive", "uniform"), "paired_adaptive_vs_uniform.csv")
    if {"adaptive", "fixed"} <= methods:
        save(paired_differences(df, "adaptive", "fixed"), "paired_adaptive_vs_fixed.csv")
    ranks = []
    for metric in ("lpips_full", "missing_psnr_db"):
        if metric in df and df[metric].notna().any() and "fixed" in methods:
            ranks.append(mask_difficulty_ranking(df, "fixed", metric))
    if ranks:
        import pandas as pd
        save(pd.concat(ranks, ignore_index=True), "mask_difficulty_ranking.csv")
    if "no_resampling" in methods:
        rel = [relative_to_no_resampling(df, m) for m in ("lpips_full", "missing_mae")
               if m in df and df[m].notna().any()]
        rel = [r for r in rel if len(r)]
        if rel:
            import pandas as pd
            save(pd.concat(rel, ignore_index=True), "relative_to_no_resampling.csv")
    return paths
