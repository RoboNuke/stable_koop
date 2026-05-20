"""Dataset-level base-policy statistics, printed after a gather call.

Re-creates the legacy ``phase_0_base_eval`` reporting from the saved dataset
instead of re-running the env. Generic per-trajectory metrics (length,
reward, action magnitude) work for any gym env. When an :class:`eval.EnvScorer`
is registered for the env, its ``compute_metrics`` and ``check_success`` are
additionally invoked to give env-specific signals (e.g. pendulum energy,
angle-success).

The printed format mirrors the legacy ``print_stats_table`` four-column
layout: ``Metric | Combined | Success | Failure``, with each cell showing
``mean ± std`` for that metric over the corresponding group.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import yaml

from data.dataloader import load_dataset


def _success_cfg_from_eval(eval_cfg_path: str) -> dict:
    """Load an EvalCfg YAML and return the dict the scorer needs."""
    from config.manager.manager import ConfigManager

    eval_cfg = ConfigManager.load_stage(eval_cfg_path, "eval_cfg")
    return {
        "success_angle_deg": eval_cfg.success_angle_deg,
        "success_max_thdot": eval_cfg.success_max_thdot,
        "success_max_cart_vel": eval_cfg.success_max_cart_vel,
        "success_hold_steps": eval_cfg.success_hold_steps,
        # Default horizon for "survived the rollout" checks; callers
        # (e.g. evaluate_dataset) may override with their own horizon.
        "max_steps": eval_cfg.eval_max_steps,
    }


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def _fmt_cell(values) -> str:
    """Format one ``mean ± std`` cell, width-padded for the table layout."""
    if len(values) == 0:
        return "          n/a       "
    mean, std = _mean_std(values)
    return f"{mean:8.3f} ± {std:8.3f}"


def _fmt_vec(v) -> str:
    return "[" + ", ".join(f"{x:.3g}" for x in np.asarray(v).ravel()) + "]"


def _print_table(per_traj: list[dict], success_mask: np.ndarray, metric_keys: list[str]) -> None:
    """Print ``Metric | Combined | Success | Failure`` table, legacy style."""
    succ_idx = np.where(success_mask)[0]
    fail_idx = np.where(~success_mask)[0]

    header = f"  {'Metric':<20} {'Combined':>20} {'Success':>20} {'Failure':>20}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in metric_keys:
        all_vals = [m[key] for m in per_traj]
        succ_vals = [per_traj[i][key] for i in succ_idx]
        fail_vals = [per_traj[i][key] for i in fail_idx]
        print(
            f"  {key:<20} "
            f"{_fmt_cell(all_vals):>20} "
            f"{_fmt_cell(succ_vals):>20} "
            f"{_fmt_cell(fail_vals):>20}"
        )


def evaluate_dataset(
    dataset_name: str,
    *,
    folder_name: str | None = None,
    eval_cfg_path: str | None = None,
) -> dict:
    """Compute + print summary stats for a saved dataset.

    Success thresholds come from the eval YAML at ``eval_cfg_path`` when
    provided. If not provided, the path stored in the dataset's saved
    ``gather_data_cfg.eval_cfg_path`` is used. If the env has a registered
    scorer and no eval cfg is resolvable, this raises — the gather pipeline
    is expected to point at an eval config so the gather-time stats use
    the same thresholds the eval step will use.

    Writes/updates ``data/datasets/<folder_name>/metadata.yaml`` keyed by
    ``dataset_name`` so ``--both`` runs produce base and ``_pert`` entries
    in the same file.
    """
    ds = load_dataset(dataset_name)
    snap = yaml.safe_load(ds.config_yaml).get("gather_data_cfg", {})
    env_name = snap.get("env_name", "<unknown>")
    max_episode_steps = snap.get("max_episode_steps", None)
    base_policy = snap.get("base_policy", {}).get("name", "<unknown>")
    dr_freeze = snap.get("dr_freeze", {})
    if eval_cfg_path is None:
        eval_cfg_path = snap.get("eval_cfg_path", "") or ""
    n = len(ds.trajectories)

    print("\n" + "=" * 86)
    print(f"  Dataset stats: {dataset_name}")
    print("=" * 86)
    print(f"  env:                 {env_name}")
    print(f"  base policy:         {base_policy}")
    print(f"  perturbations:       {bool(ds.perturbations_enabled)}")
    if max_episode_steps is not None:
        print(f"  max_episode_steps:   {max_episode_steps}")
    print(f"  state obs min / max: {_fmt_vec(ds.obs_min)} / {_fmt_vec(ds.obs_max)}")
    print(f"  action min / max:    {_fmt_vec(ds.act_min)} / {_fmt_vec(ds.act_max)}")

    if n == 0:
        print(f"  trajectories: 0")
        return {"dataset_name": dataset_name, "trajectories": 0}

    # Generic per-trajectory metrics — always computed.
    per_traj: list[dict] = []
    for states, actions, _bases, rewards in ds.trajectories:
        per_traj.append({
            "length": len(actions),
            "reward": float(rewards.sum()),
            "|action|": float(np.abs(actions).mean()),
        })

    # Env-specific scorer (if registered) — adds energy / torque / etc.
    try:
        from eval import ENV_SCORERS
    except Exception:
        ENV_SCORERS = {}
    scorer = ENV_SCORERS.get(env_name)

    success_mask: np.ndarray
    success_source: str
    if scorer is not None:
        if not eval_cfg_path:
            raise ValueError(
                f"evaluate_dataset({dataset_name!r}): env {env_name!r} has a "
                "registered scorer but no eval_cfg_path was provided. Set "
                "`eval_cfg_path` in the gather YAML to point at the matching "
                "eval config so dataset-time success thresholds match eval-time."
            )
        success_cfg = _success_cfg_from_eval(eval_cfg_path)
        # The "survived the rollout" success check needs to know the
        # expected trajectory horizon — for gather-time datasets that's
        # the env's max_episode_steps.
        if max_episode_steps is not None:
            success_cfg["max_steps"] = max_episode_steps
        # Extend per-traj dicts with scorer metrics + compute scorer-defined success.
        for i, (states, actions, _bases, _rewards) in enumerate(ds.trajectories):
            extra = scorer.compute_metrics(states, actions)
            # Scorer's "reward" placeholder is overwritten by the dataset's actual reward.
            extra.pop("reward", None)
            per_traj[i].update(extra)
        success_mask = np.array(
            [scorer.check_success(s, success_cfg) for s, *_ in ds.trajectories]
        )
        thresholds = ", ".join(
            f"{k.removeprefix('success_')}={v}" for k, v in success_cfg.items()
        )
        success_source = f"{env_name} scorer ({thresholds}) [from {eval_cfg_path}]"
    elif max_episode_steps is not None:
        success_mask = np.array([m["length"] == max_episode_steps for m in per_traj])
        success_source = f"reached max_episode_steps ({max_episode_steps})"
    else:
        success_mask = np.zeros(n, dtype=bool)
        success_source = "n/a (no scorer registered, no max_episode_steps)"

    n_success = int(success_mask.sum())
    n_failure = n - n_success
    rate = n_success / n

    print(
        f"\n  Trajectories: {n}  |  "
        f"Success: {n_success}  |  Failure: {n_failure}  |  "
        f"Rate: {rate:.1%}"
    )
    print(f"  Success criterion:   {success_source}\n")

    metric_keys = list(per_traj[0].keys())
    _print_table(per_traj, success_mask, metric_keys)
    print()

    out: dict[str, Any] = {
        "dataset_name": dataset_name,
        "env_name": env_name,
        "base_policy": base_policy,
        "perturbations_enabled": bool(ds.perturbations_enabled),
        "trajectories": n,
        "max_episode_steps": max_episode_steps,
        "dr_freeze": dr_freeze,
        "success_count": n_success,
        "failure_count": n_failure,
        "success_rate": rate,
        "success_criterion": success_source,
        "obs_min": ds.obs_min.tolist(),
        "obs_max": ds.obs_max.tolist(),
        "act_min": ds.act_min.tolist(),
        "act_max": ds.act_max.tolist(),
    }
    for group, idx in (
        ("combined", np.arange(n)),
        ("success", np.where(success_mask)[0]),
        ("failure", np.where(~success_mask)[0]),
    ):
        group_stats = {}
        for key in metric_keys:
            vals = [per_traj[i][key] for i in idx]
            mean, std = _mean_std(vals)
            group_stats[key] = {"mean": mean, "std": std}
        out[group] = group_stats

    # Persist into the dataset folder's shared metadata.yaml (keyed by dataset_name
    # so a --both gather ends up with both base and pert entries).
    _write_metadata(folder_name or dataset_name, dataset_name, out)
    return out


def _write_metadata(folder_name: str, dataset_name: str, stats: dict) -> None:
    """Merge ``stats`` into ``data/datasets/<folder_name>/metadata.yaml``."""
    from pathlib import Path

    meta_path = Path("data") / "datasets" / folder_name / "metadata.yaml"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if meta_path.is_file():
        try:
            loaded = yaml.safe_load(meta_path.read_text()) or {}
            if isinstance(loaded, dict):
                existing = loaded
        except yaml.YAMLError:
            existing = {}
    existing[dataset_name] = stats
    meta_path.write_text(yaml.safe_dump(existing, sort_keys=False, default_flow_style=False))
    print(f"  metadata saved to {meta_path}")
