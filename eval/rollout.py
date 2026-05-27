"""Single source of truth for evaluation rollouts.

Three pieces live here:

* :func:`build_residual_eval_env` — assembles the
  ``SyncVectorEnv → GymVectorAdapter → ResidualPolicyEnv → TensorToGymAdapter``
  stack used by every eval mode (``base`` / ``lqr`` / ``residual``).
* :func:`do_rollout` — the gym loop. Always saves ``states``,
  ``applied_action``, total ``reward``, and ``success``. When
  ``capture_per_step=True`` it additionally collects every batched value
  the wrapper writes to ``info`` (``gamma_t``, ``eta_t``,
  ``stability_term``, ``env_reward``, ``z_t``, ``z_next``, ``z_pred``,
  ``z_ref_t_base``, ``z_ref_t_res``, ``x_pred``, …), under the wrapper's
  own names. No env
  knowledge, no key renaming.
* :func:`group_stats`, :func:`print_stats_table`, :func:`load_eval_stats`,
  :func:`save_trajectories_npz` — small helpers used by the orchestrator.

Per-trajectory task metrics come from the env scorer's
``compute_metrics(states, applied_action)`` and live in the yaml summary —
not in the npz. The rollout never computes env-specific quantities itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

from eval import EnvScorer, get_env_scorer


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def group_stats(metrics_list):
    """Mean/std per metric across a list of per-trajectory dicts."""
    if not metrics_list:
        return {}
    stats: dict = {}
    for key in metrics_list[0]:
        vals = np.array([m[key] for m in metrics_list])
        stats[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats


def print_stats_table(results, metric_keys):
    print(
        f"\nTrajectories: {results['num_trajectories']}  |  "
        f"Success: {results['num_success']}  |  "
        f"Failure: {results['num_failure']}  |  "
        f"Rate: {results['success_rate']:.1%}\n"
    )
    header = f"{'Metric':<20} {'Combined':>24} {'Success':>24} {'Failure':>24}"
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        parts = []
        for group in ["combined", "success", "failure"]:
            m = results[group].get(key, {}).get("mean", 0.0)
            s = results[group].get(key, {}).get("std", 0.0)
            parts.append(f"{m:10.4f} +/- {s:8.4g}")
        print(f"{key:<20} {parts[0]:>24} {parts[1]:>24} {parts[2]:>24}")
    print()


def load_eval_stats(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)


def success_cfg_from_eval_cfg(eval_cfg) -> dict:
    """The threshold dict consumed by env scorers."""
    return {
        "success_angle_deg": eval_cfg.success_angle_deg,
        "success_max_thdot": eval_cfg.success_max_thdot,
        "success_max_cart_vel": eval_cfg.success_max_cart_vel,
        "success_hold_steps": eval_cfg.success_hold_steps,
        "max_steps": eval_cfg.eval_max_steps,
    }


# ----------------------------------------------------------------------
# env stack
# ----------------------------------------------------------------------

# Mode-specific wrapper config. ``reward_weight`` is always 0 at eval time
# (the stability term lives on ``info``; we don't want it folded into the
# augmented reward and muddying task-metric returns).
_MODE_KWARGS = {
    "base":     {"disable_action_augmentation": True,  "obs_augmentation": "none"},
    "lqr":      {"disable_action_augmentation": False, "obs_augmentation": "none"},
    "residual": {"disable_action_augmentation": False, "obs_augmentation": None},
}


def build_residual_eval_env(
    *,
    mode: str,
    make_single_env_fn: Callable[[], gym.Env],
    num_envs: int,
    koopman_model,
    lqr,
    gamma_max: float,
    device: torch.device,
    env_name: str,
    pred_error_space: str = "latent",
    z_ref_max_mode: str = "action_bound",
    gamma_worst_case: float = 0.0,
    obs_augmentation_override: Optional[str] = None,
    obs_scale=None,
    act_scale=None,
    aug_cfg=None,
):
    """Build the batched eval env for one mode.

    Returns a ``TensorToGymAdapter``-wrapped env that exposes the same numpy
    ``VectorEnv``-shaped interface that :func:`do_rollout` expects.

    ``obs_scale`` / ``act_scale`` / ``aug_cfg`` must reflect the koopman
    model's training-time augmentation so the wrapper feeds the encoder and
    predict step the same normalized inputs they were trained on (matching
    the LQR analysis's coverage computation).
    """
    from data.env_builder import get_base_goal_fn
    from wrappers.gym_vec_adapter import GymVectorAdapter, TensorToGymAdapter
    from wrappers.residual import ResidualPolicyEnv

    base_goal_fn = get_base_goal_fn(env_name)

    if mode not in _MODE_KWARGS:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_MODE_KWARGS)}")

    mode_kwargs = dict(_MODE_KWARGS[mode])
    if mode == "residual":
        if obs_augmentation_override is None:
            raise ValueError(
                "residual mode needs obs_augmentation_override (the value the "
                "actor was trained with)."
            )
        mode_kwargs["obs_augmentation"] = obs_augmentation_override

    vec_env = gym.vector.SyncVectorEnv(
        [make_single_env_fn for _ in range(num_envs)]
    )
    tensor_env = GymVectorAdapter(vec_env, device=device)
    wrapped = ResidualPolicyEnv(
        tensor_env,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        gamma_worst_case=gamma_worst_case,
        reward_weight=0.0,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        obs_augmentation=mode_kwargs["obs_augmentation"],
        disable_action_augmentation=mode_kwargs["disable_action_augmentation"],
        device=device,
        base_goal_fn=base_goal_fn,
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
    )
    return TensorToGymAdapter(wrapped)


# ----------------------------------------------------------------------
# rollout
# ----------------------------------------------------------------------

def _batched_to_np(value, num_envs: int) -> Optional[np.ndarray]:
    """Coerce an ``info`` entry to a per-env numpy array, or return ``None``.

    Accepts torch tensors and array-likes. Returns ``None`` for scalars or
    arrays whose first dim doesn't match ``num_envs`` — those aren't per-env
    diagnostics and can't be captured per-step.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        try:
            arr = np.asarray(value)
        except Exception:
            return None
    if arr.ndim == 0 or arr.shape[0] != num_envs:
        return None
    return arr


def _action_dim(env) -> int:
    space = env.single_action_space
    return int(np.prod(space.shape)) if space.shape else 1


def do_rollout(
    env,
    batch_policy: Callable[[np.ndarray], np.ndarray],
    *,
    num_trajectories: int,
    max_steps: int,
    env_name: str,
    success_cfg: dict,
    scorer: Optional[EnvScorer] = None,
    capture_per_step: bool = False,
):
    """Roll out ``batch_policy`` on ``env`` and assemble metrics.

    ``env`` must expose the gymnasium ``VectorEnv`` numpy interface
    (``num_envs``, ``single_observation_space``, ``single_action_space``,
    ``reset()``/``step()`` returning numpy arrays). Use
    :func:`build_residual_eval_env` to assemble one.

    Returns ``(task_results, per_trajectory)``. Every trajectory always has
    ``states`` (T+1, obs_dim), ``applied_action`` (T, action_dim), total
    ``reward``, and ``success``. When ``capture_per_step=True``, every
    batched value the wrapper put on ``info`` (per-step, length T, only
    valid pre-termination steps) is also captured under the same key the
    wrapper used — no rename, no env-specific calculations.
    """
    scorer = scorer if scorer is not None else get_env_scorer(env_name)
    check_success = scorer.check_success
    compute_metrics = scorer.compute_metrics

    num_envs = int(env.num_envs)
    per_trajectory: list[dict] = []

    while len(per_trajectory) < num_trajectories:
        remaining = num_trajectories - len(per_trajectory)
        active = min(num_envs, remaining)

        obs_batch, _ = env.reset()
        states: list[list[np.ndarray]] = [[obs_batch[i].copy()] for i in range(num_envs)]
        applied_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        rewards: list[float] = [0.0] * num_envs
        done_flags: list[bool] = [False] * num_envs
        success_flags: list[bool] = [False] * num_envs

        # Generic per-env per-key info buffer (only when capture_per_step).
        info_bufs: list[dict[str, list]] = (
            [dict() for _ in range(num_envs)] if capture_per_step else []
        )

        for _step in range(max_steps):
            action_batch = batch_policy(obs_batch)
            obs_batch, rew_batch, terminated, truncated, infos = env.step(action_batch)
            dones = np.asarray(terminated) | np.asarray(truncated)

            # Decode info batch-by-batch into numpy once per step.
            info_np: dict[str, np.ndarray] = {}
            if capture_per_step:
                for k, v in infos.items():
                    arr = _batched_to_np(v, num_envs)
                    if arr is not None:
                        info_np[k] = arr
            applied_np = info_np.get("applied_action") if capture_per_step else (
                _batched_to_np(infos.get("applied_action"), num_envs)
            )

            for i in range(active):
                if done_flags[i]:
                    continue
                if applied_np is not None:
                    applied_actions[i].append(np.asarray(applied_np[i]))
                else:
                    # Plain SyncVectorEnv (no wrapper) — env action == policy output.
                    applied_actions[i].append(action_batch[i].copy())
                rewards[i] += float(rew_batch[i])

                # Always append the post-step obs. In gymnasium 1.x
                # SyncVectorEnv with NEXT_STEP auto-reset (the default), the
                # obs returned at the done step IS the terminal observation
                # (the env auto-resets only on the *next* step call, which we
                # never reach because we ``continue`` once done_flags[i] is
                # True). This is exactly what data/gather_data.py records and
                # what inverted_pendulum_check_success expects:
                # ``len(states) >= max_steps + 1`` requires the final obs.
                states[i].append(obs_batch[i].copy())

                if capture_per_step:
                    for k, batch_arr in info_np.items():
                        info_bufs[i].setdefault(k, []).append(np.asarray(batch_arr[i]))

                if dones[i]:
                    done_flags[i] = True
                    success_flags[i] = check_success(states[i], success_cfg)
                elif check_success(states[i], success_cfg):
                    success_flags[i] = True
                    done_flags[i] = True

            if all(done_flags[:active]):
                break

        for i in range(active):
            if len(per_trajectory) >= num_trajectories:
                break
            s_arr = np.asarray(states[i], dtype=np.float32)
            applied_arr = (
                np.asarray(applied_actions[i], dtype=np.float32)
                if applied_actions[i]
                else np.empty((0, _action_dim(env)), dtype=np.float32)
            )
            traj: dict = {
                "states": s_arr,
                "applied_action": (
                    applied_arr.reshape(applied_arr.shape[0], -1)
                    if applied_arr.size else applied_arr
                ),
                "reward": rewards[i],
                "success": success_flags[i],
            }
            if capture_per_step:
                for k, vals in info_bufs[i].items():
                    if not vals:
                        continue
                    stacked = np.stack([np.asarray(v) for v in vals])
                    traj[k] = stacked.astype(np.float32) if stacked.dtype.kind == "f" else stacked
            per_trajectory.append(traj)

    # Per-trajectory task metrics (env scorer's own keys).
    success_metrics, failure_metrics, all_metrics = [], [], []
    for traj in per_trajectory:
        m = compute_metrics(traj["states"], traj["applied_action"])
        m["reward"] = float(traj["reward"])
        all_metrics.append(m)
        (success_metrics if traj["success"] else failure_metrics).append(m)

    n = len(per_trajectory)
    n_succ = len(success_metrics)
    n_fail = len(failure_metrics)
    results = {
        "num_trajectories": n,
        "num_success": n_succ,
        "num_failure": n_fail,
        "success_rate": (n_succ / n) if n else 0.0,
        "combined": group_stats(all_metrics),
        "success": group_stats(success_metrics),
        "failure": group_stats(failure_metrics),
    }
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    if metric_keys:
        print_stats_table(results, metric_keys)
    return results, per_trajectory


# ----------------------------------------------------------------------
# npz save
# ----------------------------------------------------------------------

def save_trajectories_npz(out_path: str | Path, per_trajectory: list[dict]) -> Path:
    """Persist per-trajectory arrays under ``<key>_<i>`` (variable-length safe)."""
    out_path = Path(out_path)
    save_dict: dict = {"num_trajectories": np.array(len(per_trajectory))}
    for i, traj in enumerate(per_trajectory):
        for key, arr in traj.items():
            if isinstance(arr, (bool, np.bool_)):
                save_dict[f"{key}_{i}"] = np.array(bool(arr))
            elif isinstance(arr, (int, float)):
                save_dict[f"{key}_{i}"] = np.array(arr)
            else:
                save_dict[f"{key}_{i}"] = np.asarray(arr)
    np.savez(out_path, **save_dict)
    return out_path


def concat_valid_per_step(
    per_trajectory: list[dict], keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Flatten the named per-step arrays across all trajectories.

    Missing keys (e.g. when ``capture_per_step=False`` so nothing landed in
    the npz) come back as empty ``float32`` arrays.
    """
    out: dict[str, np.ndarray] = {}
    for key in keys:
        chunks = [t[key] for t in per_trajectory if key in t and len(t[key])]
        out[key] = (
            np.concatenate(chunks)
            if chunks else np.empty((0,), dtype=np.float32)
        )
    return out
