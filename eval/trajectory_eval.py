"""Trajectory-level evaluator with per-step Koopman + policy diagnostics.

Runs ``eval_cfg.eval_num_trajectories`` rollouts driven by
``applied_action = base_action + residual_action`` (with ``residual_action``
zero until a residual policy is provided). At every step we record:

* ``x_t``                 — raw observation
* ``base_action``         — base policy's action at ``x_t``
* ``residual_action``     — residual policy's action at ``x_t`` (zeros when no
                             residual is supplied)
* ``z_t``                 — ``model.encode(x_t)``
* ``z_pred``              — ``model.predict(z_t, applied_action)``
* ``x_pred``              — ``model.decode(z_pred)``
* ``reward``              — env reward emitted for ``applied_action`` at ``x_t``

The function also returns the full task-metric summary (same shape as
:func:`eval.policy_rollout.evaluate`), so callers can write it into
``ctrl_performance.yaml`` alongside the per-step arrays.

The per-step arrays are returned as a list of per-trajectory dicts. The
caller writes them out via :func:`save_trajectory_npz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from eval import EnvScorer, get_env_scorer
from eval.policy_rollout import group_stats, print_stats_table


def _encode_predict(model, x_np: np.ndarray, applied_action_np: np.ndarray, device):
    """Return ``(z_t, z_pred, x_pred)`` as 1-D numpy arrays."""
    with torch.no_grad():
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
        u_t = torch.tensor(applied_action_np, dtype=torch.float32, device=device).unsqueeze(0)
        z_t = model.encode(x_t)
        z_pred = model.predict(z_t, u_t)
        x_pred = model.decode(z_pred)
    return (
        z_t.squeeze(0).cpu().numpy(),
        z_pred.squeeze(0).cpu().numpy(),
        x_pred.squeeze(0).cpu().numpy(),
    )


def _rollout_with_metrics(
    env,
    base_policy,
    *,
    model,
    device,
    eval_cfg,
    scorer: EnvScorer,
    residual_policy=None,
    success_cfg: dict,
):
    """One episode; returns ``(states, applied_actions, success, total_reward, per_step)``.

    ``per_step`` includes ``residual_action`` only when ``residual_policy`` is
    provided — there's no point persisting an array of zeros otherwise.
    """
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.float32)

    states = [obs]
    applied_actions: list[np.ndarray] = []
    per_step: dict = {
        "x_t": [],
        "base_action": [],
        "z_t": [],
        "z_pred": [],
        "x_pred": [],
        "reward": [],
    }
    has_residual = residual_policy is not None
    if has_residual:
        per_step["residual_action"] = []
    total_reward = 0.0
    success = False

    for _ in range(eval_cfg.eval_max_steps):
        base_action = np.asarray(base_policy(obs), dtype=np.float32)
        if has_residual:
            residual_action = np.asarray(residual_policy(obs), dtype=np.float32)
            applied_action = base_action + residual_action
        else:
            applied_action = base_action

        z_t, z_pred, x_pred = _encode_predict(model, obs, applied_action, device)

        per_step["x_t"].append(obs.astype(np.float32))
        per_step["base_action"].append(base_action.astype(np.float32))
        per_step["z_t"].append(z_t.astype(np.float32))
        per_step["z_pred"].append(z_pred.astype(np.float32))
        per_step["x_pred"].append(x_pred.astype(np.float32))
        if has_residual:
            per_step["residual_action"].append(residual_action.astype(np.float32))

        obs, reward, terminated, truncated, _ = env.step(applied_action)
        obs = np.asarray(obs, dtype=np.float32)
        per_step["reward"].append(float(reward))
        total_reward += float(reward)
        states.append(obs)
        applied_actions.append(applied_action.astype(np.float32))

        if scorer.check_success(states, success_cfg):
            success = True
            break
        if terminated or truncated:
            break

    per_step = {k: np.asarray(v) for k, v in per_step.items()}
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(applied_actions, dtype=np.float32),
        success,
        total_reward,
        per_step,
    )


def evaluate_with_trajectories(
    env,
    base_policy,
    *,
    model,
    device,
    env_name: str,
    eval_cfg,
    success_cfg: dict,
    residual_policy=None,
) -> tuple[dict, list[dict]]:
    """Run rollouts, return ``(task_metrics_summary, per_trajectory_arrays)``.

    ``success_cfg`` is the same threshold dict that
    :func:`data.dataset_stats.evaluate_dataset` passes to the env scorer.
    """
    scorer = get_env_scorer(env_name)
    compute_metrics = scorer.compute_metrics

    all_metrics, success_metrics, failure_metrics = [], [], []
    per_trajectory: list[dict] = []

    n = int(eval_cfg.eval_num_trajectories)
    for traj_idx in range(n):
        states, applied_actions, success, total_reward, per_step = _rollout_with_metrics(
            env,
            base_policy,
            model=model,
            device=device,
            eval_cfg=eval_cfg,
            scorer=scorer,
            residual_policy=residual_policy,
            success_cfg=success_cfg,
        )
        metrics = compute_metrics(states, applied_actions)
        metrics["reward"] = float(total_reward)
        all_metrics.append(metrics)
        (success_metrics if success else failure_metrics).append(metrics)
        per_trajectory.append(per_step)

    n_success = len(success_metrics)
    n_failure = len(failure_metrics)
    results = {
        "num_trajectories": n,
        "num_success": n_success,
        "num_failure": n_failure,
        "success_rate": (n_success / n) if n else 0.0,
        "combined": group_stats(all_metrics),
        "success": group_stats(success_metrics),
        "failure": group_stats(failure_metrics),
    }
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    if metric_keys:
        print_stats_table(results, metric_keys)
    return results, per_trajectory


def save_trajectory_npz(out_path: str | Path, per_trajectory: list[dict]) -> Path:
    """Write per-trajectory per-step arrays to a single .npz."""
    out_path = Path(out_path)
    save_dict: dict = {"num_trajectories": np.array(len(per_trajectory))}
    for i, per_step in enumerate(per_trajectory):
        for key, arr in per_step.items():
            save_dict[f"{key}_{i}"] = arr
    np.savez(out_path, **save_dict)
    return out_path
