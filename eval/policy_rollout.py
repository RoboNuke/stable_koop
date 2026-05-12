"""Policy-rollout evaluation: success rate + per-trajectory metrics.

Ported from ``launch/eval_policy.py``. The legacy module-level
``make_policy`` / ``make_eval_env`` / ``make_single_env`` helpers moved to
:mod:`policy` and :mod:`data.env_builder`; only the rollout/scoring logic
lives here.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import yaml


def _parse_states(states, obs_type: Optional[str] = None):
    """Extract ``(theta, cos_th, thdot)`` from a state array (pendulum)."""
    if obs_type is not None:
        use_theta = obs_type == "theta"
    else:
        use_theta = states.shape[-1] != 3
    if not use_theta:
        cos_th, sin_th, thdot = states[..., 0], states[..., 1], states[..., 2]
        theta = np.arctan2(sin_th, cos_th)
    else:
        theta, thdot = states[..., 0], states[..., 1]
        cos_th = np.cos(theta)
    return theta, cos_th, thdot


def check_success(states, cfg):
    """True iff the final ``success_hold_steps`` are inside the success region."""
    hold = cfg["success_hold_steps"]
    if len(states) < hold:
        return False
    tail = np.array(states[-hold:])
    theta, cos_th, thdot = _parse_states(tail)
    angle_ok = np.all(np.abs(theta) < np.radians(cfg["success_angle_deg"]))
    vel_ok = np.all(np.abs(thdot) < cfg["success_max_thdot"])
    return bool(angle_ok and vel_ok)


def rollout(env, policy, max_steps, cfg):
    """One episode, terminating on success. Returns ``(states, actions, success, total_reward)``."""
    obs, _ = env.reset()
    states = [obs]
    actions = []
    total_reward = 0.0
    success = False
    for _ in range(max_steps):
        action = policy(obs) if policy is not None else env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        states.append(obs)
        actions.append(action)
        total_reward += reward
        if check_success(states, cfg):
            success = True
            break
        if done:
            break
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32).reshape(-1, 1),
        success,
        total_reward,
    )


def compute_trajectory_metrics(states, actions):
    """Per-trajectory mean-absolute metrics (pendulum)."""
    theta, cos_th, thdot = _parse_states(states)
    energy = np.abs(0.5 * thdot ** 2 + 10.0 * (1.0 - cos_th))
    return {
        "length": len(actions),
        "energy": float(np.mean(energy)),
        "control_torque": float(np.mean(np.abs(actions))),
        "angular_velocity": float(np.mean(np.abs(thdot))),
        "reward": 0.0,
    }


def group_stats(metrics_list):
    """Mean/std per metric across a list of per-trajectory dicts."""
    if not metrics_list:
        return {
            k: {"mean": 0.0, "std": 0.0}
            for k in ["length", "energy", "control_torque", "angular_velocity", "reward"]
        }
    stats = {}
    for key in metrics_list[0]:
        vals = np.array([m[key] for m in metrics_list])
        stats[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats


def print_stats_table(results):
    print(
        f"\nTrajectories: {results['num_trajectories']}  |  "
        f"Success: {results['num_success']}  |  "
        f"Failure: {results['num_failure']}  |  "
        f"Rate: {results['success_rate']:.1%}\n"
    )
    header = f"{'Metric':<20} {'Combined':>20} {'Success':>20} {'Failure':>20}"
    print(header)
    print("-" * len(header))
    for key in ["length", "energy", "control_torque", "angular_velocity", "reward"]:
        parts = []
        for group in ["combined", "success", "failure"]:
            m = results[group][key]["mean"]
            s = results[group][key]["std"]
            parts.append(f"{m:8.3f} +/- {s:6.3f}")
        print(f"{key:<20} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")
    print()


def load_eval_stats(filepath):
    """Load evaluation stats YAML."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def _vectorized_evaluate(vec_env, policy, cfg):
    num_traj = cfg["eval_num_trajectories"]
    max_steps = cfg["eval_max_steps"]
    num_envs = vec_env.num_envs

    all_states, all_actions, all_successes, all_rewards = [], [], [], []
    while len(all_states) < num_traj:
        remaining = num_traj - len(all_states)
        active_envs = min(num_envs, remaining)
        obs_batch, _ = vec_env.reset()
        states = [[obs_batch[i].copy()] for i in range(num_envs)]
        actions = [[] for _ in range(num_envs)]
        rewards = [0.0] * num_envs
        done_flags = [False] * num_envs
        success_flags = [False] * num_envs
        has_batch = hasattr(policy, "batch") and callable(policy.batch)

        for step in range(max_steps):
            if policy is None:
                action_batch = np.array(
                    [vec_env.single_action_space.sample() for _ in range(num_envs)]
                )
            elif has_batch:
                action_batch = policy.batch(obs_batch)
            else:
                action_batch = np.array([policy(obs_batch[i]) for i in range(num_envs)])

            obs_batch, rew_batch, terminated, truncated, infos = vec_env.step(action_batch)
            dones = terminated | truncated
            for i in range(active_envs):
                if done_flags[i]:
                    continue
                actions[i].append(action_batch[i].copy())
                rewards[i] += float(rew_batch[i])
                if dones[i]:
                    done_flags[i] = True
                    success_flags[i] = check_success(states[i], cfg)
                else:
                    states[i].append(obs_batch[i].copy())
                    if check_success(states[i], cfg):
                        success_flags[i] = True
                        done_flags[i] = True
            if all(done_flags[:active_envs]):
                break

        for i in range(active_envs):
            if len(all_states) >= num_traj:
                break
            s = np.array(states[i], dtype=np.float32)
            a = (
                np.array(actions[i], dtype=np.float32).reshape(-1, 1)
                if actions[i]
                else np.empty((0, 1), dtype=np.float32)
            )
            all_states.append(s)
            all_actions.append(a)
            all_successes.append(success_flags[i])
            all_rewards.append(rewards[i])

    return all_states, all_actions, all_successes, all_rewards


def evaluate(env, policy, cfg):
    """Evaluate ``policy`` on ``env``. Returns ``(results, all_states, all_actions)``."""
    num_traj = cfg["eval_num_trajectories"]
    is_vectorized = hasattr(env, "num_envs")
    if is_vectorized:
        all_states, all_actions, all_successes, all_rewards = _vectorized_evaluate(
            env, policy, cfg
        )
    else:
        max_steps = cfg["eval_max_steps"]
        all_states, all_actions, all_successes, all_rewards = [], [], [], []
        for _ in range(num_traj):
            states, actions, success, total_reward = rollout(env, policy, max_steps, cfg)
            all_states.append(states)
            all_actions.append(actions)
            all_successes.append(success)
            all_rewards.append(total_reward)

    success_metrics, failure_metrics, all_metrics = [], [], []
    for states, actions, success, reward in zip(
        all_states, all_actions, all_successes, all_rewards
    ):
        metrics = compute_trajectory_metrics(states, actions)
        metrics["reward"] = float(reward)
        all_metrics.append(metrics)
        if success:
            success_metrics.append(metrics)
        else:
            failure_metrics.append(metrics)

    results = {
        "num_trajectories": num_traj,
        "num_success": len(success_metrics),
        "num_failure": len(failure_metrics),
        "success_rate": float(len(success_metrics) / num_traj),
        "success": group_stats(success_metrics),
        "failure": group_stats(failure_metrics),
        "combined": group_stats(all_metrics),
    }
    print_stats_table(results)
    return results, all_states, all_actions
