"""Policy-rollout evaluation: env-agnostic runner + per-env scoring registry.

The success criterion and per-trajectory metrics come from the env's registered
:class:`EnvScorer` (see :mod:`eval`). The runner itself only does the gym loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import yaml

from eval import EnvScorer, get_env_scorer


def rollout(env, policy, max_steps, cfg, *, check_success):
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


def group_stats(metrics_list):
    """Mean/std per metric across a list of per-trajectory dicts."""
    if not metrics_list:
        return {}
    stats = {}
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
    header = f"{'Metric':<20} {'Combined':>20} {'Success':>20} {'Failure':>20}"
    print(header)
    print("-" * len(header))
    for key in metric_keys:
        parts = []
        for group in ["combined", "success", "failure"]:
            m = results[group].get(key, {}).get("mean", 0.0)
            s = results[group].get(key, {}).get("std", 0.0)
            parts.append(f"{m:8.3f} +/- {s:6.3f}")
        print(f"{key:<20} {parts[0]:>20} {parts[1]:>20} {parts[2]:>20}")
    print()


def load_eval_stats(filepath):
    with open(filepath) as f:
        return yaml.safe_load(f)


def _vectorized_evaluate(vec_env, policy, cfg, *, check_success):
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


def evaluate(env, policy, cfg, *, env_name: str, scorer: Optional[EnvScorer] = None):
    """Evaluate ``policy`` on ``env``; returns ``(results, all_states, all_actions)``.

    ``env_name`` selects the registered :class:`EnvScorer` (success +
    per-trajectory metrics). Pass ``scorer`` directly to override the registry.
    """
    scorer = scorer if scorer is not None else get_env_scorer(env_name)
    check_success = scorer.check_success
    compute_metrics = scorer.compute_metrics

    num_traj = cfg["eval_num_trajectories"]
    is_vectorized = hasattr(env, "num_envs")
    if is_vectorized:
        all_states, all_actions, all_successes, all_rewards = _vectorized_evaluate(
            env, policy, cfg, check_success=check_success
        )
    else:
        max_steps = cfg["eval_max_steps"]
        all_states, all_actions, all_successes, all_rewards = [], [], [], []
        for _ in range(num_traj):
            states, actions, success, total_reward = rollout(
                env, policy, max_steps, cfg, check_success=check_success
            )
            all_states.append(states)
            all_actions.append(actions)
            all_successes.append(success)
            all_rewards.append(total_reward)

    success_metrics, failure_metrics, all_metrics = [], [], []
    for states, actions, success, reward in zip(
        all_states, all_actions, all_successes, all_rewards
    ):
        metrics = compute_metrics(states, actions)
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
    metric_keys = list(all_metrics[0].keys()) if all_metrics else []
    print_stats_table(results, metric_keys)
    return results, all_states, all_actions
