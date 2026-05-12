"""Trajectory collection + raw dataset writer.

CLI: ``python -m data.gather_data --config <yaml>``

One call → one dataset. The dataset is *raw*: states, actions, rewards, and
per-dimension bounds. Augmentation (state-with-action prepending) and
normalization happen at the consumer (see ``train_koopman/augmentation.py``).

Dataset .npz layout (per trajectory ``i``)::

    states_<i>        : (T+1, obs_dim)   raw env observations
    actions_<i>       : (T, action_dim)  final action applied to env (post-clip)
    base_actions_<i>  : (T, action_dim)  base policy contribution
    rewards_<i>       : (T,)             reward signal

Global keys::

    num_trajectories      : int
    state_dim, action_dim : int
    obs_space_low/high    : per-dim env.observation_space bounds
    act_space_low/high    : per-dim env.action_space bounds
    obs_min/max           : per-dim observed min/max across all collected states
    act_min/max           : per-dim observed min/max across all collected actions
    perturbations_enabled : bool
    config_yaml           : YAML snapshot of the GatherDataCfg used
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from config.manager import ConfigManager, GatherDataCfg
from data.env_builder import make_single_env
from policy import make_policy


# ---------------------------------------------------------------------------
# Trajectory collection.
# ---------------------------------------------------------------------------


def collect_trajectories(
    env,
    policy,
    *,
    num_trajectories: int,
    max_steps: int,
    seed: int,
    perturbations_enabled: bool,
    perturb_scale: float = 1.0,
    fix_perturb_range: bool = False,
    hold_steps: int = 1,
):
    """Roll out ``policy`` for ``num_trajectories`` episodes.

    Returns a list of ``(states, actions, base_actions, rewards)`` per trajectory.
    When ``perturbations_enabled`` is false the action equals the base action
    exactly (within clipping).
    """
    np.random.seed(seed)
    action_low = env.action_space.low
    action_high = env.action_space.high
    scale = perturb_scale if perturbations_enabled else 0.0
    perturb_low_default = action_low * scale
    perturb_high_default = action_high * scale
    perturb_mag = np.abs(action_high) * scale

    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        actions = []
        base_actions = []
        rewards = []
        perturbation = np.zeros_like(action_low, dtype=np.float32)

        for t in range(max_steps):
            base_action = (
                np.asarray(policy(obs), dtype=np.float32)
                if policy is not None
                else np.zeros_like(action_low, dtype=np.float32)
            )
            if perturbations_enabled and t % max(hold_steps, 1) == 0:
                if fix_perturb_range:
                    p_low = np.clip(action_low - base_action, -perturb_mag, 0)
                    p_high = np.clip(action_high - base_action, 0, perturb_mag)
                    perturbation = np.random.uniform(p_low, p_high).astype(np.float32)
                else:
                    perturbation = np.random.uniform(
                        perturb_low_default, perturb_high_default
                    ).astype(np.float32)

            total_action = np.clip(base_action + perturbation, action_low, action_high)
            obs, reward, terminated, truncated, _ = env.step(total_action)
            states.append(obs)
            actions.append(total_action)
            base_actions.append(base_action)
            rewards.append(reward)
            if terminated or truncated:
                break

        trajectories.append((
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, action_low.shape[0]),
            np.array(base_actions, dtype=np.float32).reshape(-1, action_low.shape[0]),
            np.array(rewards, dtype=np.float32),
        ))
    return trajectories


# ---------------------------------------------------------------------------
# Dataset I/O.
# ---------------------------------------------------------------------------


def save_dataset(
    dataset_name: str,
    trajectories,
    *,
    obs_space_low: np.ndarray,
    obs_space_high: np.ndarray,
    act_space_low: np.ndarray,
    act_space_high: np.ndarray,
    perturbations_enabled: bool,
    cfg: GatherDataCfg,
) -> Path:
    """Write the raw dataset .npz; companion to :func:`data.dataloader.load_dataset`."""
    out_dir = Path("data") / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}.npz"

    # Observed min/max across the actual collected data.
    all_states = np.concatenate([s for s, _, _, _ in trajectories], axis=0)
    all_actions = np.concatenate([a for _, a, _, _ in trajectories], axis=0)
    obs_min = all_states.min(axis=0).astype(np.float32)
    obs_max = all_states.max(axis=0).astype(np.float32)
    act_min = all_actions.min(axis=0).astype(np.float32)
    act_max = all_actions.max(axis=0).astype(np.float32)

    save_dict = {
        "num_trajectories": np.array(len(trajectories)),
        "state_dim": np.array(int(obs_space_low.shape[0])),
        "action_dim": np.array(int(act_space_low.shape[0])),
        "obs_space_low": obs_space_low.astype(np.float32),
        "obs_space_high": obs_space_high.astype(np.float32),
        "act_space_low": act_space_low.astype(np.float32),
        "act_space_high": act_space_high.astype(np.float32),
        "obs_min": obs_min,
        "obs_max": obs_max,
        "act_min": act_min,
        "act_max": act_max,
        "perturbations_enabled": np.array(perturbations_enabled),
        "config_yaml": np.array(
            yaml.safe_dump(
                {"gather_data_cfg": dataclasses.asdict(cfg)}, sort_keys=False
            )
        ),
    }
    for i, (s, a, ba, r) in enumerate(trajectories):
        save_dict[f"states_{i}"] = s
        save_dict[f"actions_{i}"] = a
        save_dict[f"base_actions_{i}"] = ba
        save_dict[f"rewards_{i}"] = r
    np.savez(out_path, **save_dict)
    return out_path


# ---------------------------------------------------------------------------
# Orchestrator + CLI.
# ---------------------------------------------------------------------------


def gather(cfg: GatherDataCfg) -> Path:
    """Run one gather call against ``cfg`` and write a single dataset."""
    env = make_single_env(env_name=cfg.env_name, env_kwargs=cfg.env_kwargs)
    base_policy = make_policy(cfg.base_policy.name, **cfg.base_policy.params)

    print(
        f"[gather_data] env={cfg.env_name} base_policy={cfg.base_policy.name} "
        f"perturbations_enabled={cfg.perturbations.enabled}"
    )

    trajectories = collect_trajectories(
        env,
        base_policy,
        num_trajectories=cfg.num_trajectories,
        max_steps=cfg.max_episode_steps,
        seed=cfg.seed,
        perturbations_enabled=cfg.perturbations.enabled,
        perturb_scale=cfg.perturbations.perturb_scale,
        fix_perturb_range=cfg.perturbations.fix_perturb_range,
        hold_steps=cfg.perturbations.hold_steps,
    )
    print(f"[gather_data] collected {len(trajectories)} trajectories "
          f"({sum(len(a) for _, a, _, _ in trajectories)} transitions)")

    obs_space_low = np.asarray(env.observation_space.low, dtype=np.float32)
    obs_space_high = np.asarray(env.observation_space.high, dtype=np.float32)
    act_space_low = np.asarray(env.action_space.low, dtype=np.float32)
    act_space_high = np.asarray(env.action_space.high, dtype=np.float32)

    out = save_dataset(
        cfg.dataset_name,
        trajectories,
        obs_space_low=obs_space_low,
        obs_space_high=obs_space_high,
        act_space_low=act_space_low,
        act_space_high=act_space_high,
        perturbations_enabled=cfg.perturbations.enabled,
        cfg=cfg,
    )
    print(f"[gather_data] wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather one trajectory dataset.")
    parser.add_argument("--config", required=True, help="Per-stage gather_data YAML.")
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "gather_data_cfg")
    gather(cfg)


if __name__ == "__main__":
    main()
