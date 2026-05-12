"""Trajectory collection + augmentation + dataset writing.

CLI: ``python -m data.gather_data --config config/exp_cfgs/gather_data/<exp>.yaml``

Writes to ``data/datasets/<dataset_name>.npz``. The .npz contains:
* ``base_states_<i>``, ``base_actions_<i>`` — base policy trajectories.
* ``pert_states_<i>``, ``pert_base_actions_<i>``, ``pert_perturbations_<i>`` —
  perturbed trajectories (for B fitting).
* ``obs_scale``, ``act_scale`` — per-dimension normalization vectors.
* ``num_base_trajectories``, ``num_pert_trajectories``, ``config_yaml`` —
  metadata for downstream consumers.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import yaml

from config.manager import ConfigManager, GatherDataCfg
from data.env_builder import make_single_env
from policy import make_policy


# ---------------------------------------------------------------------------
# Numeric helpers — single source of truth for env-derived scales.
# ---------------------------------------------------------------------------


def compute_obs_scale(env, augment: bool) -> np.ndarray:
    """Max absolute value per observation dimension (and action dim if augmenting)."""
    obs_scale = np.maximum(
        np.abs(env.observation_space.high), np.abs(env.observation_space.low)
    )
    if augment:
        act_scale = np.maximum(
            np.abs(env.action_space.high), np.abs(env.action_space.low)
        )
        obs_scale = np.concatenate([obs_scale, act_scale])
    return obs_scale.astype(np.float32)


def compute_act_scale(env) -> np.ndarray:
    """Max absolute value per action dimension."""
    act_scale = np.maximum(
        np.abs(env.action_space.high), np.abs(env.action_space.low)
    )
    return act_scale.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory collection.
# ---------------------------------------------------------------------------


def collect_data(env, num_trajectories, max_steps, seed, policy=None):
    """Collect trajectories using ``policy`` (random if None)."""
    np.random.seed(seed)
    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        actions = []
        for _ in range(max_steps):
            action = policy(obs) if policy else env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            if terminated or truncated:
                break
        trajectories.append((
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
        ))
    return trajectories


def collect_perturbed_data(
    env,
    policy,
    num_trajectories: int,
    max_steps: int,
    seed: int,
    perturb_scale: float = 1.0,
    fix_perturb_range: bool = False,
    hold_steps: int = 1,
):
    """Collect trajectories with base policy + uniform random perturbation.

    Returns list of ``(states, base_actions, perturbations)`` per trajectory.
    The env applies ``clip(base_action + perturbation, low, high)``.
    """
    np.random.seed(seed)
    action_low = env.action_space.low
    action_high = env.action_space.high
    scale = perturb_scale if perturb_scale is not None else 1.0
    perturb_low_default = action_low * scale
    perturb_high_default = action_high * scale
    perturb_mag = np.abs(action_high) * scale

    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        base_actions = []
        perturbations = []
        perturbation = None
        for t in range(max_steps):
            base_action = policy(obs) if policy is not None else np.zeros_like(action_low)
            if t % hold_steps == 0:
                if fix_perturb_range:
                    p_low = np.clip(action_low - base_action, -perturb_mag, 0)
                    p_high = np.clip(action_high - base_action, 0, perturb_mag)
                    perturbation = np.random.uniform(p_low, p_high).astype(np.float32)
                else:
                    perturbation = np.random.uniform(
                        perturb_low_default, perturb_high_default
                    ).astype(np.float32)
            total_action = np.clip(base_action + perturbation, action_low, action_high)
            obs, _, terminated, truncated, _ = env.step(total_action)
            states.append(obs)
            base_actions.append(base_action)
            perturbations.append(perturbation)
            if terminated or truncated:
                break
        trajectories.append((
            np.array(states, dtype=np.float32),
            np.array(base_actions, dtype=np.float32).reshape(-1, 1),
            np.array(perturbations, dtype=np.float32).reshape(-1, 1),
        ))
    return trajectories


# ---------------------------------------------------------------------------
# Augmentation — turn raw env trajectories into Koopman-state trajectories.
# ---------------------------------------------------------------------------


def augment_trajectories(trajectories, augment=True, obs_scale=None, act_scale=None):
    """Prepare base trajectories for autonomous Koopman training.

    With ``augment=True`` concatenates base policy actions into states so the
    base policy is absorbed into the autonomous dynamics. Always returns zero
    actions so B has no influence.
    """
    result = []
    for states, actions in trajectories:
        if augment:
            koopman_states = np.concatenate([states[:-1], actions], axis=-1)
        else:
            koopman_states = states[:-1]
        if obs_scale is not None:
            koopman_states = koopman_states / obs_scale
        zero_actions = np.zeros(
            (len(koopman_states) - 1, actions.shape[-1]), dtype=np.float32
        )
        result.append((koopman_states, zero_actions))
    return result


def augment_perturbed_trajectories(
    trajectories, augment=True, obs_scale=None, act_scale=None
):
    """Prepare perturbed trajectories for B-matrix training.

    Returns ``(koopman_states, normalized_perturbations)`` per trajectory.
    """
    result = []
    for states, base_actions, perturbations in trajectories:
        if augment:
            koopman_states = np.concatenate([states[:-1], base_actions], axis=-1)
        else:
            koopman_states = states[:-1]
        if obs_scale is not None:
            koopman_states = koopman_states / obs_scale
        norm_perturbations = perturbations[:-1]
        if act_scale is not None:
            norm_perturbations = norm_perturbations / act_scale
        result.append((koopman_states, norm_perturbations))
    return result


# ---------------------------------------------------------------------------
# CLI orchestration.
# ---------------------------------------------------------------------------


def gather(cfg: GatherDataCfg) -> Path:
    """Run the full gather pipeline and write ``data/datasets/<dataset_name>.npz``.

    Returns the written file path.
    """
    env = make_single_env(env_name=cfg.env_name, env_kwargs=cfg.env_kwargs)

    base_policy = make_policy(cfg.base_policy.name, **cfg.base_policy.params)
    perturb_policy = make_policy(
        cfg.perturbation.analytical_B_policy, **cfg.perturbation.params
    )

    print(
        f"[gather_data] env={cfg.env_name} env_kwargs={cfg.env_kwargs} "
        f"base_policy={cfg.base_policy.name} "
        f"perturb_policy={cfg.perturbation.analytical_B_policy}"
    )

    base_trajs = collect_data(
        env,
        num_trajectories=cfg.num_trajectories,
        max_steps=cfg.max_episode_steps,
        seed=cfg.seed,
        policy=base_policy,
    )
    print(f"[gather_data] collected {len(base_trajs)} base trajectories")

    pert_trajs = collect_perturbed_data(
        env,
        policy=perturb_policy,
        num_trajectories=cfg.num_trajectories,
        max_steps=cfg.max_episode_steps,
        seed=cfg.seed + 1,
        perturb_scale=cfg.perturbation.perturb_scale,
        fix_perturb_range=cfg.perturbation.fix_perturb_range,
        hold_steps=cfg.perturbation.hold_steps,
    )
    print(f"[gather_data] collected {len(pert_trajs)} perturbed trajectories")

    obs_scale = compute_obs_scale(env, augment=False)
    act_scale = compute_act_scale(env)

    save_path = save_dataset(
        dataset_name=cfg.dataset_name,
        base_trajectories=base_trajs,
        perturbed_trajectories=pert_trajs,
        obs_scale=obs_scale,
        act_scale=act_scale,
        cfg=cfg,
    )
    print(f"[gather_data] wrote {save_path}")
    return save_path


def save_dataset(
    dataset_name: str,
    base_trajectories,
    perturbed_trajectories,
    obs_scale: np.ndarray,
    act_scale: np.ndarray,
    cfg: GatherDataCfg,
) -> Path:
    """Canonical .npz writer; pairs with :func:`data.dataloader.load_dataset`."""
    out_dir = Path("data") / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}.npz"

    save_dict = {
        "obs_scale": obs_scale,
        "act_scale": act_scale,
        "state_dim": np.array(int(obs_scale.shape[0])),
        "action_dim": np.array(int(act_scale.shape[0])),
        "num_base_trajectories": np.array(len(base_trajectories)),
        "num_pert_trajectories": np.array(len(perturbed_trajectories)),
        "config_yaml": np.array(
            yaml.safe_dump(
                {"gather_data_cfg": dataclasses.asdict(cfg)},
                sort_keys=False,
            )
        ),
    }
    for i, (s, a) in enumerate(base_trajectories):
        save_dict[f"base_states_{i}"] = s
        save_dict[f"base_actions_{i}"] = a
    for i, (s, ba, p) in enumerate(perturbed_trajectories):
        save_dict[f"pert_states_{i}"] = s
        save_dict[f"pert_base_actions_{i}"] = ba
        save_dict[f"pert_perturbations_{i}"] = p
    np.savez(out_path, **save_dict)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Gather trajectories for stable_koop.")
    parser.add_argument("--config", required=True, help="Per-stage gather_data YAML.")
    args = parser.parse_args()

    cfg: GatherDataCfg = ConfigManager.load_stage(args.config, "gather_data_cfg")
    gather(cfg)


if __name__ == "__main__":
    main()
