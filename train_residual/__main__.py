"""CLI: ``python -m train_residual --config <yaml>``.

Loads the referenced Koopman model + LQR, reconstructs the base policy from
the dataset metadata snapshot, and runs SAC residual training.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import torch
import yaml

from config.manager import ConfigManager, TrainResidualCfg
from data.dataloader import load_dataset
from data.env_builder import make_eval_env, make_single_env
from controller.lqr.lqr import LQR
from policy import make_policy
from train_koopman.checkpointing import build_koopman_model, make_device


def _load_koopman_artifacts(experiment_name: str, device):
    ckpt_path = Path("train_koopman") / "weights" / experiment_name / "koopman_ckpt.pt"
    raw = torch.load(ckpt_path, map_location=device)
    koop_cfg = raw["config"]
    model, _ = build_koopman_model(koop_cfg, augment=True, device=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, koop_cfg


def _load_lqr(name: str):
    lqr_path = Path("controller") / "lqr" / "weights" / name / "lqr.pt"
    raw = torch.load(lqr_path, map_location="cpu")
    # Reconstruct minimal LQR-compatible object with the saved F/P.
    class _LoadedLQR:
        def __init__(self, F):
            self.F = F
            self.gain_norm = torch.linalg.norm(F, ord=2)
    return _LoadedLQR(raw["F"])


def _base_policy_from_dataset(dataset_name: str):
    """Reconstruct the base policy from the dataset's saved GatherDataCfg snapshot."""
    ds = load_dataset(dataset_name)
    snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    bp = snap["base_policy"]
    name = bp.pop("name")
    return make_policy(name, **bp), snap


def run(cfg: TrainResidualCfg) -> str:
    device = make_device()
    model, koop_cfg = _load_koopman_artifacts(cfg.koopman_experiment_name, device)
    lqr = _load_lqr(cfg.lqr_name)
    base_policy, gather_snap = _base_policy_from_dataset(koop_cfg["dataset_name"])

    env_kwargs = dict(
        env_name=gather_snap["env_name"],
        obs_type=gather_snap["obs_type"],
        limited_spawn=gather_snap["limited_spawn"],
        spawn_angle_range=gather_snap["spawn_angle_range"],
    )

    def make_env_fn():
        return make_single_env(**env_kwargs)

    def make_eval_env_fn():
        return make_eval_env(num_parallel_evals=cfg.num_envs, **env_kwargs)

    # Build a flat cfg dict the legacy training code expects.
    flat = {
        **koop_cfg,
        "residual_z_ref_limit": cfg.z_ref_limit,
        "residual_num_envs": cfg.num_envs,
        "residual_total_timesteps": cfg.total_timesteps,
        "residual_eval_interval": cfg.eval_interval,
        "residual_batch_size": cfg.batch_size,
        "residual_lr": cfg.lr,
        "residual_actor_hidden_size": cfg.actor_hidden_size,
        "residual_actor_hidden_layers": cfg.actor_hidden_layers,
        "residual_critic_hidden_size": cfg.critic_hidden_size,
        "residual_critic_hidden_layers": cfg.critic_hidden_layers,
        "residual_gamma": cfg.gamma,
        "residual_tau": cfg.tau,
        "residual_initial_entropy_value": cfg.initial_entropy_value,
        "residual_random_timesteps": cfg.random_timesteps,
        "residual_learning_starts": cfg.learning_starts,
        "residual_memory_size": cfg.memory_size,
        "experiment_name": cfg.experiment_name,
        # Eval-related fields used by the evaluate callback.
        "num_parallel_evals": cfg.num_envs,
        "eval_num_trajectories": 16,
        "eval_max_steps": 200,
        "eval_seed": cfg.seed,
        "success_angle_deg": 15.0,
        "success_max_thdot": 1.0,
        "success_hold_steps": 20,
    }

    from eval.policy_rollout import evaluate

    out_dir = Path("train_residual") / "weights" / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from train_residual.sac import train_residual as _train

    _train(
        base_policy=base_policy,
        lqr=lqr,
        cfg=flat,
        run_dir=str(out_dir),
        make_env_fn=make_env_fn,
        make_eval_env_fn=make_eval_env_fn,
        evaluate_fn=evaluate,
        z_ref_limit=cfg.z_ref_limit,
    )
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC residual policy.")
    parser.add_argument("--config", required=True, help="Per-stage train_residual YAML.")
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "train_residual_cfg")
    run(cfg)


if __name__ == "__main__":
    main()
