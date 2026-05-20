"""CLI: ``python -m train_residual --config <yaml>``.

Loads the Koopman model + LQR from filesystem paths in the cfg and runs
SAC residual training against a stability-aware env wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from config.manager import ConfigManager, TrainResidualCfg
from data.dataloader import load_dataset
from data.env_builder import make_eval_env, make_single_env
from train_koopman.checkpointing import load_koopman_experiment, make_device


def _resolve_koopman_dir(koopman_path: str) -> Path:
    p = Path(koopman_path)
    if p.is_file():
        return p.parent
    return p


def _load_lqr(controller_path: str):
    """Load the LQR ``F`` and the stability bound ``gamma_max``.

    Returns a lightweight object with ``.F`` (torch tensor) and
    ``.gain_norm`` (torch scalar), plus the loaded ``gamma_max`` float.
    """
    ctrl_dir = Path(controller_path)
    lqr_pt = ctrl_dir / "lqr.pt"
    perf_yaml = ctrl_dir / "ctrl_performance.yaml"
    if not lqr_pt.is_file():
        raise FileNotFoundError(f"Expected LQR weights at {lqr_pt}")
    if not perf_yaml.is_file():
        raise FileNotFoundError(
            f"Expected ctrl_performance.yaml at {perf_yaml} (needed for gamma_max)"
        )

    raw = torch.load(lqr_pt, map_location="cpu")
    F = raw["F"]

    class _LoadedLQR:
        def __init__(self, F):
            self.F = F
            self.gain_norm = torch.linalg.matrix_norm(F, ord=2)

    perf = yaml.safe_load(perf_yaml.read_text())
    bound = perf.get("bound") or {}
    if "gamma_max" not in bound:
        raise KeyError(
            f"'bound.gamma_max' not found in {perf_yaml}; was stability "
            "analysis run for this controller?"
        )
    gamma_max = float(bound["gamma_max"])
    return _LoadedLQR(F), gamma_max


def _env_info_from_dataset(dataset_name: str):
    """Pull ``(env_name, env_kwargs)`` from the dataset's saved gather snapshot."""
    ds = load_dataset(dataset_name)
    snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = snap["env_name"]
    env_kwargs = snap.get("env_kwargs", {}) or {}
    return env_name, env_kwargs


def run(cfg: TrainResidualCfg) -> str:
    device = make_device()
    koopman_dir = _resolve_koopman_dir(cfg.koopman_path)
    model, train_cfg, _state_dim, _action_dim = load_koopman_experiment(
        experiment_name=koopman_dir.name, device=device, ckpt_path=koopman_dir,
    )
    lqr, gamma_max = _load_lqr(cfg.controller_path)
    env_name, env_kwargs = _env_info_from_dataset(train_cfg.dataset_name)

    def make_env_fn():
        return make_single_env(env_name=env_name, env_kwargs=env_kwargs)

    def make_eval_env_fn():
        return make_eval_env(
            env_name=env_name,
            num_parallel_evals=cfg.num_envs,
            env_kwargs=env_kwargs,
        )

    flat = {
        "num_envs": cfg.num_envs,
        "total_timesteps": cfg.total_timesteps,
        "eval_interval": cfg.eval_interval,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "actor_hidden_size": cfg.actor_hidden_size,
        "actor_hidden_layers": cfg.actor_hidden_layers,
        "critic_hidden_size": cfg.critic_hidden_size,
        "critic_hidden_layers": cfg.critic_hidden_layers,
        "gamma": cfg.gamma,
        "tau": cfg.tau,
        "initial_entropy_value": cfg.initial_entropy_value,
        "random_timesteps": cfg.random_timesteps,
        "learning_starts": cfg.learning_starts,
        "memory_size": cfg.memory_size,
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

    def evaluate_for_env(env, policy, eval_cfg):
        return evaluate(env, policy, eval_cfg, env_name=env_name)

    out_dir = Path("train_residual") / "weights" / cfg.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    from train_residual.sac import train_residual as _train

    _train(
        koopman_model=model,
        lqr=lqr,
        gamma_max=gamma_max,
        gamma_worst_case=cfg.gamma_worst_case,
        reward_weight=cfg.reward_weight,
        pred_error_space=cfg.pred_error_space,
        z_ref_max_mode=cfg.z_ref_max_mode,
        obs_augmentation=cfg.obs_augmentation,
        disable_action_augmentation=cfg.disable_action_augmentation,
        cfg=flat,
        run_dir=str(out_dir),
        make_env_fn=make_env_fn,
        make_eval_env_fn=make_eval_env_fn,
        evaluate_fn=evaluate_for_env,
        device=device,
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
