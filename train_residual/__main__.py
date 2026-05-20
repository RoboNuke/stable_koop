"""CLI: ``python -m train_residual --config <yaml>``.

Loads the Koopman model + LQR from filesystem paths in the cfg, runs SAC
residual training against the stability-aware env wrapper, and (by default)
runs the multi-mode post-training eval (base + LQR + residual) via
:func:`eval.multi_mode.run_multi_mode`.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
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
    """Pull ``(env_name, env_kwargs, base_policy_snap)`` from the dataset's saved gather snapshot."""
    ds = load_dataset(dataset_name)
    snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = snap["env_name"]
    env_kwargs = snap.get("env_kwargs", {}) or {}
    base_policy_snap = snap["base_policy"]
    return env_name, env_kwargs, base_policy_snap


def run(cfg: TrainResidualCfg, *, run_post_eval: bool = True) -> str:
    device = make_device()
    koopman_dir = _resolve_koopman_dir(cfg.koopman_path)
    model, train_cfg, _state_dim, _action_dim = load_koopman_experiment(
        experiment_name=koopman_dir.name, device=device, ckpt_path=koopman_dir,
    )
    lqr, gamma_max = _load_lqr(cfg.controller_path)
    env_name, env_kwargs, base_policy_snap = _env_info_from_dataset(train_cfg.dataset_name)

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
        "success_max_cart_vel": 0.5,
        "success_hold_steps": 20,
    }

    from eval.rollout import do_rollout

    def evaluate_for_env(env, policy, eval_cfg):
        """Compat shim for sac.py's periodic eval callback.

        Routes through :func:`eval.rollout.do_rollout` — the single rollout
        source of truth.
        """
        batch_fn = getattr(policy, "batch", None)
        if batch_fn is None:
            def batch_fn(obs_batch):
                return np.stack(
                    [np.asarray(policy(obs_batch[i])) for i in range(obs_batch.shape[0])]
                )
        success_cfg = {
            "success_angle_deg": eval_cfg["success_angle_deg"],
            "success_max_thdot": eval_cfg["success_max_thdot"],
            "success_max_cart_vel": eval_cfg.get("success_max_cart_vel", 0.5),
            "success_hold_steps": eval_cfg["success_hold_steps"],
            "max_steps": eval_cfg["eval_max_steps"],
        }
        results, _ = do_rollout(
            env,
            batch_fn,
            num_trajectories=eval_cfg["eval_num_trajectories"],
            max_steps=eval_cfg["eval_max_steps"],
            env_name=env_name,
            success_cfg=success_cfg,
            capture_per_step=False,
        )
        return results, None, None

    out_dir = Path("train_residual") / "weights" / cfg.experiment_name
    phase_dir = out_dir / "residual_train"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Persist the training config alongside the weights so eval can recover
    # the actor architecture and the wrapper kwargs it was trained against.
    with (phase_dir / "config.yaml").open("w") as f:
        yaml.dump(
            {"train_residual_cfg": dataclasses.asdict(cfg)},
            f,
            default_flow_style=False,
            sort_keys=False,
        )

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

    if run_post_eval:
        _run_post_train_eval(
            cfg=cfg,
            env_name=env_name,
            env_kwargs=env_kwargs,
            base_policy_snap=base_policy_snap,
            koopman_model=model,
            lqr=lqr,
            gamma_max=gamma_max,
            phase_dir=phase_dir,
            device=device,
        )

    return str(out_dir)


def _run_post_train_eval(
    *,
    cfg: TrainResidualCfg,
    env_name: str,
    env_kwargs: dict,
    base_policy_snap: dict,
    koopman_model,
    lqr,
    gamma_max: float,
    phase_dir: Path,
    device: torch.device,
) -> None:
    """Run base + LQR + residual eval through :func:`eval.multi_mode.run_multi_mode`."""
    from eval.multi_mode import run_multi_mode
    from eval.rollout import build_residual_eval_env
    from models.residual_policy import StochasticActor
    from policy import make_policy

    base_policy = make_policy(
        base_policy_snap["name"], **(base_policy_snap.get("params", {}) or {})
    )

    # Reconstruct the actor from best.pt. Need a 1-env residual wrapper just
    # to peek the obs/act space the actor was trained on.
    best_path = phase_dir / "best.pt"
    if not best_path.is_file():
        print(
            f"[train_residual] {best_path} missing — skipping post-train eval. "
            "Training likely didn't complete an eval cycle."
        )
        return

    one_env = build_residual_eval_env(
        mode="residual",
        make_single_env_fn=lambda: make_single_env(env_name=env_name, env_kwargs=env_kwargs),
        num_envs=1,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        pred_error_space=cfg.pred_error_space,
        z_ref_max_mode=cfg.z_ref_max_mode,
        gamma_worst_case=cfg.gamma_worst_case,
        obs_augmentation_override=cfg.obs_augmentation,
    )
    obs_space = one_env.single_observation_space
    act_space = one_env.single_action_space
    one_env.close()

    actor = StochasticActor(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        hidden_size=cfg.actor_hidden_size,
        hidden_layers=cfg.actor_hidden_layers,
    )
    actor.load_state_dict(torch.load(best_path, map_location=device))

    eval_out_dir = Path("eval") / "results" / cfg.experiment_name
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal EvalCfg for the post-train rollouts. The koopman / lqr /
    # residual paths are already loaded; the cfg here only feeds the rollout
    # knobs (trajectory count, max steps, parallelism, success thresholds).
    from config.manager import EvalCfg

    eval_cfg = EvalCfg(
        num_parallel_evals=cfg.num_envs,
        eval_num_trajectories=200,
        eval_max_steps=1000,
        eval_seed=cfg.seed,
        koopman_experiment_name=Path(cfg.koopman_path).name,
        lqr_output_name=Path(cfg.controller_path).name,
        residual_experiment_name=cfg.experiment_name,
        results_name=cfg.experiment_name,
        eval_koopman_accuracy=False,
        eval_policy_rollout=True,
    )

    run_multi_mode(
        out_dir=eval_out_dir,
        eval_cfg=eval_cfg,
        env_name=env_name,
        env_kwargs=env_kwargs,
        base_policy=base_policy,
        koopman_model=koopman_model,
        device=device,
        lqr=lqr,
        gamma_max=gamma_max,
        residual_actor=actor,
        residual_obs_augmentation=cfg.obs_augmentation,
        residual_pred_error_space=cfg.pred_error_space,
        residual_z_ref_max_mode=cfg.z_ref_max_mode,
        residual_gamma_worst_case=float(cfg.gamma_worst_case),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC residual policy.")
    parser.add_argument("--config", required=True, help="Per-stage train_residual YAML.")
    parser.add_argument(
        "--no_eval", action="store_true",
        help="Skip the post-training eval (base + LQR + residual rollouts).",
    )
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "train_residual_cfg")
    run(cfg, run_post_eval=not args.no_eval)


if __name__ == "__main__":
    main()
