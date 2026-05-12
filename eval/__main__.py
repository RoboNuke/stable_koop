"""CLI: ``python -m eval --config <yaml>``.

Loads a trained Koopman model + optional residual policy and runs the
evaluations enabled by ``EvalCfg``: koopman-accuracy heatmap and/or
policy-rollout success-rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from config.manager import ConfigManager, EvalCfg
from controller.lqr.lqr import LQR
from data.dataloader import load_dataset
from data.env_builder import make_eval_env
from policy import make_policy
from train_koopman.checkpointing import load_koopman_experiment, make_device


def _flat_eval_cfg(cfg: EvalCfg) -> dict:
    return {
        "num_parallel_evals": cfg.num_parallel_evals,
        "eval_num_trajectories": cfg.eval_num_trajectories,
        "eval_max_steps": cfg.eval_max_steps,
        "eval_seed": cfg.eval_seed,
        "success_angle_deg": cfg.success_angle_deg,
        "success_max_thdot": cfg.success_max_thdot,
        "success_hold_steps": cfg.success_hold_steps,
    }


def run(cfg: EvalCfg) -> str:
    device = make_device()
    out_dir = Path("eval") / "results" / cfg.results_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, koop_cfg = load_koopman_experiment(cfg.koopman_experiment_name, device)
    ds = load_dataset(koop_cfg["dataset_name"])
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = gather_snap["env_name"]
    env_kwargs = gather_snap.get("env_kwargs", {}) or {}

    flat = _flat_eval_cfg(cfg)

    if cfg.eval_koopman_accuracy:
        from config.manager import AugmentationCfg
        from data.augmentation import augment_trajectories, compute_act_scale, compute_obs_scale
        from eval.koopman_accuracy import evaluate_model

        aug_cfg = AugmentationCfg(
            prepend_base_action=koop_cfg["prepend_base_action"],
            use_action_delta=koop_cfg["use_action_delta"],
            obs_scale_source=koop_cfg.get("obs_scale_source", "env"),
            act_scale_source=koop_cfg.get("act_scale_source", "env"),
        )
        aug_trajectories = augment_trajectories(ds, aug_cfg)
        obs_scale = compute_obs_scale(aug_cfg, ds)
        act_scale = compute_act_scale(aug_cfg, ds)
        koopman_obs_scale = (
            np.concatenate([obs_scale, act_scale])
            if aug_cfg.prepend_base_action and obs_scale is not None and act_scale is not None
            else obs_scale
        )
        fig, error_stats, heatmap_data = evaluate_model(
            model,
            aug_trajectories,
            train_horizon=koop_cfg.get("horizon", 5),
            eval_horizon=koop_cfg.get("horizon", 5) + 1,
            env_name=env_name,
            obs_scale=koopman_obs_scale.tolist(),
            obs_type=env_kwargs.get("obs_type", "cos_sin"),
        )
        if fig is not None:
            fig.savefig(out_dir / "koopman_prediction_error.png", dpi=150, bbox_inches="tight")
        with (out_dir / "koopman_eval_stats.yaml").open("w") as f:
            yaml.dump(
                {**error_stats, "heatmap": heatmap_data},
                f,
                default_flow_style=False,
                sort_keys=False,
            )
        print(f"Koopman accuracy saved to {out_dir}/koopman_eval_stats.yaml")

    if cfg.eval_policy_rollout:
        from eval.policy_rollout import evaluate

        env = make_eval_env(
            env_name=env_name,
            num_parallel_evals=cfg.num_parallel_evals,
            env_kwargs=env_kwargs,
        )

        base_policy_snap = gather_snap["base_policy"]
        policy = make_policy(
            base_policy_snap["name"], **base_policy_snap.get("params", {})
        )

        if cfg.residual_experiment_name is not None:
            print(
                "[eval] Residual policy loading is not implemented in this entry; "
                "use eval/comp_base_to_res_policy.py for combined eval."
            )

        results, all_states, all_actions = evaluate(env, policy, flat, env_name=env_name)
        with (out_dir / "base_eval_eval_stats.yaml").open("w") as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        save_dict = {}
        for i, (s, a) in enumerate(zip(all_states, all_actions)):
            save_dict[f"states_{i}"] = s
            save_dict[f"actions_{i}"] = a
        np.savez(out_dir / "base_eval_traj.npz", **save_dict)
        print(f"Policy rollout saved to {out_dir}/base_eval_eval_stats.yaml")
        env.close()

    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stable_koop artifacts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "eval_cfg")
    run(cfg)


if __name__ == "__main__":
    main()
