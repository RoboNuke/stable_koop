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
from train_koopman.checkpointing import build_koopman_model, make_device


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


def _load_koopman(experiment_name: str, device):
    ckpt_path = Path("train_koopman") / "weights" / experiment_name / "koopman_ckpt.pt"
    raw = torch.load(ckpt_path, map_location=device)
    koop_cfg = raw["config"]
    model, _ = build_koopman_model(koop_cfg, augment=True, device=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, koop_cfg


def run(cfg: EvalCfg) -> str:
    device = make_device()
    out_dir = Path("eval") / "results" / cfg.results_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, koop_cfg = _load_koopman(cfg.koopman_experiment_name, device)
    ds = load_dataset(koop_cfg["dataset_name"])
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]

    flat = _flat_eval_cfg(cfg)

    if cfg.eval_koopman_accuracy:
        from data.gather_data import augment_perturbed_trajectories
        from eval.koopman_accuracy import evaluate_model

        koopman_obs_scale = np.concatenate([ds.obs_scale, ds.act_scale])
        aug_trajectories = augment_perturbed_trajectories(
            ds.perturbed_trajectories,
            augment=True,
            obs_scale=koopman_obs_scale,
            act_scale=ds.act_scale,
        )
        fig, error_stats, heatmap_data = evaluate_model(
            model,
            aug_trajectories,
            train_horizon=koop_cfg.get("horizon", 5),
            eval_horizon=koop_cfg.get("horizon", 5) + 1,
            obs_scale=koopman_obs_scale.tolist(),
            obs_type=koop_cfg.get("obs_type", "cos_sin"),
        )
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
            env_name=gather_snap["env_name"],
            num_parallel_evals=cfg.num_parallel_evals,
            obs_type=gather_snap["obs_type"],
            limited_spawn=gather_snap["limited_spawn"],
            spawn_angle_range=gather_snap["spawn_angle_range"],
        )

        base_policy_params = {
            k: v for k, v in gather_snap["base_policy"].items() if k != "name"
        }
        policy = make_policy(gather_snap["base_policy"]["name"], **base_policy_params)

        # Residual policy: not loaded by this minimal CLI; document hook.
        if cfg.residual_experiment_name is not None:
            print(
                "[eval] Residual policy loading is not implemented in this entry; "
                "use eval/comp_base_to_res_policy.py for combined eval."
            )

        results, all_states, all_actions = evaluate(env, policy, flat)
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
