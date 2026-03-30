"""
Runner script -- train Koopman A matrix only (no B) using base policy data.

Collects trajectories with the base policy, augments states with the base
action, and trains the Koopman model with u=0 so only A (and encoder/decoder)
are learned. The base policy is absorbed into the autonomous dynamics.

Usage:
    python -m launch.run_anal_b --config config/pendulum.yaml
"""
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml

from launch.eval_policy import evaluate as evaluate_policy, make_policy, make_eval_env, make_analytical_b_policy, make_single_env
from launch.eval_pendulum import evaluate_model
from launch.train_pendulum import collect_data, train
from launch.run import (
    make_run_dir, save_config, make_env, compute_obs_scale, compute_act_scale,
    make_base_policy, save_eval_results, augment_trajectories,
    collect_perturbed_data, augment_perturbed_trajectories,
    phase_0_base_eval, phase_3_lyapunov,
)
from launch.analy_b_tuning import run_analytical_b
from model.autoencoder import KoopmanAutoencoder


def phase_1_train_A_only(model, env, policy, cfg, run_dir, augment=True,
                         obs_scale=None, act_scale=None):
    """Phase 1: Train Koopman A matrix (+ encoder/decoder) with u=0.

    Collects trajectories using the base policy on the real environment,
    then augments states with base actions and sets u=0 for training.
    This way the base policy is part of the autonomous dynamics and
    only A (not B) is meaningfully trained.
    """
    print("\n=== Phase 1: Train Koopman A (Base Policy, u=0) ===")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 1. Collect data with base policy
    trajectories = collect_data(
        env, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"], policy=policy,
    )

    # 2. Augment states with base action, set u=0
    aug_trajectories = augment_trajectories(
        trajectories, augment=augment,
        obs_scale=obs_scale, act_scale=act_scale,
    )

    # 3. Train with all configured losses
    #    (recon_pretrain_epochs in config handles recon-only pretraining inside train())
    model = train(model, aug_trajectories, cfg)

    # 4. Save checkpoint
    ckpt_path = os.path.join(run_dir, "koop_a_checkpoint.pt")
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 5. Evaluate on training data
    train_horizon = cfg["horizon"]
    fig, error_stats, heatmap_data = evaluate_model(
        model, aug_trajectories, train_horizon, eval_horizon=train_horizon,
        title="Prediction Error (A only, no B, no perturbations)",
        obs_scale=cfg.get("obs_scale"), obs_type=cfg.get("obs_type", "cos_sin"))

    # 6. Save heatmap
    plot_path = os.path.join(run_dir, "koop_a_prediction_error.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")

    # 7. Save eval stats
    eval_stats = {
        **error_stats,
        "heatmap": heatmap_data,
    }
    stats_path = os.path.join(run_dir, "koop_a_eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(eval_stats, f, default_flow_style=False, sort_keys=False)
    print(f"Eval stats saved to {stats_path}")

    print("Phase 1 complete.")
    return aug_trajectories, trajectories


def phase_2_analytical_B(model, env, policy, cfg, run_dir, augment=True,
                         obs_scale=None, base_aug_trajectories=None,
                         base_raw_trajectories=None):
    """Phase 2: Analytically derive B matrix from perturbed trajectory data.

    Collects trajectories with base policy + random perturbations, then solves
    for B via least-squares with controllability projection.
    """
    print("\n=== Phase 2: Analytical B Matrix Derivation ===")

    B_final = run_analytical_b(
        model, env, policy, cfg, run_dir,
        augment=augment, obs_scale=obs_scale,
        base_aug_trajectories=base_aug_trajectories,
        base_raw_trajectories=base_raw_trajectories,
    )

    print("Phase 2 complete.")
    return B_final


def phase_3_stability(model, env, policy, cfg, run_dir, B_final,
                      augment=True, obs_scale=None):
    """Phase 3: Lyapunov stability analysis with analytical B on perturbed data.

    Sets the model's B matrix to the analytical solution, collects perturbed
    trajectories, and runs the Lyapunov stability check.
    """
    print("\n=== Phase 3: Lyapunov Stability Analysis ===")

    # 1. Set model B to the analytical solution
    with torch.no_grad():
        model.B.weight.copy_(torch.tensor(B_final, dtype=model.B.weight.dtype,
                                          device=model.B.weight.device))
    print(f"  Set model B to analytical solution (norm={np.linalg.norm(B_final, ord=2):.4f})")

    # 2. Collect perturbed data (same as what B was derived from)
    print(f"\nCollecting perturbed trajectories for stability analysis...")
    trajectories = collect_perturbed_data(
        env, policy, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"],
        perturb_scale=cfg.get("perturb_scale", None),
        fix_perturb_range=cfg.get("fix_perturb_range", False),
        hold_steps=cfg.get("hold_steps", 1),
    )

    # 3. Augment trajectories
    aug_trajectories = augment_perturbed_trajectories(
        trajectories, augment=augment, obs_scale=obs_scale)

    # 4. Run Lyapunov stability analysis
    stability_dir = os.path.join(run_dir, "stability")
    variables, lqr = phase_3_lyapunov(model, cfg, stability_dir, aug_trajectories)

    # 5. Save checkpoint with analytical B included
    ckpt_path = os.path.join(run_dir, "koopman_ckpt.pt")
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Checkpoint (with analytical B) saved to {ckpt_path}")

    print("Phase 3 complete.")
    return variables, lqr


def main():
    parser = argparse.ArgumentParser(
        description="Train Koopman A matrix only (base policy absorbed, u=0)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Do not append base policy action to the Koopman state")
    args = parser.parse_args()

    exp_name = input("Enter experiment name: ").strip()
    if not exp_name:
        raise ValueError("Experiment name cannot be empty.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    augment = not args.no_augment
    cfg["augment_state"] = augment

    run_dir = make_run_dir(exp_name)

    # Tee stdout to log file
    import sys
    class Tee:
        def __init__(self, filepath, stream):
            self.file = open(filepath, "w")
            self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.file.write(data)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()
        def close(self):
            self.file.close()
    log_path = os.path.join(run_dir, "run.log")
    tee = Tee(log_path, sys.stdout)
    sys.stdout = tee

    print(f"Run directory: {run_dir}")
    print(f"Log file: {log_path}")
    print(f"Augment state with base policy action: {augment}")
    save_config(cfg, run_dir)

    # Vectorized env for evaluation, single env for data collection / scaling
    eval_env = make_env(cfg)
    train_env = make_single_env(cfg)

    RED = "\033[91m"
    RESET = "\033[0m"

    policy = make_base_policy(cfg)
    if cfg.get("base_policy", "none") == "none":
        print(f"{RED}WARNING: base_policy='none' but this script is designed to absorb "
              f"the base policy into autonomous dynamics.{RESET}")
        print(f"{RED}FALLBACK: Using zero_policy (u=0 everywhere). The Koopman A matrix "
              f"will learn uncontrolled dynamics only.{RESET}")

    # Compute observation and action scaling (single env has 1-D bounds)
    obs_scale = compute_obs_scale(train_env, augment)
    act_scale = compute_act_scale(train_env)
    cfg["obs_scale"] = obs_scale.tolist()
    cfg["act_scale"] = act_scale.tolist()
    print(f"Observation scale: {obs_scale}")
    print(f"Action scale: {act_scale}")

    # Build Koopman model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"{RED}WARNING: CUDA not available.{RESET}")
        print(f"{RED}FALLBACK: Using CPU. Training will be significantly slower.{RESET}")
    print(f"Using device: {device}")
    koopman_state_dim = cfg["state_dim"] + cfg["action_dim"] if augment else cfg["state_dim"]
    encoder_type = cfg.get("encoder_type", None)
    if encoder_type is None:
        encoder_type = "linear"
        print(f"{RED}WARNING: encoder_type not specified in config.{RESET}")
        print(f"{RED}FALLBACK: Using encoder_type='linear'.{RESET}")
    model = KoopmanAutoencoder(
        state_dim=koopman_state_dim,
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        encoder_type=encoder_type,
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
    ).to(device)
    print(f"Koopman model: state_dim={koopman_state_dim}, action_dim={cfg['action_dim']}, "
          f"latent_dim={cfg['latent_dim']}")

    # --- Execute phases ---
    baseline_results = phase_0_base_eval(eval_env, policy, cfg, run_dir)
    base_aug_trajectories, base_raw_trajectories = phase_1_train_A_only(
        model, train_env, policy, cfg, run_dir, augment, obs_scale, act_scale)
    anal_b_policy = make_analytical_b_policy(cfg)
    B_final = phase_2_analytical_B(
        model, train_env, anal_b_policy, cfg, run_dir, augment, obs_scale,
        base_aug_trajectories=base_aug_trajectories,
        base_raw_trajectories=base_raw_trajectories)
    variables, lqr = phase_3_stability(
        model, train_env, anal_b_policy, cfg, run_dir, B_final,
        augment, obs_scale)

    eval_env.close()
    train_env.close()
    print(f"\n=== Pipeline complete. All outputs in {run_dir} ===")

    # Restore stdout and close log
    sys.stdout = tee.stream
    tee.close()


if __name__ == "__main__":
    main()
