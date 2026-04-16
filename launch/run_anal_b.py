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
    phase_0_base_eval, phase_3_lyapunov, lipschitz_m_free,
)
from launch.pipeline_utils import Tee, make_device, build_koopman_model, save_checkpoint
from launch.stability_utils import alpha_bound, setup_lqr
from launch.analy_b_tuning import run_analytical_b


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

    # 4. Print and save A matrix
    with torch.no_grad():
        A = model.A.detach().cpu()
        print(f"\nTrained A matrix ({A.shape[0]}x{A.shape[1]}):")
        print(A.numpy())
        print(f"  ||A|| (spectral norm): {torch.linalg.norm(A, ord=2).item():.6f}")
        eigvals = torch.linalg.eigvals(A)
        print(f"  Spectral radius: {eigvals.abs().max().item():.6f}")

    # 5. Save checkpoint
    ckpt_path = os.path.join(run_dir, "koop_a_checkpoint.pt")
    save_checkpoint(model, cfg, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 5. Evaluate on training data and save
    from launch.pipeline_utils import evaluate_and_save
    evaluate_and_save(model, aug_trajectories, cfg, run_dir, "koop_a_",
                      "Prediction Error (A only, no B, no perturbations)")

    print("Phase 1 complete.")
    return aug_trajectories, trajectories


def phase_2_analytical_B(model, env, policy, cfg, run_dir, augment=True,
                         obs_scale=None, act_scale=None,
                         base_aug_trajectories=None,
                         base_raw_trajectories=None):
    """Phase 2: Analytically derive B matrix from perturbed trajectory data.

    Collects trajectories with base policy + random perturbations, then solves
    for B via least-squares with controllability projection.
    """
    print("\n=== Phase 2: Analytical B Matrix Derivation ===")

    B_final = run_analytical_b(
        model, env, policy, cfg, run_dir,
        augment=augment, obs_scale=obs_scale, act_scale=act_scale,
        base_aug_trajectories=base_aug_trajectories,
        base_raw_trajectories=base_raw_trajectories,
    )

    print("Phase 2 complete.")
    return B_final


def phase_3_stability(model, env, policy, cfg, run_dir, B_final,
                      augment=True, obs_scale=None):
    """Phase 3: Stability analysis with analytical B on perturbed data.

    Sets the model's B matrix to the analytical solution, collects perturbed
    trajectories, and runs configured stability checks (Lyapunov and/or m-free).
    """
    print("\n=== Phase 3: Stability Analysis ===")

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

    # 4. Run stability analyses
    stability_dir = os.path.join(run_dir, "stability")
    variables = {}
    lqr = None

    if cfg.get("use_lyapunov_bound", False):
        lyap_vars, lyap_lqr = phase_3_lyapunov(model, cfg, stability_dir, aug_trajectories)
        variables.update(lyap_vars)
        if lqr is None:
            lqr = lyap_lqr

    if cfg.get("use_m_free_bound", False):
        mfree_vars, mfree_lqr = lipschitz_m_free(model, cfg, stability_dir,
                                                   aug_trajectories, env)
        variables.update(mfree_vars)
        if lqr is None:
            lqr = mfree_lqr

    if cfg.get("use_alpha_bound", False):
        if lqr is None:
            lqr_ab, _, _, _ = setup_lqr(model.A.detach().cpu(), model.B_matrix.detach().cpu(), cfg)
            lqr = lqr_ab
        alpha_vars = alpha_bound(model, lqr, cfg, aug_trajectories, env)
        variables.update(alpha_vars)

    # 5. Save checkpoint with analytical B included
    ckpt_path = os.path.join(run_dir, "koopman_ckpt.pt")
    save_checkpoint(model, cfg, ckpt_path)
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
    device = make_device()
    model, koopman_state_dim = build_koopman_model(cfg, augment, device)

    # --- Execute phases ---
    baseline_results = phase_0_base_eval(eval_env, policy, cfg, run_dir)
    base_aug_trajectories, base_raw_trajectories = phase_1_train_A_only(
        model, train_env, policy, cfg, run_dir, augment, obs_scale, act_scale)
    anal_b_policy = make_analytical_b_policy(cfg)
    B_final = phase_2_analytical_B(
        model, train_env, anal_b_policy, cfg, run_dir, augment, obs_scale,
        act_scale=act_scale,
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
