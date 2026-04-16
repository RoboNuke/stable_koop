"""
Train Koopman A and B matrices together using base policy trajectories.

The base policy action is the control input to B (not augmented into the state).
Optionally prepends raw state to latent via prepend_state config.
After training, runs alpha bound stability analysis.

Usage:
    python -m launch.train_together --config config/caylay_a.yaml
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml

from launch.eval_policy import make_policy, make_single_env
from launch.train_pendulum import train, TrajectoryDataset, collect_data
from launch.pipeline_utils import (
    Tee, make_device, build_koopman_model, save_checkpoint, evaluate_and_save,
)
from launch.stability_utils import alpha_bound, setup_lqr
from launch.run import make_run_dir, save_config, phase_0_base_eval, make_env


def collect_policy_trajectories(env, policy, num_trajectories, max_steps, seed):
    """Collect trajectories with base policy, returning states and actions separately.

    Unlike augment_trajectories, actions are NOT appended to states — they are
    the control input for B.

    Returns:
        list of (states: (T, state_dim), actions: (T-1, action_dim))
    """
    np.random.seed(seed)
    result = []
    for i in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        actions = []
        for t in range(max_steps):
            action = policy(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            if terminated or truncated:
                break
        s = np.array(states, dtype=np.float32)  # (T+1, state_dim)
        a = np.array(actions, dtype=np.float32).reshape(-1, 1)  # (T, action_dim)
        # TrajectoryDataset expects (T+1 states, T actions) for sliding windows
        result.append((s, a))

    print(f"Collected {len(result)} trajectories "
          f"({sum(len(a) for _, a in result)} transitions)")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train Koopman A+B together with base policy as control")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    exp_name = input("Enter experiment name: ").strip()
    if not exp_name:
        raise ValueError("Experiment name cannot be empty.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # No augmentation — B handles the control
    cfg["augment_state"] = False

    run_dir = make_run_dir(exp_name)

    # Tee stdout to log file
    log_path = os.path.join(run_dir, "run.log")
    tee = Tee(log_path, sys.stdout)
    sys.stdout = tee

    print(f"Run directory: {run_dir}")
    print(f"Log file: {log_path}")
    print(f"Augment state: False (B handles control)")
    save_config(cfg, run_dir)

    # Environments
    eval_env = make_env(cfg)
    train_env = make_single_env(cfg)

    # Policy
    policy = make_policy(cfg)

    # Build model (no augmentation, so state_dim is raw obs dim)
    device = make_device()
    model, koopman_state_dim = build_koopman_model(cfg, augment=False, device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Phase 0: Base Policy Benchmark ---
    baseline_results = phase_0_base_eval(eval_env, policy, cfg, run_dir)

    # --- Phase 1: Collect data and train A+B together ---
    print("\n=== Phase 1: Train A+B Together (Base Policy as Control) ===")

    trajectories = collect_policy_trajectories(
        train_env, policy, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"],
    )

    # Normalize observations
    obs_scale = np.maximum(np.abs(train_env.observation_space.high),
                            np.abs(train_env.observation_space.low)).astype(np.float32)
    act_scale = np.maximum(np.abs(train_env.action_space.high),
                            np.abs(train_env.action_space.low)).astype(np.float32)
    cfg["obs_scale"] = obs_scale.tolist()
    cfg["act_scale"] = act_scale.tolist()
    print(f"Observation scale: {obs_scale}")
    print(f"Action scale: {act_scale}")

    # Normalize trajectories
    norm_trajectories = []
    for states, actions in trajectories:
        norm_states = states / obs_scale
        norm_actions = actions / act_scale
        norm_trajectories.append((norm_states, norm_actions))

    # Train
    model = train(model, norm_trajectories, cfg)

    # Print A matrix
    with torch.no_grad():
        A = model.A.detach().cpu()
        print(f"\nTrained A matrix ({A.shape[0]}x{A.shape[1]}):")
        print(A.numpy())
        print(f"  ||A|| (spectral norm): {torch.linalg.norm(A, ord=2).item():.6f}")
        eigvals = torch.linalg.eigvals(A)
        print(f"  Spectral radius: {eigvals.abs().max().item():.6f}")

    # Save checkpoint
    ckpt_path = os.path.join(run_dir, "koopman_ckpt.pt")
    save_checkpoint(model, cfg, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # Evaluate
    error_stats = evaluate_and_save(model, norm_trajectories, cfg, run_dir,
                                     "koopman_", "Prediction Error (A+B together)")

    # --- Phase 2: Alpha Bound Stability Analysis ---
    if cfg.get("use_alpha_bound", False):
        print("\n=== Phase 2: Alpha Bound Stability Analysis ===")
        lqr, Q, R_cost, B_scale = setup_lqr(
            model.A.detach().cpu(), model.B_matrix.detach().cpu(), cfg)

        # Collect perturbed data for stability analysis
        from launch.run import collect_perturbed_data, augment_perturbed_trajectories
        print(f"\nCollecting perturbed trajectories for stability analysis...")
        perturbed_trajs = collect_perturbed_data(
            train_env, policy, cfg["num_trajectories"],
            cfg["max_episode_steps"], cfg["seed"],
            perturb_scale=cfg.get("perturb_scale", None),
            fix_perturb_range=cfg.get("fix_perturb_range", False),
            hold_steps=cfg.get("hold_steps", 1),
        )
        # No augmentation — just normalize obs and perturbations
        aug_trajectories = augment_perturbed_trajectories(
            perturbed_trajs, augment=False, obs_scale=obs_scale, act_scale=act_scale)

        stability_dir = os.path.join(run_dir, "stability")
        alpha_vars = alpha_bound(model, lqr, cfg, aug_trajectories, train_env,
                                  error_stats=error_stats)

    eval_env.close()
    train_env.close()
    print(f"\n=== Pipeline complete. All outputs in {run_dir} ===")

    # Restore stdout
    sys.stdout = tee.stream
    tee.close()


if __name__ == "__main__":
    main()
