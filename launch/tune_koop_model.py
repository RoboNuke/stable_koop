"""
Compute stability/control variables from a pre-trained Koopman model.

Runs Phase 3 of the pipeline with a saved config and model checkpoint.

Usage:
    python -m launch.tune_koop_model --model-dir output/some_run/
"""
import argparse
import os
from datetime import datetime

import numpy as np
import torch
import yaml

from launch.run import (
    make_env,
    make_base_policy,
    compute_obs_scale,
    collect_perturbed_data,
    augment_perturbed_trajectories,
    phase_3_compute_variables,
    phase_3_lyapunov,
    save_config,
)
from model.autoencoder import KoopmanAutoencoder


def main():
    parser = argparse.ArgumentParser(
        description="Compute stability variables from a pre-trained Koopman model")
    parser.add_argument("model_dir", type=str,
                        help="Path to model output directory (contains config.yaml and koopman_ckpt.pt)")
    parser.add_argument("--config", type=str, default="config/pendulum.yaml",
                        help="Config file for tuning params (q_scale, r_scale, etc.)")
    parser.add_argument("--filename", type=str, default=None,
                        help="Custom folder name (replaces stability_{timestamp})")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides default location entirely)")
    args = parser.parse_args()

    # Load config and weights from model directory
    config_path = os.path.join(args.model_dir, "config.yaml")
    weights_path = os.path.join(args.model_dir, "koopman_ckpt.pt")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    augment = cfg.get("augment_state", True)

    # Output directory
    if args.output_dir is not None:
        run_dir = args.output_dir
    else:
        folder_name = args.filename if args.filename else f"stability_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir = os.path.join(args.model_dir, "tuned", folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Augment state with base policy action: {augment}")
    save_config(cfg, run_dir)

    # Environment and policy
    env = make_env(cfg)
    policy = make_base_policy(cfg)
    obs_scale = compute_obs_scale(env, augment)
    cfg["obs_scale"] = obs_scale.tolist()
    print(f"Observation scale: {obs_scale}")

    # Build model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(weights_path, map_location=device)
    koopman_state_dim = cfg["state_dim"] + cfg["action_dim"] if augment else cfg["state_dim"]
    model = KoopmanAutoencoder(
        state_dim=koopman_state_dim,
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        encoder_type=cfg.get("encoder_type", "linear"),
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    print(f"Loaded weights from {weights_path}")
    print(f"Koopman model: state_dim={koopman_state_dim}, action_dim={cfg['action_dim']}, "
          f"latent_dim={cfg['latent_dim']}")

    # Collect perturbed data for Lipschitz computation
    print("\nCollecting perturbed trajectories for Lipschitz bound...")
    trajectories = collect_perturbed_data(
        env, policy, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"],
        perturb_scale=cfg.get("perturb_scale", None),
        fix_perturb_range=cfg.get("fix_perturb_range", False),
    )
    aug_trajectories = augment_perturbed_trajectories(
        trajectories, augment=augment, obs_scale=obs_scale)

    # Override tuning parameters from the --config file
    with open(args.config) as f:
        tune_cfg = yaml.safe_load(f)
    for key in ("use_eigen_bound", "use_lyapunov_bound",
                "q_scale", "r_scale", "max_tracking_error_x", "max_displacement_x"):
        if key in tune_cfg:
            cfg[key] = tune_cfg[key]

    # Phase 3: compute stability variables
    variables = {}
    if cfg.get("use_eigen_bound", False):
        eigen_vars, _ = phase_3_compute_variables(model, cfg, run_dir, aug_trajectories,
                                                   tuning_config=args.config)
        variables.update(eigen_vars)
    if cfg.get("use_lyapunov_bound", False):
        lyap_vars, _ = phase_3_lyapunov(model, cfg, run_dir, aug_trajectories,
                                         tuning_config=args.config)
        variables.update(lyap_vars)

    env.close()
    print(f"\n=== Done. All outputs in {run_dir} ===")


if __name__ == "__main__":
    main()
