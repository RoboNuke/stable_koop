"""
Runner script -- single entry point for the Koopman pipeline.

Usage:
    python -m launch.run --config config/pendulum.yaml
"""
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import yaml

from launch.eval_policy import evaluate as evaluate_policy, make_policy
from launch.eval_pendulum import evaluate_model
from launch.train_pendulum import collect_data, train
from model.autoencoder import KoopmanAutoencoder


def make_run_dir(cfg):
    """Create and return output/{run_name}_{datetime}/."""
    run_name = cfg.get("run_name", "run")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("output", f"{run_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(cfg, run_dir):
    """Dump the full resolved config into the run directory."""
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {path}")


def make_env(cfg):
    """Create and seed the environment."""
    env = gym.make(cfg["env_name"])
    env.reset(seed=cfg["eval_seed"])
    np.random.seed(cfg["eval_seed"])
    return env


def compute_obs_scale(env, augment):
    """Compute observation scaling factors from env spaces.

    Returns scale vector where each element is the max absolute value for that
    dimension. When augment=True, action space bounds are appended.

    Args:
        env: gym environment
        augment: whether base policy action is part of the Koopman state

    Returns:
        obs_scale: (koopman_state_dim,) numpy array
    """
    obs_scale = np.maximum(np.abs(env.observation_space.high),
                           np.abs(env.observation_space.low))
    if augment:
        act_scale = np.maximum(np.abs(env.action_space.high),
                               np.abs(env.action_space.low))
        obs_scale = np.concatenate([obs_scale, act_scale])
    return obs_scale.astype(np.float32)


def compute_act_scale(env):
    """Compute action scaling factors from env action space.

    Returns scale vector where each element is the max absolute value for that
    action dimension.

    Args:
        env: gym environment

    Returns:
        act_scale: (action_dim,) numpy array
    """
    act_scale = np.maximum(np.abs(env.action_space.high),
                           np.abs(env.action_space.low))
    return act_scale.astype(np.float32)


def make_base_policy(cfg):
    """Build the base PD policy from config."""
    return make_policy(cfg)


def save_eval_results(results, all_states, all_actions, phase_dir):
    """Save evaluation stats and trajectories to a phase directory."""
    os.makedirs(phase_dir, exist_ok=True)

    stats_path = os.path.join(phase_dir, "eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Stats saved to {stats_path}")

    traj_path = os.path.join(phase_dir, "traj.npz")
    save_dict = {}
    for i, (s, a) in enumerate(zip(all_states, all_actions)):
        save_dict[f"states_{i}"] = s
        save_dict[f"actions_{i}"] = a
    np.savez(traj_path, **save_dict)
    print(f"Trajectories saved to {traj_path}")


def augment_trajectories(trajectories, augment=True, obs_scale=None,
                         act_scale=None):
    """Prepare trajectories for autonomous Koopman training.

    If augment=True, concatenates base policy actions into states so the base
    policy is treated as part of the environment.
    If augment=False, uses original states only (base policy actions are ignored).
    In both cases, zero actions are passed to the model so B has no influence.

    Args:
        trajectories: list of (states: (T+1, S), actions: (T, A))
        augment: whether to append base policy actions to state
        obs_scale: if provided, divide koopman states by this scale vector
        act_scale: if provided, divide control actions by this scale vector

    Returns:
        list of (koopman_states, zero_actions)
    """
    result = []
    for states, actions in trajectories:
        if augment:
            koopman_states = np.concatenate([states[:-1], actions], axis=-1)
        else:
            koopman_states = states[:-1]
        if obs_scale is not None:
            koopman_states = koopman_states / obs_scale
        zero_actions = np.zeros((len(koopman_states) - 1, actions.shape[-1]),
                                dtype=np.float32)
        result.append((koopman_states, zero_actions))
    return result


def collect_perturbed_data(env, policy, num_trajectories, max_steps, seed,
                          perturb_scale=None, fix_perturb_range=False,
                          hold_steps=1):
    """Collect trajectories with base policy + uniform random perturbation.

    The total action applied to the env is base_action + perturbation, clipped
    to the action space. Returns states, base_actions, and perturbation_actions
    separately so they can be used differently in the Koopman model.

    Args:
        perturb_scale: max magnitude of perturbation. If None, uses action space bounds.
        fix_perturb_range: if True, sample perturbations only in the range that
            doesn't saturate the controller given the current base action.
        hold_steps: number of steps to hold each sampled perturbation before
            resampling a new one. Default 1 (resample every step).

    Returns:
        list of (states: (T+1, S), base_actions: (T, A), perturbations: (T, A))
    """
    np.random.seed(seed)
    if perturb_scale is not None:
        perturb_low_default = -np.ones_like(env.action_space.low) * perturb_scale
        perturb_high_default = np.ones_like(env.action_space.high) * perturb_scale
    else:
        perturb_low_default = env.action_space.low
        perturb_high_default = env.action_space.high
    action_low = env.action_space.low
    action_high = env.action_space.high

    trajectories = []
    for i in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        base_actions = []
        perturbations = []
        perturbation = None
        for t in range(max_steps):
            base_action = policy(obs) if policy is not None else np.zeros_like(action_low)
            # Resample perturbation every hold_steps
            if t % hold_steps == 0:
                if fix_perturb_range:
                    p_low = np.clip(action_low - base_action, -perturb_scale, 0)
                    p_high = np.clip(action_high - base_action, 0, perturb_scale)
                    perturbation = np.random.uniform(p_low, p_high).astype(np.float32)
                else:
                    perturbation = np.random.uniform(perturb_low_default, perturb_high_default).astype(np.float32)
            total_action = np.clip(base_action + perturbation, action_low, action_high)
            obs, _, terminated, truncated, _ = env.step(total_action)
            done = terminated or truncated
            states.append(obs)
            base_actions.append(base_action)
            perturbations.append(perturbation)
            if done:
                break
        trajectories.append((
            np.array(states, dtype=np.float32),
            np.array(base_actions, dtype=np.float32).reshape(-1, 1),
            np.array(perturbations, dtype=np.float32).reshape(-1, 1),
        ))

    print(f"Collected {len(trajectories)} perturbed trajectories "
          f"({sum(len(p) for _, _, p in trajectories)} transitions)")
    return trajectories


def augment_perturbed_trajectories(trajectories, augment=True, obs_scale=None,
                                    act_scale=None):
    """Prepare perturbed trajectories for Koopman B training.

    If augment=True, base policy actions are appended to states.
    If augment=False, original states only. In both cases, perturbations
    are the control input for the B matrix.

    Args:
        trajectories: list of (states: (T+1, S), base_actions: (T, A), perturbations: (T, A))
        augment: whether to append base policy actions to state
        obs_scale: if provided, divide koopman states by this scale vector
        act_scale: if provided, divide perturbations by this scale vector

    Returns:
        list of (koopman_states, perturbations)
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


def phase_0_base_eval(env, policy, cfg, run_dir):
    """Phase 0: evaluate the base PD policy and save results."""
    print("\n=== Phase 0: Base Policy Benchmark ===")
    phase_dir = os.path.join(run_dir, "base_eval")

    results, all_states, all_actions = evaluate_policy(env, policy, cfg)
    save_eval_results(results, all_states, all_actions, phase_dir)

    print(f"Phase 0 complete. Results in {phase_dir}")


def phase_1_train_koopman(model, env, policy, cfg, run_dir, augment=True, obs_scale=None,
                          act_scale=None):
    """Phase 1: train Koopman model with base policy as part of environment."""
    print("\n=== Phase 1: Train Koopman Model ===")
    phase_dir = os.path.join(run_dir, "koopman_train")
    os.makedirs(phase_dir, exist_ok=True)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 1. Collect data with base policy
    trajectories = collect_data(
        env, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"], policy=policy,
    )

    # 2. Prepare for Koopman training
    aug_trajectories = augment_trajectories(trajectories, augment=augment,
                                            obs_scale=obs_scale,
                                            act_scale=act_scale)

    # 3. Train
    model = train(model, aug_trajectories, cfg)

    # 4. Save checkpoint
    ckpt_path = os.path.join(phase_dir, "checkpoint.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 5. Evaluate on training data
    fig, max_pred_error_latent, max_pred_error_state, heatmap_data = evaluate_model(
        model, aug_trajectories, cfg["horizon"], eval_horizon=25)

    # 6. Save heatmap
    plot_path = os.path.join(phase_dir, "prediction_error.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")

    # 7. Save eval stats
    eval_stats = {
        "max_prediction_error_latent": float(max_pred_error_latent),
        "max_prediction_error_state": float(max_pred_error_state),
        "heatmap": heatmap_data,
    }
    stats_path = os.path.join(phase_dir, "eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(eval_stats, f, default_flow_style=False, sort_keys=False)
    print(f"Eval stats saved to {stats_path}")

    print(f"Phase 1 complete. Results in {phase_dir}")


def phase_2_train_B(model, env, policy, cfg, run_dir, augment=True, obs_scale=None,
                    act_scale=None):
    """Phase 2: train A and B with random perturbations on top of base policy."""
    print("\n=== Phase 2: Train B Matrix (Perturbed Policy) ===")
    phase_dir = os.path.join(run_dir, "koopman_train_B")
    os.makedirs(phase_dir, exist_ok=True)

    # 1. Reinitialize B
    import torch.nn as nn
    if cfg.get("init_b_in_a", False):
        model.initialize_B_in_eigenbasis()
        print("Initialized B matrix in A eigenbasis")
    else:
        nn.init.kaiming_uniform_(model.B.weight)
        print("Reinitialized B matrix with random weights")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 2. Collect data with base policy + random perturbations
    trajectories = collect_perturbed_data(
        env, policy, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"],
        perturb_scale=cfg.get("perturb_scale", None),
        fix_perturb_range=cfg.get("fix_perturb_range", False),
        hold_steps=cfg.get("hold_steps", 1),
    )

    # 3. Prepare for Koopman training
    aug_trajectories = augment_perturbed_trajectories(trajectories, augment=augment,
                                                      obs_scale=obs_scale,
                                                      act_scale=act_scale)

    # 4. Train (both A and B update)
    model = train(model, aug_trajectories, cfg)

    # 5. Save checkpoint to top-level run directory
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 6. Evaluate on training data
    fig, max_pred_error_latent, max_pred_error_state, heatmap_data = evaluate_model(
        model, aug_trajectories, cfg["horizon"], eval_horizon=25)

    # 7. Save heatmap
    plot_path = os.path.join(phase_dir, "prediction_error.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")

    # 8. Save eval stats
    eval_stats = {
        "max_prediction_error_latent": float(max_pred_error_latent),
        "max_prediction_error_state": float(max_pred_error_state),
        "heatmap": heatmap_data,
    }
    stats_path = os.path.join(phase_dir, "eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(eval_stats, f, default_flow_style=False, sort_keys=False)
    print(f"Eval stats saved to {stats_path}")

    print(f"Phase 2 complete. Results in {phase_dir}")
    return aug_trajectories


def phase_3_compute_variables(model, cfg, phase_dir, aug_trajectories,
                              tuning_config=None):
    """Phase 3: compute stability and control variables from trained model.

    Args:
        tuning_config: path to a YAML file to override q_scale, r_scale,
            max_tracking_error_x, and max_displacement_x from the saved config.
    """
    print("\n=== Phase 3: Compute Stability Variables ===")
    os.makedirs(phase_dir, exist_ok=True)

    from controllers.lqr import LQR
    from model.utils import (spectral_radius, transient_constant,
                             compute_lower_lipschitz,
                             state_error_to_latent_error,
                             max_tolerable_model_error)

    # Override tuning parameters from external config if provided
    if tuning_config is not None:
        with open(tuning_config) as f:
            tune_cfg = yaml.safe_load(f)
        for key in ("q_scale", "r_scale", "max_tracking_error_x", "max_displacement_x"):
            if key in tune_cfg:
                cfg[key] = tune_cfg[key]

    device = next(model.parameters()).device
    model.eval()

    # Extract A and B from trained model
    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()

    # Spectral radii of A and B
    rho_A = spectral_radius(A)
    B_sigma_max = torch.norm(B_mat, p=2).item()
    print(f"  Spectral radius of A (open):           {rho_A:.6f}")
    print(f"  B largest singular value:              {B_sigma_max:.6f}")

    A_np = A.numpy()
    B_np = B_mat.numpy()
    C_mat = np.hstack([np.linalg.matrix_power(A_np, i) @ B_np
                       for i in range(A_np.shape[0])])
    ctrl_rank = np.linalg.matrix_rank(C_mat)
    print(f"  Controllability rank:                  {ctrl_rank} / {A_np.shape[0]}")

    if cfg.get("k_type") in ["unbounded","normalized"]:
        eigenvalues, V = torch.linalg.eig(A)
        unstable_mask = eigenvalues.abs() > 1.0
        for i, (ev, unstable) in enumerate(zip(eigenvalues, unstable_mask)):
            if unstable:
                proj = (V[:, i].conj() @ B_mat.to(torch.cfloat)).abs()
                print(f"      Unstable mode λ={ev.abs():.4f}, B projection={proj.item():.4f}")

    # 1. Encoder lower Lipschitz bound (m)
    model_cpu = model.cpu()
    training_states = []
    for states, actions in aug_trajectories:
        for s in states:
            training_states.append(s)
    m = compute_lower_lipschitz(model_cpu.encode, training_states)
    model.to(device)
    print(f"  Encoder lower Lipschitz (m):           {m:.6f}")

    # 2. LQR: F and gain norm
    latent_dim = cfg["latent_dim"]
    action_dim = cfg["action_dim"]
    Q = torch.eye(latent_dim) * cfg.get("q_scale", 1.0)
    R = torch.eye(action_dim) * cfg["r_scale"]
    lqr = LQR(A, B_mat, Q, R)
    F = lqr.F
    gain_norm = lqr.gain_norm.item()
    print(f"  Q scale:                               {cfg.get('q_scale', 1.0)}")
    print(f"  R scale:                               {cfg['r_scale']}")
    BtPB = (B_mat.T @ lqr.P @ B_mat).item()
    print(f"  B^T P B:                               {BtPB:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")

    # 3. Closed-loop spectral radius and transient constant
    closed_loop = lqr.closed_loop
    rho = spectral_radius(closed_loop)
    C = transient_constant(closed_loop)
    print(f"  Spectral radius A-BF (closed):         {rho:.6f}")
    print(f"  Transient constant (C):                {C:.6f}")

    # 4. Derived quantities
    max_tracking_error_x = cfg["max_tracking_error_x"]
    max_displacement_x = cfg["max_displacement_x"]

    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)
    max_runtime_error_latent = max_tolerable_model_error(rho, C, max_tracking_error_latent, eta)
    residual_ctrl_budget = max_tracking_error_latent * gain_norm

    print(f"  max_tracking_error_x:                  {max_tracking_error_x:.6f}")
    print(f"  max_displacement_x:                    {max_displacement_x:.6f}")
    print(f"  max_tracking_error_latent:             {max_tracking_error_latent:.6f}")
    print(f"  eta (m * max_displacement_x):          {eta:.6f}")
    if max_runtime_error_latent < 0:
        print(f"\033[91m  max_runtime_error_latent:              {max_runtime_error_latent:.6f}\033[0m")
    else:
        print(f"\033[92m  max_runtime_error_latent:              {max_runtime_error_latent:.6f}\033[0m")
    print(f"  residual_ctrl_budget:                  {residual_ctrl_budget:.6f}")

    # Heatmap: max_runtime_error_latent over (max_tracking_error_x, max_displacement_x)
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    n_heatmap = 200
    track_sweep = np.linspace(0.0, 1.0, n_heatmap)
    disp_sweep = np.linspace(0.0, 1.0, n_heatmap)
    Z = np.zeros((n_heatmap, n_heatmap))
    for i, d in enumerate(disp_sweep):
        for j, t in enumerate(track_sweep):
            eps_lat = state_error_to_latent_error(t, m)
            eta_lat = state_error_to_latent_error(d, m)
            Z[i, j] = max_tolerable_model_error(rho, C, eps_lat, eta_lat)

    norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=max(Z.max(), 1e-6))
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("RdWtGn", ["red", "white", "green"])
    im = ax3.imshow(Z, origin='lower', aspect='auto',
                    extent=[0, 1, 0, 1], norm=norm, cmap=cmap)
    cbar = fig3.colorbar(im, ax=ax3, label="max_runtime_error_latent")
    cbar.set_ticks([Z.min(), 0, Z.max()])
    cbar.set_ticklabels([f"{Z.min():.4f}", "0", f"{Z.max():.4f}"])
    ax3.plot(max_tracking_error_x, max_displacement_x, 'bx', markersize=10, markeredgewidth=2)
    ax3.set_xlabel("max_tracking_error_x")
    ax3.set_ylabel("max_displacement_x")
    ax3.set_title("Runtime Error Budget Heatmap")
    ax3.text(0.5, -0.12,
             f"rho={rho:.4f}  C={C:.4f}  m={m:.4f}  Q_scale={cfg.get('q_scale', 1.0)}  R_scale={cfg['r_scale']}",
             transform=ax3.transAxes, ha='center', fontsize=9)
    fig3.tight_layout()
    folder_name = os.path.basename(os.path.normpath(phase_dir))
    heatmap_name = f"{folder_name}_heatmap.png"
    fig3.savefig(os.path.join(phase_dir, heatmap_name), dpi=150)
    plt.close(fig3)
    print(f"Saved {heatmap_name}")

    # Save
    variables = {
        "m": float(m),
        "gain_norm": float(gain_norm),
        "rho": float(rho),
        "C": float(C),
        "max_tracking_error_x": float(max_tracking_error_x),
        "max_displacement_x": float(max_displacement_x),
        "max_tracking_error_latent": float(max_tracking_error_latent),
        "eta": float(eta),
        "max_runtime_error_latent": float(max_runtime_error_latent),
        "residual_ctrl_budget": float(residual_ctrl_budget),
        "A": A.numpy().tolist(),
        "B": B_mat.numpy().tolist(),
    }
    stats_path = os.path.join(phase_dir, "variables.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"Variables saved to {stats_path}")

    print(f"Phase 3 complete. Results in {phase_dir}")
    return variables


def main():
    parser = argparse.ArgumentParser(description="Koopman pipeline runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Do not append base policy action to the Koopman state")
    parser.add_argument("--skip-pretrain", action="store_true", default=False,
                        help="Skip Phase 1 (A-only training), go straight to joint A+B training")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    augment = not args.no_augment
    cfg["augment_state"] = augment

    run_dir = make_run_dir(cfg)
    print(f"Run directory: {run_dir}")
    print(f"Augment state with base policy action: {augment}")
    save_config(cfg, run_dir)

    env = make_env(cfg)
    if cfg.get("no_base_policy", False):
        policy = None
        print("No base policy — using random actions only")
    else:
        policy = make_base_policy(cfg)

    # Compute observation and action scaling
    obs_scale = compute_obs_scale(env, augment)
    act_scale = compute_act_scale(env)
    cfg["obs_scale"] = obs_scale.tolist()
    cfg["act_scale"] = act_scale.tolist()
    print(f"Observation scale: {obs_scale}")
    print(f"Action scale: {act_scale}")

    # Build Koopman model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    print(f"Koopman model: state_dim={koopman_state_dim}, action_dim={cfg['action_dim']}, "
          f"latent_dim={cfg['latent_dim']}")

    # --- Execute phases ---
    phase_0_base_eval(env, policy, cfg, run_dir)
    if not args.skip_pretrain:
        phase_1_train_koopman(model, env, policy, cfg, run_dir, augment, obs_scale,
                              act_scale)
    aug_trajectories = phase_2_train_B(model, env, policy, cfg, run_dir, augment, obs_scale,
                                       act_scale)
    stability_dir = os.path.join(run_dir, "stability")
    variables = phase_3_compute_variables(model, cfg, stability_dir, aug_trajectories)

    env.close()
    print(f"\n=== Pipeline complete. All outputs in {run_dir} ===")


if __name__ == "__main__":
    main()
