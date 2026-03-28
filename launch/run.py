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

from launch.eval_policy import evaluate as evaluate_policy, make_policy, make_eval_env
from launch.eval_pendulum import evaluate_model
from launch.train_pendulum import collect_data, train
from model.autoencoder import KoopmanAutoencoder


def make_run_dir(exp_name):
    """Create and return output/{exp_name}_{datetime}/."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("output", f"{exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(cfg, run_dir):
    """Dump the full resolved config into the run directory."""
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to {path}")


def make_env(cfg):
    """Create and seed the environment (vectorized if num_parallel_evals > 1)."""
    env = make_eval_env(cfg)
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


def save_eval_results(results, all_states, all_actions, run_dir, prefix=""):
    """Save evaluation stats and trajectories to run_dir with optional prefix."""
    os.makedirs(run_dir, exist_ok=True)

    stats_name = f"{prefix}eval_stats.yaml" if prefix else "eval_stats.yaml"
    stats_path = os.path.join(run_dir, stats_name)
    with open(stats_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"Stats saved to {stats_path}")

    traj_name = f"{prefix}traj.npz" if prefix else "traj.npz"
    traj_path = os.path.join(run_dir, traj_name)
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
        perturb_scale: fraction of action space bounds to use as perturbation
            range. E.g. 0.5 with action range [-2, 2] gives perturbations in
            [-1, 1]. If None, uses full action space bounds (scale=1.0).
        fix_perturb_range: if True, sample perturbations only in the range that
            doesn't saturate the controller given the current base action.
        hold_steps: number of steps to hold each sampled perturbation before
            resampling a new one. Default 1 (resample every step).

    Returns:
        list of (states: (T+1, S), base_actions: (T, A), perturbations: (T, A))
    """
    np.random.seed(seed)
    action_low = env.action_space.low
    action_high = env.action_space.high
    # Perturbation range: sample u ~ uniform(-1, 1) scaled by action bounds * perturb_scale
    scale = perturb_scale if perturb_scale is not None else 1.0
    perturb_low_default = action_low * scale
    perturb_high_default = action_high * scale
    # Absolute max perturbation magnitude per dimension (for fix_perturb_range clipping)
    perturb_mag = np.abs(action_high) * scale

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
                    p_low = np.clip(action_low - base_action, -perturb_mag, 0)
                    p_high = np.clip(action_high - base_action, 0, perturb_mag)
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

    results, all_states, all_actions = evaluate_policy(env, policy, cfg)
    save_eval_results(results, all_states, all_actions, run_dir, prefix="base_eval_")

    print(f"Phase 0 complete.")
    return results


def phase_1_train_koopman(model, env, policy, cfg, run_dir, augment=True, obs_scale=None,
                          act_scale=None):
    """Phase 1: pre-train Koopman model with perturbed data but only core losses.

    Same data collection as Phase 2 (base policy + random perturbations), but
    only recon, pred, and latent_consistency losses are active.
    """
    print("\n=== Phase 1: Pre-Train Koopman Model (Core Losses Only) ===")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 1. Collect data with base policy + random perturbations (same as Phase 2)
    trajectories = collect_perturbed_data(
        env, policy, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"],
        perturb_scale=cfg.get("perturb_scale", None),
        fix_perturb_range=cfg.get("fix_perturb_range", False),
        hold_steps=cfg.get("hold_steps", 1),
    )

    # 2. Prepare for Koopman training (same augmentation as Phase 2)
    aug_trajectories = augment_perturbed_trajectories(trajectories, augment=augment,
                                                      obs_scale=obs_scale,
                                                      act_scale=act_scale)

    # 3. Train with only core losses (override config to disable optional losses)
    phase1_cfg = dict(cfg)
    # Disable all optional losses — keep only recon, pred, latent_consistency
    for key in ("controllability_loss", "bi_lipschitz_loss", "spectral_loss",
                "normality_loss", "cl_normality_loss", "b_eigen_loss",
                "b_scale_loss", "eig_spread_loss", "unstable_ctrl_loss",
                "upper_lipschitz_loss", "unit_circle_gap_loss"):
        phase1_cfg[key] = False

    model = train(model, aug_trajectories, phase1_cfg)

    # 4. Save checkpoint
    ckpt_path = os.path.join(run_dir, "koop_a_checkpoint.pt")
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 5. Evaluate on training data
    fig, error_stats, heatmap_data = evaluate_model(
        model, aug_trajectories, cfg["horizon"], eval_horizon=25)

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

    print(f"Phase 1 complete.")


def phase_2_train_B(model, env, policy, cfg, run_dir, augment=True, obs_scale=None,
                    act_scale=None):
    """Phase 2: train A and B with random perturbations on top of base policy."""
    print("\n=== Phase 2: Train B Matrix (Perturbed Policy) ===")

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

    # 5. Save checkpoint
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "koopman_ckpt.pt")
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 6. Evaluate on training data
    fig, error_stats, heatmap_data = evaluate_model(
        model, aug_trajectories, cfg["horizon"], eval_horizon=25)

    # 7. Save heatmap
    plot_path = os.path.join(run_dir, "koop_b_prediction_error.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")

    # 8. Save eval stats
    eval_stats = {
        **error_stats,
        "heatmap": heatmap_data,
    }
    stats_path = os.path.join(run_dir, "koop_b_eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(eval_stats, f, default_flow_style=False, sort_keys=False)
    print(f"Eval stats saved to {stats_path}")

    print(f"Phase 2 complete.")
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
    q_scale = cfg.get("q_scale", 1.0)
    Q = torch.eye(latent_dim) * q_scale
    print(f"  Q: {q_scale} * I")
    R = torch.eye(action_dim) * cfg["r_scale"]
    lqr = LQR(A, B_mat, Q, R, q_scale=q_scale,
              controllable_subspace=cfg.get("controllable_subspace", False),
              ctrl_threshold=cfg.get("ctrl_threshold", None))
    F = lqr.F
    gain_norm = lqr.gain_norm.item()
    print(f"  R scale:                               {cfg['r_scale']}")
    if lqr.V_ctrl is not None:
        B_for_P = lqr.V_ctrl @ B_mat
    else:
        B_for_P = B_mat
    BtPB = (B_for_P.T @ lqr.P @ B_for_P).item()
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

    # One-step prediction error in latent space
    model.to(device)
    model.eval()
    all_latent_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_next = model.encode(states_t[1:T_act + 1])
            z_pred = model.predict(z_all, actions_t[:T_act])
            errs = torch.linalg.norm(z_next - z_pred, dim=-1)
            all_latent_errs.append(errs.cpu())
    all_latent_errs = torch.cat(all_latent_errs)
    err_mean = all_latent_errs.mean().item()
    err_std = all_latent_errs.std().item()
    err_2sigma = err_mean + 2 * err_std
    print(f"  " + "-" * 48)
    print(f"  One-step latent error mean:            {err_mean:.6f}")
    print(f"  One-step latent error std:             {err_std:.6f}")
    if err_2sigma > max_runtime_error_latent:
        print(f"\033[91m  One-step latent error mean+2σ:         {err_2sigma:.6f}\033[0m")
    else:
        print(f"\033[92m  One-step latent error mean+2σ:         {err_2sigma:.6f}\033[0m")

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
    heatmap_name = "eigen_heatmap.png"
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
    stats_path = os.path.join(phase_dir, "eigen_variables.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"Variables saved to {stats_path}")

    print(f"Phase 3 complete. Results in {phase_dir}")
    return variables, lqr


def phase_3_lyapunov(model, cfg, phase_dir, aug_trajectories,
                     tuning_config=None):
    """Phase 3 (Lyapunov): compute stability variables using Lyapunov bound."""
    import math
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

    print("\n" + "=" * 60)
    print("  Phase 3: Lyapunov Stability Analysis")
    print("=" * 60)
    os.makedirs(phase_dir, exist_ok=True)

    from controllers.lqr import LQR
    from model.utils import (compute_lower_lipschitz,
                             state_error_to_latent_error)

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

    # =====================================================================
    #  Control Analysis
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Control Analysis")
    print("=" * 50)

    A_np = A.numpy()
    B_np = B_mat.numpy()
    C_mat = np.hstack([np.linalg.matrix_power(A_np, i) @ B_np
                       for i in range(A_np.shape[0])])
    ctrl_rank = np.linalg.matrix_rank(C_mat)
    print(f"  Controllability rank:                  {ctrl_rank} / {A_np.shape[0]}")

    eigenvalues, V = torch.linalg.eig(A)
    unstable_mask = eigenvalues.abs() > 1.0
    if unstable_mask.any():
        for i, (ev, unstable) in enumerate(zip(eigenvalues, unstable_mask)):
            if unstable:
                proj = (V[:, i].conj() @ B_mat.to(torch.cfloat)).abs()
                print(f"      Unstable mode λ={ev.abs():.4f}, B projection={proj.item():.4f}")
    else:
        print("      No unstable modes detected")

    # =====================================================================
    #  Latent / X-Space Relations
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Latent / X-Space Relations")
    print("=" * 50)

    # Encoder lower Lipschitz bound (m)
    model_cpu = model.cpu()
    training_states = []
    for states, actions in aug_trajectories:
        for s in states:
            training_states.append(s)
    m = compute_lower_lipschitz(model_cpu.encode, training_states)
    model.to(device)

    max_tracking_error_x = cfg["max_tracking_error_x"]
    max_displacement_x = cfg["max_displacement_x"]

    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)

    print(f"  max_tracking_error_x:                  {max_tracking_error_x:.6f}")
    print(f"  max_tracking_error_latent (ε_max):     {max_tracking_error_latent:.6f}")
    print(f"  max_displacement_x:                    {max_displacement_x:.6f}")
    print("  " + "-" * 48)
    # Color m: red if <0.5 or >1.5
    if m < 0.5 or m > 1.5:
        print(f"\033[91m  Encoder lower Lipschitz (m):           {m:.6f}\033[0m")
    else:
        print(f"  Encoder lower Lipschitz (m):           {m:.6f}")
    print(f"  eta (m * max_displacement_x):          {eta:.6f}")

    # =====================================================================
    #  LQR Stability
    # =====================================================================
    print("\n" + "=" * 50)
    print("  LQR Stability")
    print("=" * 50)

    latent_dim = cfg["latent_dim"]
    action_dim = cfg["action_dim"]
    q_scale = cfg.get("q_scale", 1.0)
    R = torch.eye(action_dim) * cfg["r_scale"]
    scale_B = cfg.get("scale_B", False)
    if scale_B:
        print("Scaling B")
        B_scale = torch.norm(B_mat, p=2)
        B_norm = B_mat / B_scale
        B_for_lqr = B_norm
    else:
        B_scale = 1.0
        B_for_lqr = B_mat

    Q = torch.eye(latent_dim) * q_scale

    lqr = LQR(A, B_for_lqr, Q, R, q_scale=q_scale,
              controllable_subspace=cfg.get("controllable_subspace", False),
              ctrl_threshold=cfg.get("ctrl_threshold", None))
    gain_norm = lqr.gain_norm.item()
    P = lqr.P

    # Condition number of P
    P_eigvals = torch.linalg.eigvalsh(P)
    kappa_P = (P_eigvals.max() / P_eigvals.min()).item()

    # Lyapunov contraction rate — use Q_ctrl if in subspace mode
    Q_for_cert = getattr(lqr, 'Q_ctrl', lqr.Q)
    Q_eigvals = torch.linalg.eigvalsh(Q_for_cert)
    rho_sq = 1.0 - Q_eigvals.min().item() / P_eigvals.max().item()

    # Lyapunov delta_max
    delta_max_lyap = math.sqrt(max(max_tracking_error_latent**2 * (1.0 - rho_sq) / kappa_P, 0.0)) - eta

    print(f"  Q scale:                               {cfg.get('q_scale', 1.0)}")
    print(f"  R scale:                               {cfg['r_scale']}")
    if lqr.V_ctrl is not None:
        B_for_P = (lqr.V_ctrl @ B_mat)  # project B into controllable subspace
    else:
        B_for_P = B_mat
    BtPB = (B_for_P.T @ P @ B_for_P).item()
    print(f"  B^T P B:                               {BtPB:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    residual_ctrl_budget = max_tracking_error_latent * gain_norm
    if scale_B:
        residual_ctrl_budget /= B_scale
    if residual_ctrl_budget > 0.5:
        print(f"\033[92m  Residual ctrl budget:                  {residual_ctrl_budget:.6f} N-m\033[0m")
    else:
        print(f"  Residual ctrl budget:                  {residual_ctrl_budget:.6f} N-m")
    # What max_tracking_error_x would give budget = 2 N-m?
    if gain_norm > 0 and m > 0:
        denom = m * gain_norm
        if scale_B:
            denom /= B_scale
        tracking_x_for_2 = 2.0 / denom
        print(f"  max_tracking_error_x for budget=2:     {tracking_x_for_2:.6f}")

        # If budget=2 requires a smaller tracking error, use that instead
        if tracking_x_for_2 < max_tracking_error_x:
            print(f"\033[93m  Clamping max_tracking_error_x: {max_tracking_error_x:.6f} -> "
                  f"{tracking_x_for_2:.6f} (to fit actuator budget)\033[0m")
            max_tracking_error_x = tracking_x_for_2
            max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
            residual_ctrl_budget = max_tracking_error_latent * gain_norm
            if scale_B:
                residual_ctrl_budget /= B_scale
            delta_max_lyap = math.sqrt(max(max_tracking_error_latent**2 * (1.0 - rho_sq) / kappa_P, 0.0)) - eta
            print(f"  Updated max_tracking_error_latent:     {max_tracking_error_latent:.6f}")
            print(f"  Updated residual_ctrl_budget:          {residual_ctrl_budget:.6f} N-m")

    print(f"  κ(P) = λ_max(P)/λ_min(P):             {kappa_P:.6f}")
    print(f"  ρ² (Lyapunov):                         {rho_sq:.6f}")
    if delta_max_lyap < 0:
        print(f"\033[91m  δ_max (Lyapunov):                      {delta_max_lyap:.6f}\033[0m")
    else:
        print(f"\033[92m  δ_max (Lyapunov):                      {delta_max_lyap:.6f}\033[0m")

    # One-step prediction error in latent space
    model_on_device = model.to(device)
    model_on_device.eval()
    all_latent_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model_on_device.encode(states_t[:T_act])
            z_next = model_on_device.encode(states_t[1:T_act + 1])
            z_pred = model_on_device.predict(z_all, actions_t[:T_act])
            errs = torch.linalg.norm(z_next - z_pred, dim=-1)
            all_latent_errs.append(errs.cpu())
    all_latent_errs = torch.cat(all_latent_errs)
    err_mean = all_latent_errs.mean().item()
    err_std = all_latent_errs.std().item()
    err_2sigma = err_mean + 2 * err_std
    print(f"  " + "-" * 48)
    print(f"  One-step latent error mean:                {err_mean:.6f}")
    print(f"  One-step latent error std:                 {err_std:.6f}")
    if err_2sigma > delta_max_lyap:
        print(f"\033[91m  One-step latent error mean+2σ:             {err_2sigma:.6f}\033[0m")
    else:
        print(f"\033[92m  One-step latent error mean+2σ:             {err_2sigma:.6f}\033[0m")

    # Heatmap: delta_max_lyap over (max_tracking_error_x, max_displacement_x)
    n_heatmap = 200
    track_sweep = np.linspace(0.0, 1.0, n_heatmap)
    disp_sweep = np.linspace(0.0, 1.0, n_heatmap)
    Z = np.zeros((n_heatmap, n_heatmap))
    for i, d in enumerate(disp_sweep):
        for j, t in enumerate(track_sweep):
            eps_lat = state_error_to_latent_error(t, m)
            eta_lat = state_error_to_latent_error(d, m)
            Z[i, j] = math.sqrt(max(eps_lat**2 * (1.0 - rho_sq) / kappa_P, 0.0)) - eta_lat

    norm = TwoSlopeNorm(vmin=Z.min(), vcenter=0, vmax=max(Z.max(), 1e-6))
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("RdWtGn", ["red", "white", "green"])
    im = ax.imshow(Z, origin='lower', aspect='auto',
                   extent=[0, 1, 0, 1], norm=norm, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, label="δ_max (Lyapunov)")
    cbar.set_ticks([Z.min(), 0, Z.max()])
    cbar.set_ticklabels([f"{Z.min():.4f}", "0", f"{Z.max():.4f}"])
    ax.plot(max_tracking_error_x, max_displacement_x, 'bx', markersize=10, markeredgewidth=2)
    ax.set_xlabel("max_tracking_error_x")
    ax.set_ylabel("max_displacement_x")
    ax.set_title("Lyapunov δ_max Heatmap")
    ax.text(0.5, -0.12,
            f"κ(P)={kappa_P:.4f}  ρ²={rho_sq:.4f}  m={m:.4f}  Q={cfg.get('q_scale', 1.0)}  R={cfg['r_scale']}",
            transform=ax.transAxes, ha='center', fontsize=9)
    fig.tight_layout()
    heatmap_name = "lyapunov_heatmap.png"
    fig.savefig(os.path.join(phase_dir, heatmap_name), dpi=150)
    plt.close(fig)
    print(f"Saved {heatmap_name}")

    # Save
    variables = {
        "m": float(m),
        "gain_norm": float(gain_norm),
        "kappa_P": float(kappa_P),
        "rho_sq_lyapunov": float(rho_sq),
        "max_tracking_error_x": float(max_tracking_error_x),
        "max_displacement_x": float(max_displacement_x),
        "max_tracking_error_latent": float(max_tracking_error_latent),
        "eta": float(eta),
        "delta_max_lyapunov": float(delta_max_lyap),
        "ctrl_rank": int(ctrl_rank),
        "A": A.numpy().tolist(),
        "B": B_mat.numpy().tolist(),
    }
    stats_path = os.path.join(phase_dir, "lyapunov_variables.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"Variables saved to {stats_path}")

    print(f"Phase 3 (Lyapunov) complete. Results in {phase_dir}")
    return variables, lqr


def phase_4_residual_policy(base_policy, lqr, cfg, run_dir, z_ref_limit=1.0, keep_all_ckpts=False):
    """Phase 4: train residual policy with SAC. Returns trained actor model."""
    print("\n=== Phase 4: Residual Policy Training ===")
    print(f"  z_ref_limit (from eta): {z_ref_limit:.6f}")
    from launch.train_residual import train_residual
    return train_residual(base_policy, lqr, cfg, run_dir, z_ref_limit=z_ref_limit,
                          keep_all_ckpts=keep_all_ckpts)


def phase_5_final_eval(env, base_policy, residual_model, lqr, cfg, run_dir, baseline_results, z_ref_limit=1.0):
    """Phase 5: evaluate combined policy and compare against Phase 0 baseline."""
    print("\n=== Phase 5: Final Combined Policy Benchmark ===")
    import torch
    from launch.train_residual import make_composite_policy

    device = next(residual_model.parameters()).device
    lqr_F_np = lqr.F.numpy().astype(np.float32)
    residual_model.eval()
    action_bounds = (env.action_space.low, env.action_space.high)
    policy = make_composite_policy(base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds)

    results, all_states, all_actions = evaluate_policy(env, policy, cfg)
    save_eval_results(results, all_states, all_actions, run_dir, prefix="final_eval_")

    # Compare against baseline
    # Green (improved) = lower for most metrics, higher for reward/success_rate
    higher_is_better = {"reward", "success_rate"}
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Final':>12} {'Delta':>12}")
    print("-" * 65)

    # Success rate
    base_sr = baseline_results["success_rate"]
    final_sr = results["success_rate"]
    delta_sr = final_sr - base_sr
    color = GREEN if delta_sr >= 0 else RED
    print(f"{'success_rate':<25} {base_sr:>11.1%} {final_sr:>11.1%} {color}{delta_sr:>+11.1%}{RESET}")

    # Per-metric comparison
    metrics = ["length", "energy", "control_torque", "angular_velocity", "reward"]
    for key in metrics:
        base_val = baseline_results["combined"][key]["mean"]
        final_val = results["combined"][key]["mean"]
        delta = final_val - base_val
        if key in higher_is_better:
            color = GREEN if delta >= 0 else RED
        else:
            color = GREEN if delta <= 0 else RED
        print(f"{key:<25} {base_val:>12.3f} {final_val:>12.3f} {color}{delta:>+12.3f}{RESET}")

    print()
    print(f"Phase 5 complete. Results in {phase_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Koopman pipeline runner")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--no-augment", action="store_true", default=False,
                        help="Do not append base policy action to the Koopman state")
    parser.add_argument("--skip-pretrain", action="store_true", default=False,
                        help="Skip Phase 1 (A-only training), go straight to joint A+B training")
    parser.add_argument("--keep-all-ckpts", action="store_true", default=False,
                        help="Save every residual policy checkpoint (not just best)")
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
    train_env = gym.make(cfg["env_name"])

    policy = make_base_policy(cfg)

    # Compute observation and action scaling (single env has 1-D bounds)
    obs_scale = compute_obs_scale(train_env, augment)
    act_scale = compute_act_scale(train_env)
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
    baseline_results = phase_0_base_eval(eval_env, policy, cfg, run_dir)
    if not args.skip_pretrain:
        phase_1_train_koopman(model, train_env, policy, cfg, run_dir, augment, obs_scale,
                              act_scale)
    aug_trajectories = phase_2_train_B(model, train_env, policy, cfg, run_dir, augment, obs_scale,
                                       act_scale)
    variables = {}
    lqr = None
    if cfg.get("use_eigen_bound", False):
        eigen_vars, lqr = phase_3_compute_variables(model, cfg, run_dir, aug_trajectories)
        variables.update(eigen_vars)
    if cfg.get("use_lyapunov_bound", False):
        lyap_vars, lyap_lqr = phase_3_lyapunov(model, cfg, run_dir, aug_trajectories)
        variables.update(lyap_vars)
        if lqr is None:
            lqr = lyap_lqr

    # Check stability feasibility before proceeding
    delta_max = variables.get("max_runtime_error_latent",
                              variables.get("delta_max_lyapunov", None))
    if delta_max is not None and delta_max < 0:
        print(f"\n\033[91mStopping after Phase 3: delta_max = {delta_max:.6f} < 0 "
              f"(infeasible error budget). Tune Q/R scales or retrain.\033[0m")
        eval_env.close()
        train_env.close()
        print(f"\n=== Pipeline stopped. Outputs in {run_dir} ===")
        return

    # Phase 4: Residual policy training
    # Phase 5: Final combined benchmark
    if lqr is not None:
        z_ref_limit = variables.get("eta", 1.0)
        residual_model = phase_4_residual_policy(policy, lqr, cfg, run_dir, z_ref_limit=z_ref_limit,
                                                    keep_all_ckpts=args.keep_all_ckpts)
        phase_5_final_eval(eval_env, policy, residual_model, lqr, cfg, run_dir, baseline_results, z_ref_limit=z_ref_limit)
    else:
        print("\nSkipping Phases 4-5: no LQR available (enable use_eigen_bound)")

    eval_env.close()
    train_env.close()
    print(f"\n=== Pipeline complete. All outputs in {run_dir} ===")

    # Restore stdout and close log
    sys.stdout = tee.stream
    tee.close()


if __name__ == "__main__":
    main()
