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

from launch.eval_policy import evaluate as evaluate_policy, make_policy, make_eval_env, make_single_env
from launch.eval_pendulum import evaluate_model, make_recon_and_lc_heatmaps
from launch.train_pendulum import collect_data, train
from launch.pipeline_utils import (
    Tee, make_device, build_koopman_model, save_checkpoint, load_checkpoint,
    evaluate_and_save,
)
from launch.stability_utils import (
    control_analysis, compute_encoder_lipschitz_bounds, setup_lqr, compute_lyapunov_params,
    compute_BtPB, compute_latent_errors, compute_max_latent_diff,
    run_sdp_optimization, lyapunov_gamma, alpha_bound, count_steps_under_threshold,
)
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
    save_checkpoint(model, cfg, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 5. Evaluate on training data and save
    evaluate_and_save(model, aug_trajectories, cfg, run_dir, "koop_a_",
                      "Prediction Error (A+B, with perturbations, core losses only)")

    # # 8. Theta/thdot heatmaps (reconstruction + LC)
    # make_recon_and_lc_heatmaps(model, aug_trajectories, trajectories,
    #                             run_dir, prefix="phase1_")

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
    save_checkpoint(model, cfg, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    # 6. Evaluate on training data and save
    error_stats = evaluate_and_save(model, aug_trajectories, cfg, run_dir, "koop_b_",
                                     "Prediction Error (A+B, with perturbations, all losses)")

    # # 9. Theta/thdot heatmaps (reconstruction + LC)
    # make_recon_and_lc_heatmaps(model, aug_trajectories, trajectories,
    #                             run_dir, prefix="phase2_")

    print(f"Phase 2 complete.")
    return aug_trajectories, error_stats


def phase_3_compute_variables(model, cfg, phase_dir, aug_trajectories,
                              tuning_config=None, error_stats=None):
    """Phase 3: compute stability and control variables from trained model.

    Args:
        tuning_config: path to a YAML file to override q_scale, r_scale,
            max_tracking_error_x, and max_displacement_x from the saved config.
    """
    print("\n=== Phase 3: Compute Stability Variables ===")
    os.makedirs(phase_dir, exist_ok=True)

    from controllers.lqr import LQR
    from model.utils import (spectral_radius, transient_constant,
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
    B_sigma_max = torch.linalg.norm(B_mat, ord=2).item()
    print(f"  Spectral radius of A (open):           {rho_A:.6f}")
    print(f"  B largest singular value:              {B_sigma_max:.6f}")

    ctrl_rank = control_analysis(A, B_mat)

    # 1. Encoder Lipschitz bounds
    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(model, aug_trajectories, device)
    m = m_full if m_full is not None else 1.0
    if m_gx is not None:
        print(f"  Encoder lower Lipschitz (m, g(x)):     {m_gx:.6f}")
        print(f"  Encoder upper Lipschitz (L, g(x)):     {L_gx:.6f}")
        print(f"  Encoder lower Lipschitz (m, full):     {m_full:.6f}")
        print(f"  Encoder upper Lipschitz (L, full):     {L_full:.6f}")

    # 2. LQR: F and gain norm
    lqr, Q, R, B_scale = setup_lqr(A, B_mat, cfg)
    gain_norm = lqr.gain_norm.item()
    print(f"  Q: {cfg.get('q_scale', 1.0)} * I")
    print(f"  R scale:                               {cfg['r_scale']}")
    A_norm = torch.linalg.norm(A, ord=2).item()
    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    BtPB = compute_BtPB(lqr, B_mat, lqr.P)
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
    err_mean, err_std = compute_latent_errors(model, aug_trajectories, device, error_stats)
    err_2sigma = err_mean + 2 * err_std
    print(f"  " + "-" * 48)
    print(f"  One-step latent error mean:            {err_mean:.6f}")
    print(f"  One-step latent error std:             {err_std:.6f}")
    if err_2sigma > max_runtime_error_latent:
        print(f"\033[91m  One-step latent error mean+2σ:         {err_2sigma:.6f}\033[0m")
    else:
        print(f"\033[92m  One-step latent error mean+2σ:         {err_2sigma:.6f}\033[0m")

    # # Heatmap: max_runtime_error_latent over (max_tracking_error_x, max_displacement_x)
    # import matplotlib.pyplot as plt
    # from matplotlib.colors import TwoSlopeNorm
    # n_heatmap = 200
    # track_sweep = np.linspace(0.0, 1.0, n_heatmap)
    # disp_sweep = np.linspace(0.0, 1.0, n_heatmap)
    # Z = np.zeros((n_heatmap, n_heatmap))
    # for i, d in enumerate(disp_sweep):
    #     for j, t in enumerate(track_sweep):
    #         eps_lat = state_error_to_latent_error(t, m)
    #         eta_lat = state_error_to_latent_error(d, m)
    #         Z[i, j] = max_tolerable_model_error(rho, C, eps_lat, eta_lat)
    #
    # norm = TwoSlopeNorm(vmin=min(Z.min(), -1e-6), vcenter=0, vmax=max(Z.max(), 1e-6))
    # fig3, ax3 = plt.subplots(figsize=(8, 6))
    # from matplotlib.colors import LinearSegmentedColormap
    # cmap = LinearSegmentedColormap.from_list("RdWtGn", ["red", "white", "green"])
    # im = ax3.imshow(Z, origin='lower', aspect='auto',
    #                 extent=[0, 1, 0, 1], norm=norm, cmap=cmap)
    # cbar = fig3.colorbar(im, ax=ax3, label="max_runtime_error_latent")
    # cbar.set_ticks([Z.min(), 0, Z.max()])
    # cbar.set_ticklabels([f"{Z.min():.4f}", "0", f"{Z.max():.4f}"])
    # ax3.plot(max_tracking_error_x, max_displacement_x, 'bx', markersize=10, markeredgewidth=2)
    # ax3.set_xlabel("max_tracking_error_x")
    # ax3.set_ylabel("max_displacement_x")
    # ax3.set_title("Runtime Error Budget Heatmap")
    # ax3.text(0.5, -0.12,
    #          f"rho={rho:.4f}  C={C:.4f}  m={m:.4f}  Q_scale={cfg.get('q_scale', 1.0)}  R_scale={cfg['r_scale']}",
    #          transform=ax3.transAxes, ha='center', fontsize=9)
    # fig3.tight_layout()
    # heatmap_name = "eigen_heatmap.png"
    # fig3.savefig(os.path.join(phase_dir, heatmap_name), dpi=150)
    # plt.close(fig3)
    # print(f"Saved {heatmap_name}")

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
                     tuning_config=None, error_stats=None):
    """Phase 3 (Lyapunov): compute stability variables using Lyapunov bound."""
    import math
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
    from model.utils import state_error_to_latent_error

    print("\n" + "=" * 60)
    print("  Phase 3: Lyapunov Stability Analysis")
    print("=" * 60)
    os.makedirs(phase_dir, exist_ok=True)

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

    ctrl_rank = control_analysis(A, B_mat)

    # =====================================================================
    #  Latent / X-Space Relations
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Latent / X-Space Relations")
    print("=" * 50)

    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(model, aug_trajectories, device)
    m = m_full if m_full is not None else 1.0

    max_tracking_error_x = cfg["max_tracking_error_x"]
    max_displacement_x = cfg["max_displacement_x"]

    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)

    print(f"  max_tracking_error_x:                  {max_tracking_error_x:.6f}")
    print(f"  max_tracking_error_latent (ε_max):     {max_tracking_error_latent:.6f}")
    print(f"  max_displacement_x:                    {max_displacement_x:.6f}")
    print("  " + "-" * 48)
    if m_gx is not None:
        print(f"  Encoder lower Lipschitz (m, g(x)):     {m_gx:.6f}")
        print(f"  Encoder upper Lipschitz (L, g(x)):     {L_gx:.6f}")
        print(f"  Encoder lower Lipschitz (m, full):     {m_full:.6f}")
        print(f"  Encoder upper Lipschitz (L, full):     {L_full:.6f}")
    print(f"  eta (m * max_displacement_x):          {eta:.6f}")

    # =====================================================================
    #  LQR Stability
    # =====================================================================
    print("\n" + "=" * 50)
    print("  LQR Stability")
    print("=" * 50)

    lqr, Q, R_cost, B_scale = setup_lqr(A, B_mat, cfg)
    gain_norm = lqr.gain_norm.item()
    P, kappa_P, rho_sq, P_eigvals = compute_lyapunov_params(lqr, Q, R_cost)
    rho_sq_lqr = rho_sq
    kappa_P_lqr = kappa_P
    scale_B = cfg.get("scale_B", False)

    # SDP optimization of P
    sdp_result = run_sdp_optimization(lqr, max_tracking_error_latent, eta, cfg)
    if sdp_result is not None:
        gamma_lqr = lyapunov_gamma(max_tracking_error_latent, rho_sq_lqr, kappa_P_lqr, eta)
        rho_sq, kappa_P, sdp_gamma = sdp_result
        print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
        print(f"  κ(P) (LQR):                            {kappa_P_lqr:.6f}")
        print(f"  γ_max (LQR):                           {gamma_lqr:.6f}")
        print(f"  ρ² (SDP-optimized):                    {rho_sq:.6f}")
        print(f"  κ(P) (SDP-optimized):                  {kappa_P:.6f}")
        print(f"  γ_max (SDP):                           {sdp_gamma:.6f}")

    # Lyapunov delta_max
    delta_max_lyap = lyapunov_gamma(max_tracking_error_latent, rho_sq, kappa_P, eta)

    print(f"  Q scale:                               {cfg.get('q_scale', 1.0)}")
    print(f"  R scale:                               {cfg['r_scale']}")
    A_norm = torch.linalg.norm(A, ord=2).item()
    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    BtPB = compute_BtPB(lqr, B_mat, P)
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
            delta_max_lyap = lyapunov_gamma(max_tracking_error_latent, rho_sq, kappa_P, eta)
            print(f"  Updated max_tracking_error_latent:     {max_tracking_error_latent:.6f}")
            print(f"  Updated residual_ctrl_budget:          {residual_ctrl_budget:.6f} N-m")

    print(f"  κ(P) = λ_max(P)/λ_min(P):             {kappa_P:.6f}")
    print(f"  ρ² (Lyapunov):                         {rho_sq:.6f}")
    if delta_max_lyap < 0:
        print(f"\033[91m  δ_max (Lyapunov):                      {delta_max_lyap:.6f}\033[0m")
    else:
        print(f"\033[92m  δ_max (Lyapunov):                      {delta_max_lyap:.6f}\033[0m")

    # One-step prediction error in latent space
    err_mean, err_std = compute_latent_errors(model, aug_trajectories, device, error_stats)
    err_2sigma = err_mean + 2 * err_std
    print(f"  " + "-" * 48)
    print(f"  One-step latent error mean:                {err_mean:.6f}")
    print(f"  One-step latent error std:                 {err_std:.6f}")
    if err_2sigma > delta_max_lyap:
        print(f"\033[91m  One-step latent error mean+2σ:             {err_2sigma:.6f}\033[0m")
    else:
        print(f"\033[92m  One-step latent error mean+2σ:             {err_2sigma:.6f}\033[0m")

    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, delta_max_lyap, space="latent")
    print(f"  Steps under δ_max:                        {count_under}/{total} ({fraction*100:.1f}%)")

    # # Heatmap: delta_max_lyap over (max_tracking_error_x, max_displacement_x)
    # n_heatmap = 200
    # track_sweep = np.linspace(0.0, 1.0, n_heatmap)
    # disp_sweep = np.linspace(0.0, 1.0, n_heatmap)
    # Z = np.zeros((n_heatmap, n_heatmap))
    # for i, d in enumerate(disp_sweep):
    #     for j, t in enumerate(track_sweep):
    #         eps_lat = state_error_to_latent_error(t, m)
    #         eta_lat = state_error_to_latent_error(d, m)
    #         Z[i, j] = lyapunov_gamma(eps_lat, rho_sq, kappa_P, eta_lat)
    #
    # z_min, z_max = Z.min(), Z.max()
    # norm = TwoSlopeNorm(vmin=min(z_min, -1e-6), vcenter=0, vmax=max(z_max, 1e-6))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # cmap = LinearSegmentedColormap.from_list("RdWtGn", ["red", "white", "green"])
    # im = ax.imshow(Z, origin='lower', aspect='auto',
    #                extent=[0, 1, 0, 1], norm=norm, cmap=cmap)
    # cbar = fig.colorbar(im, ax=ax, label="δ_max (Lyapunov)")
    # cbar.set_ticks([Z.min(), 0, Z.max()])
    # cbar.set_ticklabels([f"{Z.min():.4f}", "0", f"{Z.max():.4f}"])
    # ax.plot(max_tracking_error_x, max_displacement_x, 'bx', markersize=10, markeredgewidth=2)
    # ax.set_xlabel("max_tracking_error_x")
    # ax.set_ylabel("max_displacement_x")
    # ax.set_title("Lyapunov δ_max Heatmap")
    # ax.text(0.5, -0.12,
    #         f"κ(P)={kappa_P:.4f}  ρ²={rho_sq:.4f}  m={m:.4f}  Q={cfg.get('q_scale', 1.0)}  R={cfg['r_scale']}",
    #         transform=ax.transAxes, ha='center', fontsize=9)
    # fig.tight_layout()
    # heatmap_name = "lyapunov_heatmap.png"
    # fig.savefig(os.path.join(phase_dir, heatmap_name), dpi=150)
    # plt.close(fig)
    # print(f"Saved {heatmap_name}")

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


def lipschitz_m_free(model, cfg, phase_dir, aug_trajectories, env,
                     error_stats=None):
    """Lyapunov stability analysis without Lipschitz constant m.

    Derives all latent-space bounds from measured model errors and control
    limits rather than converting from state-space via m.

    Args:
        model: KoopmanAutoencoder (on device)
        cfg: config dict
        phase_dir: output directory
        aug_trajectories: list of (koopman_states, actions)
        env: single gym environment (for action bounds)
    """
    import math
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    print("\n" + "=" * 60)
    print("  Phase 3: Lipschitz-m-Free Stability Analysis")
    print("=" * 60)
    os.makedirs(phase_dir, exist_ok=True)

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

    ctrl_rank = control_analysis(A, B_mat)

    # =====================================================================
    #  Set Constants
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Set Constants")
    print("=" * 50)

    q_scale = cfg.get("q_scale", 1.0)
    r_scale = cfg["r_scale"]
    u_max = float(np.max(np.abs(env.action_space.high)))

    print(f"  Q scale:                               {q_scale}")
    print(f"  R scale:                               {r_scale}")
    print(f"  u_max:                                 {u_max:.6f}")

    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(model, aug_trajectories, device)
    m = m_full if m_full is not None else 1.0
    if m_gx is not None:
        print(f"  Encoder lower Lipschitz (m, g(x)):     {m_gx:.6f}")
        print(f"  Encoder upper Lipschitz (L, g(x)):     {L_gx:.6f}")
        print(f"  Encoder lower Lipschitz (m, full):     {m_full:.6f}")
        print(f"  Encoder upper Lipschitz (L, full):     {L_full:.6f}")

    max_latent_diff = compute_max_latent_diff(model, cfg, device)
    print(f"  max latent space difference:           {max_latent_diff:.6f}")

    # =====================================================================
    #  One-Step Latent Error
    # =====================================================================
    print("  " + "-" * 48)

    err_mean, err_std = compute_latent_errors(model, aug_trajectories, device, error_stats)
    R_val = err_mean + 2 * err_std

    print(f"  One-step latent error mean:             {err_mean:.6f}")
    print(f"  One-step latent error std:              {err_std:.6f}")
    print(f"  R (mean + 2σ):                          {R_val:.6f}")
    print(f"  R / max_latent_diff:                    {R_val / max_latent_diff:.6f}")

    # =====================================================================
    #  LQR Stability
    # =====================================================================
    print("\n" + "=" * 50)
    print("  LQR Stability")
    print("=" * 50)

    lqr, Q, R_cost, B_scale_val = setup_lqr(A, B_mat, cfg)
    gain_norm = lqr.gain_norm.item()
    P, kappa_P, rho_sq, P_eigvals = compute_lyapunov_params(lqr, Q, R_cost)
    rho_sq_lqr = rho_sq
    kappa_P_lqr = kappa_P
    F = lqr.F

    # LQR-derived values
    P_lqr_eigs = P_eigvals
    lam_min_P_lqr = P_lqr_eigs.min().item()
    lam_max_P_lqr = P_lqr_eigs.max().item()
    BtPB_lqr = compute_BtPB(lqr, B_mat, P)
    A_norm = torch.linalg.norm(A, ord=2).item()

    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    print(f"  B^T P B (LQR):                         {BtPB_lqr:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
    print(f"  κ(P) (LQR):                            {kappa_P_lqr:.6f}")
    print(f"  λ_min(P) (LQR):                        {lam_min_P_lqr:.6f}")
    print(f"  λ_max(P) (LQR):                        {lam_max_P_lqr:.6f}")

    # SDP optimization of P
    eps_for_opt = m * cfg["max_tracking_error_x"]
    BF_pre = B_mat @ F
    BF_norm_pre = torch.linalg.norm(BF_pre, ord=2).item()
    eta_for_opt = BF_norm_pre * (u_max / gain_norm)
    gamma_lqr = lyapunov_gamma(eps_for_opt, rho_sq_lqr, kappa_P_lqr, eta_for_opt)
    print(f"  γ_max (LQR, m-derived ε):              {gamma_lqr:.6f}")

    print("\n" + "=" * 50)
    print("  SDP Optimization")
    print("=" * 50)
    sdp_result = run_sdp_optimization(lqr, eps_for_opt, eta_for_opt, cfg)
    if sdp_result is not None:
        rho_sq, kappa_P, sdp_gamma = sdp_result
        print(f"  ρ² (SDP-optimized):                    {rho_sq:.6f}")
        print(f"  κ(P) (SDP-optimized):                  {kappa_P:.6f}")
        print(f"  γ_max (SDP, m-derived ε):              {sdp_gamma:.6f}")

    # Final values (SDP if available, else LQR)
    print("  " + "-" * 48)
    print(f"  ρ² (final):                            {rho_sq:.6f}")
    print(f"  κ(P) (final):                          {kappa_P:.6f}")

    # =====================================================================
    #  Derived Quantities
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Derived Quantities")
    print("=" * 50)

    # z_ref_limit
    z_ref_limit = u_max / gain_norm
    print(f"  z_ref_limit = u_max / ||F||:           {z_ref_limit:.6f}")

    # eta = ||B @ F||_2 * z_ref_limit
    BF = B_mat @ F
    BF_norm = torch.linalg.norm(BF, ord=2).item()
    eta = BF_norm * z_ref_limit
    print(f"  ||B @ F||_2:                           {BF_norm:.6f}")
    print(f"  eta = ||BF|| * z_ref_limit:            {eta:.6f}")

    # epsilon_max = (R + eta) * sqrt(kappa_P) / (1 - rho)
    if rho_sq < 1.0:
        epsilon_max = (R_val + eta) * math.sqrt(kappa_P) / (1.0 - math.sqrt(rho_sq))
    else:
        epsilon_max = float('inf')
    if epsilon_max > max_latent_diff / 2:
        print(f"\033[91m  ε_max (max tracking error latent):     {epsilon_max:.6f}\033[0m")
    else:
        print(f"\033[92m  ε_max (max tracking error latent):     {epsilon_max:.6f}\033[0m")
    print(f"  ε_max / max_latent_diff:               {epsilon_max / max_latent_diff:.6f}")

    gamma_max = lyapunov_gamma(epsilon_max, rho_sq, kappa_P, eta)
    if gamma_max < 0:
        print(f"\033[91m  γ_max (Lyapunov):                      {gamma_max:.6f}\033[0m")
    else:
        print(f"\033[92m  γ_max (Lyapunov):                      {gamma_max:.6f}\033[0m")

    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, gamma_max, space="latent")
    print(f"  Steps under γ_max:                     {count_under}/{total} ({fraction*100:.1f}%)")

    # # =====================================================================
    # #  Heatmap: R vs Residual Control → ε_max / max_latent_diff
    # # =====================================================================
    # n_heatmap = 200
    # R_max = err_mean + 3 * err_std
    # R_sweep = np.linspace(0, R_max, n_heatmap)
    # u_res_sweep = np.linspace(0, u_max, n_heatmap)
    # Z = np.zeros((n_heatmap, n_heatmap))
    #
    # for i, u_res in enumerate(u_res_sweep):
    #     z_ref_i = u_res / gain_norm if gain_norm > 0 else 0.0
    #     eta_i = BF_norm * z_ref_i
    #     for j, R_j in enumerate(R_sweep):
    #         if rho_sq < 1.0:
    #             eps_j = (R_j + eta_i) * math.sqrt(kappa_P) / (1.0 - math.sqrt(rho_sq))
    #             Z[i, j] = np.clip(eps_j / max_latent_diff, 0.0, 1.0)
    #         else:
    #             Z[i, j] = 1.0
    #
    # fig, ax = plt.subplots(figsize=(10, 7))
    # cmap = LinearSegmentedColormap.from_list("WtGn", ["white", "green"])
    # im = ax.pcolormesh(R_sweep, u_res_sweep, Z, cmap=cmap, vmin=0, vmax=1, shading="nearest")
    # cbar = fig.colorbar(im, ax=ax, pad=0.02)
    # cbar.set_label("ε_max / max_latent_diff")
    #
    # y_bot = u_max * 0.02
    # ax.axvline(x=err_mean, color="red", linestyle="-", linewidth=1.5)
    # ax.text(err_mean - R_max * 0.01, y_bot, r"$\mu$", color="black", fontsize=10,
    #         ha="right", va="bottom", fontweight="bold",
    #         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    # for k in range(1, 4):
    #     x_pos = err_mean + k * err_std
    #     x_neg = err_mean - k * err_std
    #     if x_pos <= R_max:
    #         ax.axvline(x=x_pos, color="red", linestyle="--", linewidth=1.0)
    #         ax.text(x_pos - R_max * 0.01, y_bot, f"${k}\\sigma$", color="black", fontsize=9,
    #                 ha="right", va="bottom",
    #                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    #     if x_neg >= 0:
    #         ax.axvline(x=x_neg, color="red", linestyle="--", linewidth=1.0)
    #         ax.text(x_neg - R_max * 0.01, y_bot, f"$-{k}\\sigma$", color="black", fontsize=9,
    #                 ha="right", va="bottom",
    #                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1))
    #
    # z_max_val = Z.max()
    # all_contour_levels = [0.05, 0.1, 0.15, 0.2, 0.5]
    # all_contour_colors = ["darkgreen", "blue", "darkorange", "purple", "red"]
    # contour_levels = [lvl for lvl in all_contour_levels if lvl <= z_max_val]
    # contour_colors = [c for lvl, c in zip(all_contour_levels, all_contour_colors) if lvl <= z_max_val]
    # if contour_levels:
    #     cs = ax.contour(R_sweep, u_res_sweep, Z, levels=contour_levels,
    #                      colors=contour_colors, linestyles="dotted", linewidths=1.5)
    #     contour_handles = [plt.Line2D([], [], color=c, linestyle="dotted", linewidth=1.5,
    #                                    label=f"{lvl:.0%}") for lvl, c in zip(contour_levels, contour_colors)]
    #     ax.legend(handles=contour_handles, loc="upper left", fontsize=8, title="ε_max / Δz_max")
    #
    # ax.plot(R_val, u_max, 'bx', markersize=12, markeredgewidth=2)
    # ax.set_xlabel("R (one-step latent error)")
    # ax.set_ylabel("Residual Control Budget (N·m)")
    # ax.set_title("ε_max / max_latent_diff vs Model Error & Control Budget")
    # ax.text(0.5, -0.10,
    #         f"κ(P)={kappa_P:.4f}  ρ²={rho_sq:.4f}  ||F||={gain_norm:.4f}  ||BF||={BF_norm:.4f}  "
    #         f"Q={q_scale}  R_lqr={r_scale}",
    #         transform=ax.transAxes, ha='center', fontsize=8)
    # fig.tight_layout()
    # heatmap_name = "m_free_gamma_max_heatmap.png"
    # fig.savefig(os.path.join(phase_dir, heatmap_name), dpi=150)
    # plt.close(fig)
    # print(f"Saved {heatmap_name}")

    # Save variables
    variables = {
        "m": float(m),
        "u_max": float(u_max),
        "max_latent_diff": float(max_latent_diff),
        "gain_norm": float(gain_norm),
        "BF_norm": float(BF_norm),
        "kappa_P": float(kappa_P),
        "rho_sq_lyapunov": float(rho_sq),
        "R_model_error": float(R_val),
        "latent_error_mean": float(err_mean),
        "latent_error_std": float(err_std),
        "z_ref_limit": float(z_ref_limit),
        "eta": float(eta),
        "max_tracking_error_latent": float(epsilon_max),
        "gamma_max_lyapunov": float(gamma_max),
        "ctrl_rank": int(ctrl_rank),
        "A": A.numpy().tolist(),
        "B": B_mat.numpy().tolist(),
    }
    stats_path = os.path.join(phase_dir, "m_free_variables.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"Variables saved to {stats_path}")

    print(f"Phase 3 (m-free) complete. Results in {phase_dir}")
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
    print(f"Phase 5 complete. Results in {run_dir}")
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
    parser.add_argument("--no-residual", action="store_true", default=False,
                        help="Skip Phases 4-5 (residual policy training and final eval)")
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

    policy = make_base_policy(cfg)

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
    if not args.skip_pretrain:
        phase_1_train_koopman(model, train_env, policy, cfg, run_dir, augment, obs_scale,
                              act_scale)
    aug_trajectories, error_stats = phase_2_train_B(model, train_env, policy, cfg, run_dir, augment, obs_scale,
                                       act_scale)
    variables = {}
    lqr = None
    if cfg.get("use_eigen_bound", False):
        eigen_vars, lqr = phase_3_compute_variables(model, cfg, run_dir, aug_trajectories,
                                                     error_stats=error_stats)
        variables.update(eigen_vars)
    if cfg.get("use_lyapunov_bound", False):
        lyap_vars, lyap_lqr = phase_3_lyapunov(model, cfg, run_dir, aug_trajectories,
                                                error_stats=error_stats)
        variables.update(lyap_vars)
        if lqr is None:
            lqr = lyap_lqr
    if cfg.get("use_m_free_bound", False):
        mfree_vars, mfree_lqr = lipschitz_m_free(model, cfg, run_dir, aug_trajectories, train_env,
                                                    error_stats=error_stats)
        variables.update(mfree_vars)
        if lqr is None:
            lqr = mfree_lqr
    if cfg.get("use_alpha_bound", False):
        if lqr is None:
            # Need LQR for alpha bound initialization
            lqr_ab, _, _, _ = setup_lqr(model.A.detach().cpu(), model.B_matrix.detach().cpu(), cfg)
            lqr = lqr_ab
        alpha_vars = alpha_bound(model, lqr, cfg, aug_trajectories, train_env,
                                  error_stats=error_stats)
        variables.update(alpha_vars)

    # Check stability feasibility before proceeding
    delta_max = variables.get("max_runtime_error_latent",
                              variables.get("delta_max_lyapunov",
                              variables.get("gamma_max_lyapunov",
                              variables.get("gamma_max_alpha", None))))
    if delta_max is not None and delta_max < 0:
        print(f"\n\033[91mStopping after Phase 3: delta_max = {delta_max:.6f} < 0 "
              f"(infeasible error budget). Tune Q/R scales or retrain.\033[0m")
        eval_env.close()
        train_env.close()
        print(f"\n=== Pipeline stopped. Outputs in {run_dir} ===")
        return

    # Phase 4: Residual policy training
    # Phase 5: Final combined benchmark
    if args.no_residual:
        print("\nSkipping Phases 4-5 (--no-residual)")
    elif lqr is not None:
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
