"""
Analytically solve for the B matrix of a pre-trained Koopman model.

Loads a Koopman model (A matrix already trained), collects perturbed
trajectory data, and solves for B via least-squares with a controllability
projection, normalized to unit spectral norm.

Usage:
    python -m launch.analy_b_tuning output/pretrain_2026-03-26_21-40-50
"""
import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from launch.eval_pendulum import obs_to_angle
from launch.run import (
    make_env,
    make_base_policy,
    compute_obs_scale,
    collect_perturbed_data,
    augment_perturbed_trajectories,
    save_config,
)
from model.autoencoder import KoopmanAutoencoder

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"


def _fmt(arr):
    """Format a numpy array with fixed-point notation to 3 decimal places."""
    return np.array2string(np.asarray(arr), formatter={'float_kind': lambda x: f"{x:.3f}"})


def _ctrl_rank_and_sv(A, B, n):
    """Compute controllability matrix rank and singular values."""
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    sv = np.linalg.svd(C, compute_uv=False)
    rank = int(np.linalg.matrix_rank(C))
    return rank, sv, C


def project_for_controllability(A, B_ls, n):
    """Project B onto the maximally controllable subspace.

    Uses a multi-stage approach:
      1. PBH test to find eigenvectors orthogonal to B
      2. Direct augmentation of B in failing directions
      3. SVD-based fallback for remaining rank deficiency

    All linear algebra is done in float64 for numerical stability.

    Args:
        A: (n, n) dynamics matrix
        B_ls: (n, m) least-squares B matrix
        n: latent dimension

    Returns:
        B_projected: (n, m) with full controllability rank
    """
    # Upcast to float64 for numerical stability
    A = A.astype(np.float64)
    B_ls = B_ls.astype(np.float64)
    m = B_ls.shape[1]
    b_norm = np.linalg.norm(B_ls, ord=2)

    rank_init, sv_init, _ = _ctrl_rank_and_sv(A, B_ls, n)
    print(f"  Initial controllability rank: {rank_init}/{n}")
    print(f"  Controllability SV: {_fmt(sv_init)}")

    if rank_init == n:
        print(f"  {GREEN}B_ls already fully controllable.{RESET}")
        return B_ls.astype(np.float32)

    # --- Stage 1: PBH test ---
    eigvals, V_right = np.linalg.eig(A)
    try:
        W = np.linalg.inv(V_right).conj()  # rows are left eigenvectors
    except np.linalg.LinAlgError:
        print(f"  {RED}PROBLEM: A is defective (non-diagonalizable).{RESET}")
        print(f"  {YELLOW}FALLBACK: Using SVD of controllability matrix directly.{RESET}")
        W = None

    B_aug = B_ls.copy()

    if W is not None:
        # Check PBH condition: w^H B must be nonzero for each left eigenvector w
        pbh_failures = []
        for i in range(n):
            w = W[i]
            projection = np.abs(w @ B_ls)
            if np.max(projection) < b_norm * 1e-4:
                pbh_failures.append(i)

        print(f"  PBH test: {len(pbh_failures)} failing eigenvalue(s) "
              f"(threshold: |w^H B| < {b_norm * 1e-4:.3f})")
        for idx in pbh_failures:
            print(f"    lambda={eigvals[idx]:.3f}, |lambda|={abs(eigvals[idx]):.3f}, "
                  f"|w^H B|={np.max(np.abs(W[idx] @ B_ls)):.3f}")

        # Add corrections for PBH-failing directions
        correction_scale = max(b_norm, 1e-3)
        already_added = []

        for idx in pbh_failures:
            w = W[idx]
            # Try real part first, then imaginary
            for part_name, part in [("real", w.real), ("imag", w.imag)]:
                norm = np.linalg.norm(part)
                if norm < 1e-12:
                    continue
                direction = part / norm

                # Skip duplicates (conjugate pairs share real/imag parts)
                is_dup = any(abs(np.dot(direction, p)) > 0.99 for p in already_added)
                if is_dup:
                    continue

                already_added.append(direction)
                B_aug += correction_scale * direction[:, None] @ np.ones((1, m))
                print(f"    Added {part_name} direction for lambda={eigvals[idx]:.3f}")

    rank_pbh, sv_pbh, _ = _ctrl_rank_and_sv(A, B_aug, n)
    print(f"  After PBH augmentation: rank = {rank_pbh}/{n}")
    print(f"  SV: {_fmt(sv_pbh)}")

    if rank_pbh == n:
        print(f"  {GREEN}Full rank achieved via PBH augmentation.{RESET}")
        return B_aug.astype(np.float32)

    # --- Stage 2: SVD-based boost for remaining weak directions ---
    print(f"  {RED}PROBLEM: PBH augmentation insufficient (rank {rank_pbh}/{n}).{RESET}")
    print(f"  {YELLOW}FALLBACK: Directly boosting weak SVD directions of "
          f"controllability matrix.{RESET}")

    C_mat = np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)])
    U_svd, S, _ = np.linalg.svd(C_mat, full_matrices=True)

    # Boost scale: make weak directions comparable to the median SV
    target_sv = np.median(S[:rank_pbh]) if rank_pbh > 0 else b_norm
    boost = max(target_sv * 0.1, b_norm, 1e-3)

    for i in range(n):
        if S[i] < S[0] * 1e-8:  # weak direction
            direction = U_svd[:, i]
            B_aug += boost * direction[:, None] @ np.ones((1, m))
            print(f"    Boosted SVD direction {i}: S={S[i]:.3f} -> adding scale {boost:.3f}")

    rank_svd, sv_svd, _ = _ctrl_rank_and_sv(A, B_aug, n)
    print(f"  After SVD boost: rank = {rank_svd}/{n}")
    print(f"  SV: {_fmt(sv_svd)}")

    if rank_svd == n:
        print(f"  {GREEN}Full rank achieved via SVD boost.{RESET}")
        return B_aug.astype(np.float32)

    # --- Stage 3: Iterative random perturbation (last resort) ---
    print(f"  {RED}PROBLEM: SVD boost insufficient (rank {rank_svd}/{n}).{RESET}")
    print(f"  {YELLOW}FALLBACK: Iterative perturbation with rank checking.{RESET}")

    rng = np.random.RandomState(42)
    for attempt in range(100):
        C_cur = np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)])
        U_c, S_c, _ = np.linalg.svd(C_cur, full_matrices=True)

        # Find weakest direction and perturb B along it
        weakest_idx = np.argmin(S_c)
        direction = U_c[:, weakest_idx]
        # Also add a small random component for diversity
        noise = rng.randn(n)
        noise -= direction * np.dot(noise, direction)  # orthogonalize to strong dirs
        noise /= (np.linalg.norm(noise) + 1e-12)

        perturb = boost * (direction + 0.1 * noise)
        B_aug += perturb[:, None] @ np.ones((1, m))

        rank_try = np.linalg.matrix_rank(
            np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)]))
        if rank_try == n:
            print(f"    Achieved full rank after {attempt + 1} perturbation(s)")
            break
    else:
        print(f"  {RED}WARNING: Could not achieve full controllability rank after "
              f"100 iterations.{RESET}")

    rank_final, sv_final, _ = _ctrl_rank_and_sv(A, B_aug, n)
    color = GREEN if rank_final == n else RED
    print(f"  {color}Final controllability rank: {rank_final}/{n}{RESET}")
    print(f"  Final SV: {_fmt(sv_final)}")

    return B_aug.astype(np.float32)


def compute_analytical_B(model, trajectories, cfg):
    """Compute B analytically from trajectory data.

    Args:
        model: KoopmanAutoencoder with trained A matrix
        trajectories: list of (koopman_states, perturbations)
        cfg: config dict

    Returns:
        B_final: (latent_dim, action_dim) numpy array
    """
    device = next(model.parameters()).device
    n = cfg["latent_dim"]
    m = cfg["action_dim"]

    # Collect all z_t, z_{t+1}, u_t from trajectories
    all_residuals = []
    all_actions = []
    all_z = []

    model.eval()
    with torch.no_grad():
        A = model.A.detach()
        for states, actions in trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)

            # Encode all states
            z_all = model.encode(states_t)  # (T, latent_dim)
            all_z.append(z_all.cpu().numpy())

            # z_t and z_{t+1}
            z_t = z_all[:-1]       # (T-1, latent_dim)
            z_next = z_all[1:]     # (T-1, latent_dim)
            u_t = actions_t[:len(z_t)]  # (T-1, action_dim)

            # Residual: what B*u needs to explain
            # z_{t+1} = A @ z_t + B @ u_t  =>  R = z_{t+1} - A @ z_t
            z_pred_autonomous = z_t @ A.T  # (T-1, latent_dim)
            residual = z_next - z_pred_autonomous  # (T-1, latent_dim)

            all_residuals.append(residual.cpu().numpy())
            all_actions.append(u_t.cpu().numpy())

    all_z_cat = np.concatenate(all_z, axis=0)  # (total_states, latent_dim)
    z_mean = all_z_cat.mean(axis=0)
    z_std = all_z_cat.std(axis=0)
    print(f"Latent z statistics ({all_z_cat.shape[0]} states):")
    print(f"  mean: {_fmt(z_mean)}")
    print(f"  std:  {_fmt(z_std)}")

    R = np.concatenate(all_residuals, axis=0).T  # (latent_dim, N)
    U = np.concatenate(all_actions, axis=0).T     # (action_dim, N)
    print(f"Data matrix shapes: R={R.shape}, U={U.shape}")
    print(f"  R norm: {np.linalg.norm(R):.4f}, U norm: {np.linalg.norm(U):.4f}")

    A_np = A.cpu().numpy()

    # Step 1: Solve unconstrained least-squares  R = B @ U  =>  B = R @ pinv(U)
    B_ls = R @ np.linalg.pinv(U)  # (latent_dim, action_dim)
    print(f"\nStep 1 - Least-squares B:")
    print(f"  B_ls shape: {B_ls.shape}")
    print(f"  B_ls spectral norm: {np.linalg.norm(B_ls, ord=2):.3f}")
    print(f"  Reconstruction error: {np.linalg.norm(R - B_ls @ U):.3f}")

    # Step 2: Project onto maximally controllable subspace
    print(f"\nStep 2 - Controllability projection:")
    B_proj = project_for_controllability(A_np, B_ls, n)

    # Step 3: Normalize to unit spectral norm
    B_scale = np.linalg.norm(B_proj, ord=2)
    print(f"\nStep 3 - Normalize to unit spectral norm:")
    print(f"  B_proj spectral norm before: {B_scale:.3f}")
    B_proj = B_proj / B_scale
    print(f"  Scaled to: {np.linalg.norm(B_proj, ord=2):.3f}")

    # Step 4: Final verification
    rank, sv, _ = _ctrl_rank_and_sv(A_np.astype(np.float64), B_proj.astype(np.float64), n)
    print(f"\nStep 4 - Final verification:")
    color = GREEN if rank == n else RED
    print(f"  {color}Controllability rank: {rank}/{n}{RESET}")
    print(f"  B spectral norm: {np.linalg.norm(B_proj, ord=2):.3f}")
    print(f"  B Frobenius norm: {np.linalg.norm(B_proj, 'fro'):.3f}")
    print(f"  Controllability singular values: {_fmt(sv)}")

    return B_proj, B_ls


def make_prediction_heatmap(model, B_final, aug_trajectories, raw_trajectories,
                            cfg, run_dir, eval_horizon=25):
    """Heatmap 1: prediction error vs step forward (x) and pendulum angle (y).

    Uses the analytical B with model's A for multi-step Koopman prediction,
    same layout as eval_pendulum.evaluate_model.
    """
    device = next(model.parameters()).device
    model.eval()
    train_horizon = cfg.get("horizon", 5)

    B_t = torch.tensor(B_final, dtype=torch.float32, device=device)

    true_angles_all = []
    errors_all = []

    with torch.no_grad():
        A = model.A.detach()
        for (states_norm, actions_norm), (states_raw, _, _) in zip(
                aug_trajectories, raw_trajectories):
            states_t = torch.tensor(states_norm, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions_norm, dtype=torch.float32, device=device)

            # Multi-step prediction using analytical B
            z = model.encode(states_t[0:1])
            T = min(eval_horizon, len(actions_norm))
            for t in range(T):
                u = actions_t[t:t+1]  # (1, action_dim)
                z = z @ A.T + u @ B_t.T
                x_pred = model.decode(z).cpu().numpy()[0]

                # True angle from raw (unnormalized) states
                true_angle = obs_to_angle(states_raw[t + 1])
                pred_angle = obs_to_angle(x_pred)

                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

    # Bin into 2D histogram: x=step, y=true angle, value=mean error
    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    steps = np.arange(1, eval_horizon + 1)

    heatmap = np.full((len(angle_centers), eval_horizon), np.nan)
    true_angles_all = np.array(true_angles_all)
    errors_arr = np.array(errors_all)

    for t in range(eval_horizon):
        mask = errors_arr[:, 0] == (t + 1)
        angles_t = true_angles_all[mask]
        errs_t = errors_arr[mask, 1]
        bin_idx = np.digitize(angles_t, angle_bins) - 1
        for b in range(len(angle_centers)):
            in_bin = errs_t[bin_idx == b]
            if len(in_bin) > 0:
                heatmap[b, t] = np.mean(in_bin)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_deg = np.degrees(heatmap)
    im = ax.pcolormesh(
        steps, np.degrees(angle_centers), heatmap_deg,
        cmap=cmap, shading="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Angle Error (degrees)")

    no_data = np.argwhere(np.isnan(heatmap))
    for b, t in no_data:
        ax.text(steps[t], np.degrees(angle_centers[b]), "x",
                ha="center", va="center", color="white", fontsize=6, alpha=0.7)
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("True Pendulum Angle (degrees)")
    ax.set_title("Koopman Prediction Error (Analytical B) vs Angle & Horizon")

    vmin = float(np.nanmin(heatmap_deg))
    vmax = float(np.nanmax(heatmap_deg))

    path = os.path.join(run_dir, "prediction_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Prediction heatmap saved to {path}")

    return vmin, vmax


def make_prediction_heatmap_a_only(model, aug_trajectories, raw_trajectories,
                                   cfg, run_dir, eval_horizon=25,
                                   vmin=None, vmax=None,
                                   filename="A_only_prediction_heatmap.png"):
    """Prediction heatmap using only A (no B), with perturbations still applied to env.

    Same layout as prediction_heatmap but multi-step prediction uses z_{t+1} = A z_t
    (ignoring control input). Shows how A alone handles perturbed trajectories.
    """
    device = next(model.parameters()).device
    model.eval()
    train_horizon = cfg.get("horizon", 5)

    true_angles_all = []
    errors_all = []

    with torch.no_grad():
        A = model.A.detach()
        for (states_norm, actions_norm), (states_raw, _, _) in zip(
                aug_trajectories, raw_trajectories):
            states_t = torch.tensor(states_norm, dtype=torch.float32, device=device)

            z = model.encode(states_t[0:1])
            T = min(eval_horizon, len(actions_norm))
            for t in range(T):
                z = z @ A.T  # A-only, no B term
                x_pred = model.decode(z).cpu().numpy()[0]

                true_angle = obs_to_angle(states_raw[t + 1])
                pred_angle = obs_to_angle(x_pred)

                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    steps = np.arange(1, eval_horizon + 1)

    heatmap = np.full((len(angle_centers), eval_horizon), np.nan)
    true_angles_all = np.array(true_angles_all)
    errors_arr = np.array(errors_all)

    for t in range(eval_horizon):
        mask = errors_arr[:, 0] == (t + 1)
        angles_t = true_angles_all[mask]
        errs_t = errors_arr[mask, 1]
        bin_idx = np.digitize(angles_t, angle_bins) - 1
        for b in range(len(angle_centers)):
            in_bin = errs_t[bin_idx == b]
            if len(in_bin) > 0:
                heatmap[b, t] = np.mean(in_bin)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_deg = np.degrees(heatmap)
    im = ax.pcolormesh(
        steps, np.degrees(angle_centers), heatmap_deg,
        cmap=cmap, shading="nearest",
        vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Angle Error (degrees)")

    no_data = np.argwhere(np.isnan(heatmap))
    for b, t in no_data:
        ax.text(steps[t], np.degrees(angle_centers[b]), "x",
                ha="center", va="center", color="white", fontsize=6, alpha=0.7)
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("True Pendulum Angle (degrees)")
    ax.set_title("A-only Prediction Error (no B) vs Angle & Horizon")

    path = os.path.join(run_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"A-only prediction heatmap saved to {path}")


def make_latent_recon_heatmap(model, B_final, aug_trajectories, raw_trajectories,
                              cfg, run_dir):
    """Heatmap 2: latent reconstruction error vs perturbation u (x) and angle (y).

    Each training data point is binned by its raw (unnormalized) perturbation
    value and current pendulum angle. Color shows mean one-step latent
    prediction error ||z_{t+1} - (A z_t + B u_t)||.
    """
    device = next(model.parameters()).device
    model.eval()

    B_t = torch.tensor(B_final, dtype=torch.float32, device=device)

    all_angles = []
    all_perturbations = []
    all_errors = []

    with torch.no_grad():
        A = model.A.detach()
        for (states_norm, actions_norm), (states_raw, _, perturbations_raw) in zip(
                aug_trajectories, raw_trajectories):
            states_t = torch.tensor(states_norm, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions_norm, dtype=torch.float32, device=device)

            z_all = model.encode(states_t)  # (T, latent_dim)
            z_t = z_all[:-1]
            z_next = z_all[1:]
            u_t = actions_t[:len(z_t)]

            # One-step prediction with analytical B
            z_pred = z_t @ A.T + u_t @ B_t.T
            errs = torch.linalg.norm(z_next - z_pred, dim=-1).cpu().numpy()  # (T-1,)

            # Angles from raw states (use states at time t)
            angles = np.array([obs_to_angle(s) for s in states_raw[:-2]])

            # Raw perturbation values (first column for 1-D action)
            perturbs = perturbations_raw[:len(angles), 0]

            all_angles.append(angles)
            all_perturbations.append(perturbs)
            all_errors.append(errs[:len(angles)])

    all_angles = np.concatenate(all_angles)
    all_perturbations = np.concatenate(all_perturbations)
    all_errors = np.concatenate(all_errors)

    # Bin: x = perturbation, y = angle
    u_min, u_max = all_perturbations.min(), all_perturbations.max()
    u_bins = np.linspace(u_min, u_max, 41)
    u_centers = 0.5 * (u_bins[:-1] + u_bins[1:])

    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])

    heatmap = np.full((len(angle_centers), len(u_centers)), np.nan)
    u_idx = np.digitize(all_perturbations, u_bins) - 1
    a_idx = np.digitize(all_angles, angle_bins) - 1

    for ai in range(len(angle_centers)):
        for ui in range(len(u_centers)):
            mask = (a_idx == ai) & (u_idx == ui)
            if mask.sum() > 0:
                heatmap[ai, ui] = np.mean(all_errors[mask])

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        u_centers, np.degrees(angle_centers), heatmap,
        cmap=cmap, shading="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Latent Reconstruction Error")

    no_data = np.argwhere(np.isnan(heatmap))
    for b, u in no_data:
        ax.text(u_centers[u], np.degrees(angle_centers[b]), "x",
                ha="center", va="center", color="white", fontsize=4, alpha=0.5)

    ax.set_xlabel("Control Perturbation (u)")
    ax.set_ylabel("Pendulum Angle (degrees)")
    ax.set_title("Latent Reconstruction Error vs Perturbation & Angle")

    vmin = float(np.nanmin(heatmap))
    vmax = float(np.nanmax(heatmap))

    path = os.path.join(run_dir, "latent_recon_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Latent reconstruction heatmap saved to {path}")

    return vmin, vmax


def make_a_only_heatmap(model, aug_trajectories, raw_trajectories, cfg, run_dir,
                        vmin=None, vmax=None, filename="A_with_pert_heatmap.png"):
    """Heatmap 3: autonomous (A-only) latent error vs perturbation u (x) and angle (y).

    Same layout as latent_recon_heatmap but prediction uses only A (no B),
    showing how much the perturbations degrade the autonomous prediction.
    Error is ||z_{t+1} - A z_t||.
    """
    device = next(model.parameters()).device
    model.eval()

    all_angles = []
    all_perturbations = []
    all_errors = []

    with torch.no_grad():
        A = model.A.detach()
        for (states_norm, _), (states_raw, _, perturbations_raw) in zip(
                aug_trajectories, raw_trajectories):
            states_t = torch.tensor(states_norm, dtype=torch.float32, device=device)

            z_all = model.encode(states_t)
            z_t = z_all[:-1]
            z_next = z_all[1:]

            # A-only prediction (no B term)
            z_pred = z_t @ A.T
            errs = torch.linalg.norm(z_next - z_pred, dim=-1).cpu().numpy()

            angles = np.array([obs_to_angle(s) for s in states_raw[:-2]])
            perturbs = perturbations_raw[:len(angles), 0]

            all_angles.append(angles)
            all_perturbations.append(perturbs)
            all_errors.append(errs[:len(angles)])

    all_angles = np.concatenate(all_angles)
    all_perturbations = np.concatenate(all_perturbations)
    all_errors = np.concatenate(all_errors)

    u_min, u_max = all_perturbations.min(), all_perturbations.max()
    u_bins = np.linspace(u_min, u_max, 41)
    u_centers = 0.5 * (u_bins[:-1] + u_bins[1:])

    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])

    heatmap = np.full((len(angle_centers), len(u_centers)), np.nan)
    u_idx = np.digitize(all_perturbations, u_bins) - 1
    a_idx = np.digitize(all_angles, angle_bins) - 1

    for ai in range(len(angle_centers)):
        for ui in range(len(u_centers)):
            mask = (a_idx == ai) & (u_idx == ui)
            if mask.sum() > 0:
                heatmap[ai, ui] = np.mean(all_errors[mask])

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        u_centers, np.degrees(angle_centers), heatmap,
        cmap=cmap, shading="nearest",
        vmin=vmin, vmax=vmax,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Latent Error (A only, no B)")

    no_data = np.argwhere(np.isnan(heatmap))
    for b, u in no_data:
        ax.text(u_centers[u], np.degrees(angle_centers[b]), "x",
                ha="center", va="center", color="white", fontsize=4, alpha=0.5)

    ax.set_xlabel("Control Perturbation (u)")
    ax.set_ylabel("Pendulum Angle (degrees)")
    ax.set_title("A-only Latent Error with Perturbations (no B correction)")

    path = os.path.join(run_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"A-only perturbation heatmap saved to {path}")


def run_analytical_b(model, env, policy, cfg, run_dir, augment=True,
                     obs_scale=None, num_trajectories=None):
    """Analytically derive B matrix for a pre-trained Koopman model.

    Collects perturbed trajectory data, solves for B via least-squares with
    controllability projection, generates heatmaps, and saves results.

    Args:
        model: KoopmanAutoencoder with trained A matrix (on device)
        env: single (non-vectorized) gym environment for data collection
        policy: base policy callable
        cfg: config dict
        run_dir: output directory
        augment: whether to augment states with base action
        obs_scale: observation scaling vector (required if augment=True)
        num_trajectories: override for number of trajectories (default: from config)

    Returns:
        B_final: (latent_dim, action_dim) numpy array
    """
    # Print A matrix properties
    A = model.A.detach().cpu().numpy()
    eigvals = np.linalg.eigvals(A)
    print(f"\nA matrix properties:")
    print(f"  Spectral radius: {np.max(np.abs(eigvals)):.3f}")
    print(f"  Eigenvalue magnitudes: {_fmt(np.sort(np.abs(eigvals))[::-1])}")

    # Collect perturbed data
    num_traj = num_trajectories or cfg["num_trajectories"]
    print(f"\nCollecting {num_traj} perturbed trajectories...")
    trajectories = collect_perturbed_data(
        env, policy, num_traj,
        cfg["max_episode_steps"], cfg["seed"],
        perturb_scale=cfg.get("perturb_scale", None),
        fix_perturb_range=cfg.get("fix_perturb_range", False),
        hold_steps=cfg.get("hold_steps", 1),
    )
    aug_trajectories = augment_perturbed_trajectories(
        trajectories, augment=augment, obs_scale=obs_scale)

    # Compute analytical B
    print(f"\n{'='*60}")
    print(f"Computing analytical B matrix")
    print(f"{'='*60}")
    B_final, B_ls = compute_analytical_B(model, aug_trajectories, cfg)

    # Generate heatmaps (use B_ls for prediction/recon — unnormalized, data-scale B)
    print(f"\nGenerating heatmaps...")
    train_horizon = cfg.get("horizon", 5)
    pred_vmin, pred_vmax = make_prediction_heatmap(
        model, B_ls, aug_trajectories, trajectories, cfg, run_dir,
        eval_horizon=train_horizon)
    make_prediction_heatmap_a_only(
        model, aug_trajectories, trajectories, cfg, run_dir,
        eval_horizon=train_horizon)
    make_prediction_heatmap_a_only(
        model, aug_trajectories, trajectories, cfg, run_dir,
        eval_horizon=train_horizon,
        vmin=pred_vmin, vmax=pred_vmax,
        filename="A_only_prediction_heatmap-scaled.png")
    recon_vmin, recon_vmax = make_latent_recon_heatmap(
        model, B_ls, aug_trajectories, trajectories, cfg, run_dir)
    make_a_only_heatmap(model, aug_trajectories, trajectories, cfg, run_dir)
    make_a_only_heatmap(model, aug_trajectories, trajectories, cfg, run_dir,
                        vmin=recon_vmin, vmax=recon_vmax,
                        filename="A_with_pert_heatmap-scaled.png")

    # Save results
    A_f64 = A.astype(np.float64)
    B_f64 = B_final.astype(np.float64)
    rank, sv, _ = _ctrl_rank_and_sv(A_f64, B_f64, cfg["latent_dim"])

    results = {
        "B_matrix": B_final.tolist(),
        "B_spectral_norm": float(np.linalg.norm(B_final, ord=2)),
        "B_frobenius_norm": float(np.linalg.norm(B_final, "fro")),
        "num_trajectories": num_traj,
        "controllability_rank": int(rank),
        "controllability_singular_values": sv.tolist(),
    }

    results_path = os.path.join(run_dir, "analytical_b_results.yaml")
    with open(results_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)
    print(f"\nResults saved to {results_path}")

    # Also save B as a .npy file for easy loading
    b_path = os.path.join(run_dir, "B_analytical.npy")
    np.save(b_path, B_final)
    print(f"B matrix saved to {b_path}")

    # Inject analytical B into model and save as koopman_ckpt.pt
    B_tensor = torch.tensor(B_final, dtype=torch.float32, device=next(model.parameters()).device)
    with torch.no_grad():
        if model.b_from_k_mod:
            # For normalized k_type: B = Q @ b_eigen, so b_eigen = Q^T @ B
            Q = model.K_module.Q
            model.K_module.b_eigen.copy_(Q.T @ B_tensor)
        else:
            model.B.weight.copy_(B_tensor)
    ckpt_path = os.path.join(run_dir, "koopman_ckpt.pt")
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

    # Save updated config
    save_config(cfg, run_dir)

    color = GREEN if rank == cfg["latent_dim"] else RED
    print(f"\n{color}=== Done. Controllability rank: {rank}/{cfg['latent_dim']}. "
          f"B norm: {np.linalg.norm(B_final, ord=2):.3f}. "
          f"All outputs in {run_dir} ==={RESET}")

    return B_final


def main():
    parser = argparse.ArgumentParser(
        description="Analytically compute B matrix for a pre-trained Koopman model")
    parser.add_argument("model_dir", type=str,
                        help="Path to model output directory (contains config.yaml and koopman_ckpt.pt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file to load (default: koopman_ckpt.pt or koop_a_checkpoint.pt)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (overrides default location)")
    parser.add_argument("--num-trajectories", type=int, default=None,
                        help="Number of trajectories to collect (default: from config)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(args.model_dir, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    augment = cfg.get("augment_state", False)

    # Determine checkpoint path
    if args.checkpoint:
        weights_path = args.checkpoint
    else:
        # Prefer koop_a_checkpoint.pt (A-only) if available, else koopman_ckpt.pt
        a_path = os.path.join(args.model_dir, "koop_a_checkpoint.pt")
        ab_path = os.path.join(args.model_dir, "koopman_ckpt.pt")
        if os.path.exists(a_path):
            weights_path = a_path
        else:
            weights_path = ab_path

    # Output directory
    if args.output_dir:
        run_dir = args.output_dir
    else:
        folder_name = f"anal_b_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        run_dir = os.path.join(args.model_dir, "tuned", folder_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Checkpoint: {weights_path}")
    print(f"Augment state: {augment}")

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

    state_dict = checkpoint["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {weights_path}")
    print(f"Koopman model: state_dim={koopman_state_dim}, action_dim={cfg['action_dim']}, "
          f"latent_dim={cfg['latent_dim']}, k_type={cfg['k_type']}")

    # Run analytical B computation
    B_final = run_analytical_b(model, env, policy, cfg, run_dir,
                               augment=augment, obs_scale=obs_scale,
                               num_trajectories=args.num_trajectories)

    env.close()


if __name__ == "__main__":
    main()
