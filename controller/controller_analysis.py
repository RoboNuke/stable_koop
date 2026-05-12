"""Controller-agnostic analysis utilities.

Anything that operates on the ``(A, B, encoder, decoder)`` tuple — and would be
duplicated across controller types — lives here. Concretely:

* matrix-level diagnostics (spectral radius, transient constant, controllability rank);
* encoder Lipschitz bounds (per-trajectory and aggregate);
* one-step prediction error statistics in state and latent space;
* error-conversion helpers used by every stability bound (state↔latent via ``m``).

Single source of truth — folded from the legacy ``model/utils.py`` (pure numerics)
and the generic half of ``launch/stability_utils.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch.func import jacrev, vmap


# ---------------------------------------------------------------------------
# Matrix-level diagnostics.
# ---------------------------------------------------------------------------


def spectral_radius(M):
    """Largest eigenvalue magnitude of ``M``."""
    eigvals = torch.linalg.eigvals(M)
    return eigvals.abs().max().item()


def transient_constant(M):
    """Transient constant ``C = ||V|| · ||V^{-1}||`` where ``V`` is M's eigenvector matrix."""
    eigenvalues, V = torch.linalg.eig(M)
    cond = torch.linalg.cond(V, p=2).item()
    if cond > 1e10:
        warnings.warn("V is poorly conditioned, M may not be diagonalizable")
    return cond


def control_analysis(A, B_mat):
    """Print controllability rank + project unstable modes onto B. Returns rank."""
    A_np = A.numpy()
    B_np = B_mat.numpy()
    C_mat = np.hstack(
        [np.linalg.matrix_power(A_np, i) @ B_np for i in range(A_np.shape[0])]
    )
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
    return ctrl_rank


# ---------------------------------------------------------------------------
# Encoder Lipschitz bounds.
# ---------------------------------------------------------------------------


def compute_encoder_lipschitz(encoder, training_data):
    """Lower / upper Lipschitz constants of ``encoder`` over ``training_data``."""
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])

    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)

    J_batch = vmap(jacrev(encode_single))(X)
    svdvals = torch.linalg.svdvals(J_batch)
    sigma_mins = svdvals[:, -1]
    sigma_maxs = svdvals[:, 0]
    m = float(sigma_mins.min().detach())
    L = float(sigma_maxs.max().detach())
    print(
        f"  σ_min distribution ({len(sigma_mins)} points): "
        f"min={sigma_mins.min():.6f}  p1={sigma_mins.quantile(0.01):.6f}  "
        f"p5={sigma_mins.quantile(0.05):.6f}  median={sigma_mins.median():.6f}  "
        f"mean={sigma_mins.mean():.6f}"
    )
    return m, L


def compute_lower_lipschitz(encoder, training_data):
    """Lower Lipschitz constant of ``encoder`` (min singular value of its Jacobian)."""
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])

    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)

    J_batch = vmap(jacrev(encode_single))(X)
    sigma_mins = torch.linalg.svdvals(J_batch)[:, -1]
    return float(sigma_mins.min().detach())


def compute_encoder_lipschitz_bounds(model, aug_trajectories, device):
    """Compute Lipschitz bounds for both ``g(x)`` and the full ``encode`` map."""
    if getattr(model, "_trig_encoder", False) or model.encoder is None:
        print("  Lipschitz bounds: skipped (fixed encoder)")
        return None, None, None, None

    model_cpu = model.cpu()
    training_states = []
    for states, actions in aug_trajectories:
        for s in states:
            training_states.append(s)

    if model_cpu.prepend_state:
        print("  --- g(x) encoder only ---")
        m_gx, L_gx = compute_encoder_lipschitz(model_cpu.encoder, training_states)
        print("  --- full encode [x; g(x)] ---")
        m_full, L_full = compute_encoder_lipschitz(model_cpu.encode, training_states)
    else:
        m_gx, L_gx = compute_encoder_lipschitz(model_cpu.encode, training_states)
        m_full, L_full = m_gx, L_gx

    model.to(device)
    return m_gx, L_gx, m_full, L_full


# ---------------------------------------------------------------------------
# Error conversions and tolerable-error bound (purely algebraic).
# ---------------------------------------------------------------------------


def max_tolerable_model_error(rho, C, epsilon_max, eta):
    """``ε_max · (1 − ρ) / C − η`` — the maximum tolerable per-step model error."""
    return (epsilon_max * (1 - rho) / C) - eta


def latent_error_to_state_error(latent_error, m):
    """Convert latent error to state error via lower-Lipschitz ``m``."""
    return latent_error / m


def state_error_to_latent_error(state_error, m):
    """Convert state error to latent error via lower-Lipschitz ``m``."""
    return state_error * m


# ---------------------------------------------------------------------------
# One-step prediction error statistics on collected trajectories.
# ---------------------------------------------------------------------------


def compute_latent_errors(model, aug_trajectories, device, error_stats=None):
    """Mean/std of one-step latent prediction error across ``aug_trajectories``."""
    if error_stats is not None:
        return error_stats["mean_pred_error_latent"], error_stats["std_pred_error_latent"]
    model.to(device)
    model.eval()
    all_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_next = model.encode(states_t[1 : T_act + 1])
            z_pred = model.predict(z_all, actions_t[:T_act])
            errs = torch.linalg.norm(z_next - z_pred, dim=-1)
            all_errs.append(errs.cpu())
    all_errs = torch.cat(all_errs)
    return all_errs.mean().item(), all_errs.std().item()


def compute_state_recon_errors(model, aug_trajectories, device):
    """Mean/std of one-step state-space reconstruction error across ``aug_trajectories``."""
    model.to(device)
    model.eval()
    all_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_pred = model.predict(z_all, actions_t[:T_act])
            x_pred = model.decode(z_pred)
            x_next = states_t[1 : T_act + 1]
            errs = torch.linalg.norm(x_next - x_pred, dim=-1)
            all_errs.append(errs.cpu())
    all_errs = torch.cat(all_errs)
    return all_errs.mean().item(), all_errs.std().item()


def count_steps_under_threshold(model, aug_trajectories, device, threshold, space="state"):
    """Count transitions with one-step prediction error below ``threshold``."""
    model.to(device)
    model.eval()
    all_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_pred = model.predict(z_all, actions_t[:T_act])
            if space == "latent":
                z_next = model.encode(states_t[1 : T_act + 1])
                errs = torch.linalg.norm(z_next - z_pred, dim=-1)
            else:
                x_pred = model.decode(z_pred)
                x_next = states_t[1 : T_act + 1]
                errs = torch.linalg.norm(x_next - x_pred, dim=-1)
            all_errs.append(errs.cpu())
    all_errs = torch.cat(all_errs)
    count_under = int((all_errs < threshold).sum().item())
    total = len(all_errs)
    return count_under, total, count_under / total


