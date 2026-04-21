"""Shared stability analysis utilities for Koopman pipeline."""
import math
import warnings

import numpy as np
import torch

from controllers.lqr import LQR
from model.utils import compute_encoder_lipschitz, optimize_lyapunov_P


def control_analysis(A, B_mat):
    """Compute controllability rank and report unstable modes.

    Args:
        A: (n, n) torch tensor
        B_mat: (n, m) torch tensor

    Returns:
        ctrl_rank: int
    """
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

    return ctrl_rank


def compute_encoder_lipschitz_bounds(model, aug_trajectories, device):
    """Compute encoder Lipschitz constants for both g(x) and full encode.

    When prepend_state=True, computes bounds for both the raw encoder g(x)
    and the full encode [x; g(x)]. Otherwise they are identical.
    Skipped when encoder is fixed (e.g. trig mode).

    Returns:
        m_gx: float or None, lower Lipschitz of g(x) network
        L_gx: float or None, upper Lipschitz of g(x) network
        m_full: float or None, lower Lipschitz of full encode
        L_full: float or None, upper Lipschitz of full encode
    """
    # Skip if encoder is not a learned network (e.g. trig)
    if getattr(model, '_trig_encoder', False) or model.encoder is None:
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


def setup_lqr(A, B_mat, cfg):
    """Set up and solve LQR with optional B scaling.

    Args:
        A: (n, n) torch tensor
        B_mat: (n, m) torch tensor
        cfg: config dict with q_scale, r_scale, scale_B, etc.

    Returns:
        lqr: LQR object
        Q: (n, n) torch tensor (state cost)
        R_cost: (m, m) torch tensor (control cost)
        B_scale: float (1.0 if no scaling)
    """
    actual_latent = A.shape[0]  # matches A dimension (may include prepended state)
    action_dim = cfg["action_dim"]
    q_scale = cfg.get("q_scale", 1.0)
    r_scale = cfg["r_scale"]

    # Q penalizes real state dims (θ, θ̇) at q_scale, other dims at q_epsilon_scale
    real_state_dim = cfg["state_dim"]
    q_eps = cfg.get("q_epsilon_scale", 0.0)
    Q = torch.eye(actual_latent) * q_eps
    Q[:real_state_dim, :real_state_dim] = torch.eye(real_state_dim) * q_scale
    R_cost = torch.eye(action_dim) * r_scale

    scale_B = cfg.get("scale_B", False)
    if scale_B:
        print("  Scaling B")
        B_scale = torch.linalg.norm(B_mat, ord=2)
        B_for_lqr = B_mat / B_scale
    else:
        B_scale = 1.0
        B_for_lqr = B_mat

    lqr = LQR(A, B_for_lqr, Q, R_cost, q_scale=q_scale,
              controllable_subspace=cfg.get("controllable_subspace", False),
              ctrl_threshold=cfg.get("ctrl_threshold", None))

    return lqr, Q, R_cost, B_scale


def compute_lyapunov_params(lqr, Q, R_cost):
    """Compute Lyapunov stability parameters from LQR solution.

    Args:
        lqr: LQR object with P, F attributes
        Q: (n, n) state cost matrix
        R_cost: (m, m) control cost matrix

    Returns:
        P: Lyapunov matrix
        kappa_P: condition number of P
        rho_sq: Lyapunov contraction rate ρ²
        P_eigvals: eigenvalues of P
    """
    P = lqr.P
    F = lqr.F

    P_eigvals = torch.linalg.eigvalsh(P)
    kappa_P = (P_eigvals.max() / P_eigvals.min()).item()

    # ρ² = 1 - λ_min(Q + F^T R F) / λ_max(P)
    Q_plus_FRF = Q + F.T @ R_cost @ F
    Q_eigvals = torch.linalg.eigvalsh(Q_plus_FRF)
    rho_sq = 1.0 - Q_eigvals.min().item() / P_eigvals.max().item()

    return P, kappa_P, rho_sq, P_eigvals


def compute_BtPB(lqr, B_mat, P):
    """Compute B^T P B with optional controllable subspace projection.

    Returns:
        BtPB: float scalar
    """
    if lqr.V_ctrl is not None:
        B_for_P = lqr.V_ctrl @ B_mat
    else:
        B_for_P = B_mat
    return (B_for_P.T @ P @ B_for_P).item()


def compute_latent_errors(model, aug_trajectories, device, error_stats=None):
    """Compute one-step latent prediction error statistics.

    If error_stats is provided, reads from it. Otherwise recomputes.

    Args:
        model: KoopmanAutoencoder
        aug_trajectories: list of (states, actions)
        device: torch device
        error_stats: optional dict with mean_pred_error_latent, std_pred_error_latent

    Returns:
        err_mean: float
        err_std: float
    """
    if error_stats is not None:
        return error_stats["mean_pred_error_latent"], error_stats["std_pred_error_latent"]

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
    return all_latent_errs.mean().item(), all_latent_errs.std().item()


def compute_state_recon_errors(model, aug_trajectories, device):
    """Compute one-step state-space reconstruction error statistics.

    For each transition (x_t, u_t, x_{t+1}), computes:
        x_pred = decode(predict(encode(x_t), u_t))
        error  = ||x_{t+1} - x_pred||

    All computations are in normalized state space (same space as epsilon_x).

    Args:
        model: KoopmanAutoencoder
        aug_trajectories: list of (states, actions) where states are normalized
        device: torch device

    Returns:
        err_mean: float
        err_std: float
    """
    model.to(device)
    model.eval()
    all_state_errs = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_pred = model.predict(z_all, actions_t[:T_act])
            x_pred = model.decode(z_pred)
            x_next = states_t[1:T_act + 1]
            errs = torch.linalg.norm(x_next - x_pred, dim=-1)
            all_state_errs.append(errs.cpu())
    all_state_errs = torch.cat(all_state_errs)
    return all_state_errs.mean().item(), all_state_errs.std().item()


def count_steps_under_threshold(model, aug_trajectories, device, threshold, space="state"):
    """Count transitions with one-step prediction error below a threshold.

    Args:
        model: KoopmanAutoencoder
        aug_trajectories: list of (states, actions)
        device: torch device
        threshold: max error threshold
        space: "state" for ||x_{t+1} - decode(predict(encode(x_t), u_t))||,
               "latent" for ||z_{t+1} - predict(z_t, u_t)||

    Returns:
        count_under: int, number of steps under threshold
        total: int, total number of steps
        fraction: float, count_under / total
    """
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
                z_next = model.encode(states_t[1:T_act + 1])
                errs = torch.linalg.norm(z_next - z_pred, dim=-1)
            else:
                x_pred = model.decode(z_pred)
                x_next = states_t[1:T_act + 1]
                errs = torch.linalg.norm(x_next - x_pred, dim=-1)
            all_errs.append(errs.cpu())
    all_errs = torch.cat(all_errs)
    count_under = int((all_errs < threshold).sum().item())
    total = len(all_errs)
    return count_under, total, count_under / total


def compute_max_latent_diff(model, cfg, device):
    """Compute max latent space difference between extreme states.

    Uses (θ=0, θ̇=-8) and (θ=π, θ̇=8) as extremes.

    Returns:
        max_latent_diff: float
    """
    with torch.no_grad():
        obs_type = cfg.get("obs_type", "cos_sin")
        augment = cfg.get("augment_state", False)
        if obs_type == "cos_sin":
            x_origin = [1.0, 0.0, -8.0]
            x_extreme = [-1.0, 0.0, 8.0]
        else:
            x_origin = [0.0, -8.0]
            x_extreme = [np.pi, 8.0]
        if augment:
            action_dim = cfg["action_dim"]
            x_origin = x_origin + [0.0] * action_dim
            x_extreme = x_extreme + [0.0] * action_dim
        x_origin = torch.tensor([x_origin], dtype=torch.float32, device=device)
        x_extreme = torch.tensor([x_extreme], dtype=torch.float32, device=device)
        z_origin = model.encode(x_origin)
        z_extreme = model.encode(x_extreme)
        return torch.linalg.norm(z_extreme - z_origin).item()


def run_sdp_optimization(lqr, epsilon, eta, cfg):
    """Run SDP optimization of Lyapunov P if enabled in config.

    Args:
        lqr: LQR object (for closed_loop matrix)
        epsilon: tracking error bound for optimization objective
        eta: disturbance bound
        cfg: config dict (checks optimize_lyapunov_P)

    Returns:
        (rho_sq, kappa_P, sdp_gamma) if optimization succeeded
        None if disabled or failed
    """
    if not cfg.get("optimize_lyapunov_P", True):
        return None

    A_cl_np = lqr.closed_loop.numpy()
    sdp_P, sdp_rho, sdp_kappa, sdp_gamma = optimize_lyapunov_P(
        A_cl_np, epsilon, eta)

    if sdp_P is None:
        print(f"\033[91m  SDP optimization failed, using LQR values\033[0m")
        return None

    return sdp_rho ** 2, sdp_kappa, sdp_gamma


def lyapunov_gamma(epsilon, rho_sq, kappa_P, eta):
    """Compute Lyapunov γ_max = ε * (1-ρ) / √κ(P) - η."""
    return epsilon * (1.0 - math.sqrt(rho_sq)) / math.sqrt(kappa_P) - eta


def alpha_bound(model, lqr, cfg, aug_trajectories, env, error_stats=None):
    """Alpha bound stability analysis using [x; g(x)] latent space.

    Uses C = [I_p, 0_{p×q}] extraction matrix and optimizes P via
    Cholesky parameterization to maximize γ.

    γ = ε_x * (1-ρ) / √(α * λ_max(P)) - η
    where α = λ_max(P⁻¹ CᵀC)

    Args:
        model: KoopmanAutoencoder (must have prepend_state=True)
        lqr: LQR object (for closed_loop and DARE P initialization)
        cfg: config dict with alpha_epsilon_x, alpha_eta
        aug_trajectories: trajectory data (for latent error stats)
        env: gym environment (for u_max)
        error_stats: optional pre-computed error stats

    Returns:
        variables: dict of computed quantities
    """
    from model.utils import optimize_alpha_P

    print("\n" + "=" * 60)
    print("  Alpha Bound Stability Analysis")
    print("=" * 60)

    device = next(model.parameters()).device
    model.eval()

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()

    # Dimensions
    if not model.prepend_state:
        raise ValueError("Alpha bound requires prepend_state=True")
    n = A.shape[0]  # full latent dim
    real_state_dim = cfg["state_dim"]  # real state dim (without action augmentation)
    q = n - model.prepend_dim  # encoder output dim
    p = real_state_dim  # C extracts real state dims only

    # Construct C = [I_p | 0_{p×(n-p)}] — extracts real state from full latent
    C = np.zeros((p, n), dtype=np.float64)
    C[:p, :p] = np.eye(p)

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Matrix dimensions
    print(f"  A: {A.shape[0]}x{A.shape[1]}")
    print(f"  B: {B_mat.shape[0]}x{B_mat.shape[1]}")
    print(f"  C: {C.shape[0]}x{C.shape[1]}")

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
    epsilon_x = cfg["alpha_epsilon_x"]
    eta = cfg["alpha_eta"]

    print(f"  Q scale:                               {q_scale}")
    print(f"  R scale:                               {r_scale}")
    print(f"  u_max:                                 {u_max:.6f}")
    print(f"  ε_x (from config):                     {epsilon_x:.6f}")
    print(f"  η (from config):                       {eta:.6f}")
    print(f"  C matrix: ({p}, {n}) = [I_{p} | 0_{{{p}×{q}}}]")

    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(model, aug_trajectories, device)
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

    err_mean_latent, err_std_latent = compute_latent_errors(
        model, aug_trajectories, device, error_stats)
    R_latent = err_mean_latent + 2 * err_std_latent

    print(f"  One-step latent error mean:             {err_mean_latent:.6f}")
    print(f"  One-step latent error std:              {err_std_latent:.6f}")
    print(f"  R_latent (mean + 2σ):                   {R_latent:.6f}")
    print(f"  R_latent / max_latent_diff:             {R_latent / max_latent_diff:.6f}")

    # =====================================================================
    #  One-Step State Reconstruction Error
    # =====================================================================
    print("  " + "-" * 48)

    err_mean_state, err_std_state = compute_state_recon_errors(
        model, aug_trajectories, device)
    R_state = err_mean_state + 2 * err_std_state

    print(f"  One-step state recon error mean:        {err_mean_state:.6f}")
    print(f"  One-step state recon error std:         {err_std_state:.6f}")
    print(f"  R_state (mean + 2σ):                    {R_state:.6f}")

    # Select R based on config
    alpha_r_space = cfg.get("alpha_r_space", "state")
    if alpha_r_space == "latent":
        R_val = R_latent
        print(f"  R = R_latent (alpha_r_space='latent'):  {R_val:.6f}")
    else:
        R_val = R_state
        print(f"  R = R_state (alpha_r_space='state'):    {R_val:.6f}")

    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, R_val, space=alpha_r_space)
    print(f"  Steps under R:                         {count_under}/{total} ({fraction*100:.1f}%)")

    # # Δt distribution: ||z_{t+1}^ref - A * z_t^ref||
    # print("  " + "-" * 48)
    # print("  Δt distribution: ||z_{t+1} - A z_t||")
    # model.to(device)
    # model.eval()
    # all_delta_norms = []
    # with torch.no_grad():
    #     A_mat = model.A.detach()
    #     for states, actions in aug_trajectories:
    #         states_t = torch.tensor(states, dtype=torch.float32, device=device)
    #         z_all = model.encode(states_t)
    #         z_t = z_all[:-1]
    #         z_next = z_all[1:]
    #         z_pred_auto = z_t @ A_mat.T
    #         deltas = torch.linalg.norm(z_next - z_pred_auto, dim=-1)
    #         all_delta_norms.append(deltas.cpu())
    # all_delta_norms = torch.cat(all_delta_norms)
    # print(f"    min:        {all_delta_norms.min().item():.6f}")
    # print(f"    max:        {all_delta_norms.max().item():.6f}")
    # print(f"    median:     {all_delta_norms.median().item():.6f}")
    # print(f"    mean:       {all_delta_norms.mean().item():.6f}")
    # print(f"    std:        {all_delta_norms.std().item():.6f}")
    # dt_mean_2sigma = all_delta_norms.mean().item() + 2 * all_delta_norms.std().item()
    # print(f"    mean+2σ:    {dt_mean_2sigma:.6f}")

    # =====================================================================
    #  LQR Stability
    # =====================================================================
    print("\n" + "=" * 50)
    print("  LQR Stability")
    print("=" * 50)

    gain_norm = lqr.gain_norm.item()
    P_lqr = lqr.P
    F_lqr = lqr.F
    A_cl = lqr.closed_loop

    print(f"  A_cl: {A_cl.shape[0]}x{A_cl.shape[1]}")
    print(f"  F:  {F_lqr.shape[0]}x{F_lqr.shape[1]}")
    print(f"  P:  {P_lqr.shape[0]}x{P_lqr.shape[1]}")

    P_lqr_eigs = torch.linalg.eigvalsh(P_lqr)
    lam_min_P_lqr = P_lqr_eigs.min().item()
    lam_max_P_lqr = P_lqr_eigs.max().item()
    kappa_P_lqr = lam_max_P_lqr / lam_min_P_lqr
    BtPB = compute_BtPB(lqr, B_mat, P_lqr)

    # Compute LQR rho_sq for reference (Q only on real state dims)
    q_eps_local = cfg.get("q_epsilon_scale", 0.0)
    Q_lqr = torch.eye(A.shape[0]) * q_eps_local
    Q_lqr[:real_state_dim, :real_state_dim] = torch.eye(real_state_dim) * cfg.get("q_scale", 1.0)
    R_lqr_cost = torch.eye(cfg["action_dim"]) * cfg["r_scale"]
    Q_plus_FRF = Q_lqr + F_lqr.T @ R_lqr_cost @ F_lqr
    rho_sq_lqr = 1.0 - torch.linalg.eigvalsh(Q_plus_FRF).min().item() / lam_max_P_lqr

    A_norm = torch.linalg.norm(A, ord=2).item()
    B_norm = torch.linalg.norm(B_mat, ord=2).item()
    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    print(f"  ||B|| (spectral norm):                 {B_norm:.6f}")
    print(f"  B^T P B (LQR):                         {BtPB:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
    print(f"  κ(P) (LQR):                            {kappa_P_lqr:.6f}")
    print(f"  λ_min(P) (LQR):                        {lam_min_P_lqr:.6f}")
    print(f"  λ_max(P) (LQR):                        {lam_max_P_lqr:.6f}")
    P_lqr_np = P_lqr.numpy().astype(np.float64)
    CtC = (C.T @ C).astype(np.float64)
    alpha_lqr = np.max(np.linalg.eigvalsh(np.linalg.solve(P_lqr_np, CtC)))
    print(f"  α (LQR) = λ_max(P⁻¹CᵀC):             {alpha_lqr:.6f}")
    import math
    rho_lqr = math.sqrt(max(rho_sq_lqr, 0.0))
    gamma_lqr_no_eta = epsilon_x * (1.0 - rho_lqr) / math.sqrt(alpha_lqr * lam_max_P_lqr)
    gamma_lqr_config = gamma_lqr_no_eta - eta
    max_displacement_x = cfg.get("max_displacement_x", 0.1)
    print(f"  γ_max (LQR, η=0):                      {gamma_lqr_no_eta:.6f}")
    print(f"  γ_max (LQR, config η):                 {gamma_lqr_config:.6f}")
    if L_full is not None:
        eta_computed_lqr = max_displacement_x * L_full * (1.0 + A_norm)
        gamma_lqr_computed = gamma_lqr_no_eta - eta_computed_lqr
        print(f"  γ_max (LQR, computed η):               {gamma_lqr_computed:.6f}")

    # =====================================================================
    #  Alpha Bound Optimization
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Alpha Bound P Optimization")
    print("=" * 50)

    A_cl = lqr.closed_loop.numpy().astype(np.float64)
    P_init = lqr.P.numpy().astype(np.float64)

    best_P, best_rho, best_alpha, lam_max_P, best_gamma = optimize_alpha_P(
        A_cl, C, epsilon_x, eta, P_init)

    if best_P is None:
        print(f"{RED}  Alpha bound optimization failed, using LQR P{RESET}")
        import math
        rho_lqr = math.sqrt(max(rho_sq_lqr, 0.0))
        gamma_lqr_alpha = epsilon_x * (1.0 - rho_lqr) / math.sqrt(alpha_lqr * lam_max_P_lqr) - eta
        print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
        print(f"  α (LQR):                               {alpha_lqr:.6f}")
        print(f"  λ_max(P) (LQR):                        {lam_max_P_lqr:.6f}")
        if gamma_lqr_alpha < 0:
            print(f"{RED}  γ_max (LQR, config η):                 {gamma_lqr_alpha:.6f}{RESET}")
        else:
            print(f"{GREEN}  γ_max (LQR, config η):                 {gamma_lqr_alpha:.6f}{RESET}")
        print(f"  γ_max / R:                             {gamma_lqr_alpha / R_val:.6f}" if R_val > 0 else "  γ_max / R:                             N/A")
        return {"gamma_max_alpha": float(gamma_lqr_alpha)}

    best_P_eigs = np.linalg.eigvalsh(best_P)
    lam_min_P = best_P_eigs.min()
    kappa_P = lam_max_P / lam_min_P

    print(f"  ρ² (optimized):                        {best_rho**2:.6f}")
    print(f"  α = λ_max(P⁻¹CᵀC):                    {best_alpha:.6f}")
    print(f"  λ_min(P):                              {lam_min_P:.6f}")
    print(f"  λ_max(P):                              {lam_max_P:.6f}")
    print(f"  κ(P) = λ_max(P)/λ_min(P):             {kappa_P:.6f}")

    # =====================================================================
    #  Gamma Bounds
    # =====================================================================
    print("\n" + "=" * 50)
    print("  Gamma Bounds")
    print("=" * 50)

    import math
    gamma_base = epsilon_x * (1.0 - best_rho) / math.sqrt(best_alpha * lam_max_P)

    # Control-free bound (η=0)
    gamma_no_eta = gamma_base
    if gamma_no_eta < 0:
        print(f"{RED}  γ_max (control-free, η=0):             {gamma_no_eta:.6f}{RESET}")
    else:
        print(f"{GREEN}  γ_max (control-free, η=0):             {gamma_no_eta:.6f}{RESET}")
    print(f"  γ_max / R:                             {gamma_no_eta / R_val:.6f}" if R_val > 0 else "  γ_max / R:                             N/A")
    c, t, f = count_steps_under_threshold(model, aug_trajectories, device, gamma_no_eta, space=alpha_r_space)
    print(f"  Steps under γ_max:                     {c}/{t} ({f*100:.1f}%)")
    print(f"  " + "-" * 48)

    # gamma with config eta
    gamma_config_eta = gamma_base - eta
    if gamma_config_eta < 0:
        print(f"{RED}  γ_max (config η={eta:.4f}):              {gamma_config_eta:.6f}{RESET}")
    else:
        print(f"{GREEN}  γ_max (config η={eta:.4f}):              {gamma_config_eta:.6f}{RESET}")
    print(f"  γ_max / R:                             {gamma_config_eta / R_val:.6f}" if R_val > 0 else "  γ_max / R:                             N/A")
    c, t, f = count_steps_under_threshold(model, aug_trajectories, device, gamma_config_eta, space=alpha_r_space)
    print(f"  Steps under γ_max:                     {c}/{t} ({f*100:.1f}%)")

    # gamma with computed eta = max_displacement_x * L * (1 + ||A||)
    max_displacement_x = cfg.get("max_displacement_x", 0.1)
    L_for_eta = L_full if L_full is not None else 1.0
    eta_computed = max_displacement_x * L_for_eta * (1.0 + A_norm)
    gamma_computed_eta = epsilon_x * (1.0 - best_rho) / math.sqrt(best_alpha * lam_max_P) - eta_computed
    print(f"  " + "-" * 48)
    print(f"  η_computed = Δx * L * (1+||A||):       {eta_computed:.6f}")
    print(f"    (Δx={max_displacement_x}, L={L_for_eta:.4f}, ||A||={A_norm:.4f})")
    if gamma_computed_eta < 0:
        print(f"{RED}  γ_max (computed η={eta_computed:.4f}):     {gamma_computed_eta:.6f}{RESET}")
    else:
        print(f"{GREEN}  γ_max (computed η={eta_computed:.4f}):     {gamma_computed_eta:.6f}{RESET}")
    print(f"  γ_max / R:                             {gamma_computed_eta / R_val:.6f}" if R_val > 0 else "  γ_max / R:                             N/A")
    c, t, f = count_steps_under_threshold(model, aug_trajectories, device, gamma_computed_eta, space=alpha_r_space)
    print(f"  Steps under γ_max:                     {c}/{t} ({f*100:.1f}%)")

    variables = {
        "alpha_epsilon_x": float(epsilon_x),
        "alpha_eta_config": float(eta),
        "alpha_eta_computed": float(eta_computed),
        "alpha": float(best_alpha),
        "rho_alpha": float(best_rho),
        "lambda_min_P": float(lam_min_P),
        "lambda_max_P": float(lam_max_P),
        "kappa_P": float(kappa_P),
        "gamma_max_alpha_config_eta": float(gamma_config_eta),
        "gamma_max_alpha_computed_eta": float(gamma_computed_eta),
        "ctrl_rank": int(ctrl_rank),
        "latent_error_mean": float(err_mean_latent),
        "latent_error_std": float(err_std_latent),
        "state_recon_error_mean": float(err_mean_state),
        "state_recon_error_std": float(err_std_state),
        "alpha_r_space": alpha_r_space,
        "m_gx": float(m_gx) if m_gx is not None else None,
        "L_gx": float(L_gx) if L_gx is not None else None,
        "m_full": float(m_full) if m_full is not None else None,
        "L_full": float(L_full) if L_full is not None else None,
        "max_latent_diff": float(max_latent_diff),
    }

    return variables
