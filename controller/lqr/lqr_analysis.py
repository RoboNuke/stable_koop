"""LQR-specific stability analysis — anything that uses ``F`` or ``P``.

Folded from the LQR-specific half of the legacy ``launch/stability_utils.py``
and the Lyapunov optimization helpers from ``model/utils.py``. Generic
analyses live in :mod:`controller.controller_analysis`.

The four ``*_bound_report`` orchestrators (eigen, Lyapunov, m-free, alpha)
each print their full diagnostic block and return a dict of computed
quantities. They take pre-computed common inputs (``A``, ``B``, the fitted
LQR, encoder Lipschitz constants, etc.) so callers can share work.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from controller.controller_analysis import (
    compute_latent_errors,
    compute_max_latent_diff_observed,
    count_steps_under_threshold,
    max_tolerable_model_error,
    spectral_radius,
    state_error_to_latent_error,
    transient_constant,
)
from controller.lqr.lqr import LQR


# ---------------------------------------------------------------------------
# LQR construction.
# ---------------------------------------------------------------------------


def setup_lqr(A, B_mat, cfg):
    """Build and solve the LQR controller from a flat ``cfg`` dict.

    Returns ``(lqr, Q, R_cost, B_scale)``. ``B_scale`` is 1.0 unless
    ``cfg["scale_B"]`` normalizes B's spectral norm before solving.
    """
    actual_latent = A.shape[0]
    action_dim = cfg["action_dim"]
    q_scale = cfg.get("q_scale", 1.0)
    r_scale = cfg["r_scale"]

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

    lqr = LQR(
        A,
        B_for_lqr,
        Q,
        R_cost,
        q_scale=q_scale,
        controllable_subspace=cfg.get("controllable_subspace", False),
        ctrl_threshold=cfg.get("ctrl_threshold", None),
    )
    return lqr, Q, R_cost, B_scale


# ---------------------------------------------------------------------------
# Lyapunov diagnostics derived from the LQR P.
# ---------------------------------------------------------------------------


def compute_lyapunov_params(lqr, Q, R_cost):
    """Return ``(P, κ(P), ρ², λ(P))`` from the LQR Lyapunov certificate."""
    P = lqr.P
    F = lqr.F
    P_eigvals = torch.linalg.eigvalsh(P)
    kappa_P = (P_eigvals.max() / P_eigvals.min()).item()
    Q_plus_FRF = Q + F.T @ R_cost @ F
    Q_eigvals = torch.linalg.eigvalsh(Q_plus_FRF)
    rho_sq = 1.0 - Q_eigvals.min().item() / P_eigvals.max().item()
    return P, kappa_P, rho_sq, P_eigvals


def compute_BtPB(lqr, B_mat, P):
    """``B^T P B`` (projected through the controllable subspace if used)."""
    if lqr.V_ctrl is not None:
        B_for_P = lqr.V_ctrl @ B_mat
    else:
        B_for_P = B_mat
    return (B_for_P.T @ P @ B_for_P).item()


def lyapunov_gamma(epsilon, rho_sq, kappa_P, eta):
    """``γ_max = ε · (1 − √ρ²) / √κ(P) − η``."""
    return epsilon * (1.0 - math.sqrt(rho_sq)) / math.sqrt(kappa_P) - eta


# ---------------------------------------------------------------------------
# SDP / scipy optimization of the Lyapunov P.
# ---------------------------------------------------------------------------


def optimize_lyapunov_P(A_cl_np, epsilon, eta, rho_grid_size=100):
    """SDP sweep over ρ to maximize ``γ = ε(1−ρ)/√κ − η``.

    Returns ``(P*, ρ*, κ*, γ*)`` or ``(None, None, None, -inf)`` on failure.
    """
    import cvxpy as cp

    n = A_cl_np.shape[0]
    A_cl = A_cl_np.astype(np.float64)

    best_gamma = -np.inf
    best_P = None
    best_rho = None
    best_kappa = None

    for rho in np.linspace(0.01, 0.99, rho_grid_size):
        rho_sq = rho ** 2
        P = cp.Variable((n, n), symmetric=True)
        kappa = cp.Variable()
        constraints = [
            A_cl.T @ P @ A_cl << rho_sq * P,
            P >> np.eye(n),
            P << kappa * np.eye(n),
        ]
        prob = cp.Problem(cp.Minimize(kappa), constraints)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=cp.SCS, verbose=False)
        except cp.SolverError:
            continue

        if prob.status in ("optimal", "optimal_inaccurate") and kappa.value is not None:
            kappa_val = float(kappa.value)
            if kappa_val <= 0:
                continue
            gamma = epsilon * (1.0 - rho) / math.sqrt(kappa_val) - eta
            if gamma > best_gamma:
                best_gamma = gamma
                best_P = P.value.copy()
                best_rho = rho
                best_kappa = kappa_val

    return best_P, best_rho, best_kappa, best_gamma


def optimize_alpha_P(A_cl_np, C, epsilon_x, eta, P_init):
    """L-BFGS-B over ``P = LᵀL`` maximizing α-bound ``γ``.

    Returns ``(P*, ρ*, α*, λ_max(P*), γ*)`` or all-``None`` on failure.
    """
    from scipy.linalg import solve_discrete_lyapunov
    from scipy.optimize import minimize as scipy_minimize

    n = A_cl_np.shape[0]
    A_cl = A_cl_np.astype(np.float64)
    CtC = (C.T @ C).astype(np.float64)

    def neg_gamma_from_params(L_params):
        L = np.zeros((n, n))
        L[np.tril_indices(n)] = L_params
        P = L @ L.T
        try:
            P_eigvals = np.linalg.eigvalsh(P)
            if P_eigvals.min() < 1e-10:
                return 1e10
            M = np.linalg.solve(P, A_cl.T @ P @ A_cl)
            rho_sq = np.max(np.abs(np.linalg.eigvals(M)))
            if rho_sq >= 1.0:
                return 1e4 * rho_sq
            if rho_sq < 0:
                return 1e10
            rho = np.sqrt(rho_sq)
            alpha = np.max(np.linalg.eigvalsh(np.linalg.solve(P, CtC)))
            if alpha <= 0:
                return 1e10
            lam_max_P = P_eigvals.max()
            gamma = epsilon_x * (1.0 - rho) / np.sqrt(alpha * lam_max_P) - eta
            return -gamma
        except np.linalg.LinAlgError:
            return 1e10

    candidates = []
    eigvals_cl = np.linalg.eigvals(A_cl)
    rho_cl = np.max(np.abs(eigvals_cl))
    print(f"  A_cl spectral radius: {rho_cl:.6f}")
    try:
        P_lyap = solve_discrete_lyapunov(A_cl.T, np.eye(n))
        lyap_eigvals = np.linalg.eigvalsh(P_lyap)
        print(
            f"  Lyapunov P: λ_min={lyap_eigvals.min():.6f}, "
            f"λ_max={lyap_eigvals.max():.6f}, "
            f"cond={lyap_eigvals.max() / lyap_eigvals.min():.4f}"
        )
        candidates.append(("Lyapunov", P_lyap))
    except Exception as e:
        print(f"  Lyapunov solve failed: {e}")

    candidates.append(("DARE", P_init))
    candidates.append(("identity", np.eye(n)))
    for scale in [0.1, 0.5, 2.0, 5.0, 10.0, 100.0]:
        candidates.append((f"{scale}*I", np.eye(n) * scale))

    x0 = None
    for name, P_candidate in candidates:
        try:
            P_c = P_candidate.astype(np.float64)
            L_c = np.linalg.cholesky(P_c)
            params_c = L_c[np.tril_indices(n)]
            val = neg_gamma_from_params(params_c)
            if val < 1e4:
                x0 = params_c
                print(f"  Initialized from {name} (γ={-val:.6f})")
                break
            else:
                print(f"  {name}: infeasible (obj={val:.4f})")
        except np.linalg.LinAlgError:
            print(f"  {name}: Cholesky failed (not positive definite)")
            continue

    if x0 is None:
        print("\033[91m  Could not find feasible initial P (A_cl may be unstable)\033[0m")
        return None, None, None, None, -np.inf

    print(f"  Optimizing alpha bound P ({len(x0)} parameters)...")
    result = scipy_minimize(
        fun=neg_gamma_from_params,
        x0=x0,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    L_best = np.zeros((n, n))
    L_best[np.tril_indices(n)] = result.x
    P_best = L_best @ L_best.T
    best_gamma = -result.fun
    rho_sq = np.max(np.abs(np.linalg.eigvals(np.linalg.solve(P_best, A_cl.T @ P_best @ A_cl))))
    best_rho = np.sqrt(max(rho_sq, 0.0))
    best_alpha = np.max(np.linalg.eigvalsh(np.linalg.solve(P_best, CtC)))
    lam_max_P = np.max(np.linalg.eigvalsh(P_best))
    print(
        f"  Optimization {'converged' if result.success else 'did not converge'} "
        f"after {result.nit} iterations"
    )
    return P_best, best_rho, best_alpha, lam_max_P, best_gamma


def run_sdp_optimization(lqr, epsilon, eta, cfg):
    """Run SDP P-optimization if enabled; returns ``(ρ², κ, γ)`` or ``None``."""
    if not cfg.get("optimize_lyapunov_P", True):
        return None
    A_cl_np = lqr.closed_loop.numpy()
    sdp_P, sdp_rho, sdp_kappa, sdp_gamma = optimize_lyapunov_P(A_cl_np, epsilon, eta)
    if sdp_P is None:
        print("\033[91m  SDP optimization failed, using LQR values\033[0m")
        return None
    return sdp_rho ** 2, sdp_kappa, sdp_gamma


# ---------------------------------------------------------------------------
# Bound-report orchestrators (ported from legacy launch/run.py +
# launch/stability_utils.py). Each prints its block and returns a dict.
# ---------------------------------------------------------------------------


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def eigen_bound_report(
    *,
    A,
    B_mat,
    lqr,
    m: float,
    max_tracking_error_x: float,
    max_displacement_x: float,
    err_mean: float,
    err_std: float,
) -> dict:
    """Spectral-radius / transient-constant (eigen) bound report."""
    print("\n" + "=" * 50)
    print("  Eigen Bound")
    print("=" * 50)

    closed_loop = lqr.closed_loop
    rho = spectral_radius(closed_loop)
    C = transient_constant(closed_loop)
    gain_norm = lqr.gain_norm.item()

    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)
    max_runtime_error_latent = max_tolerable_model_error(rho, C, max_tracking_error_latent, eta)
    residual_ctrl_budget = max_tracking_error_latent * gain_norm

    print(f"  Spectral radius A-BF (closed):         {rho:.6f}")
    print(f"  Transient constant (C):                {C:.6f}")
    print(f"  max_tracking_error_x:                  {max_tracking_error_x:.6f}")
    print(f"  max_displacement_x:                    {max_displacement_x:.6f}")
    print(f"  max_tracking_error_latent:             {max_tracking_error_latent:.6f}")
    print(f"  eta (m * max_displacement_x):          {eta:.6f}")
    if max_runtime_error_latent < 0:
        print(f"{_RED}  max_runtime_error_latent:              {max_runtime_error_latent:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  max_runtime_error_latent:              {max_runtime_error_latent:.6f}{_RESET}")
    print(f"  residual_ctrl_budget:                  {residual_ctrl_budget:.6f}")

    err_2sigma = err_mean + 2 * err_std
    print("  " + "-" * 48)
    print(f"  One-step latent error mean:            {err_mean:.6f}")
    print(f"  One-step latent error std:             {err_std:.6f}")
    if err_2sigma > max_runtime_error_latent:
        print(f"{_RED}  One-step latent error mean+2σ:         {err_2sigma:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  One-step latent error mean+2σ:         {err_2sigma:.6f}{_RESET}")

    return {
        "rho_closed_loop": float(rho),
        "C": float(C),
        "max_tracking_error_latent": float(max_tracking_error_latent),
        "eta": float(eta),
        "max_runtime_error_latent": float(max_runtime_error_latent),
        "residual_ctrl_budget": float(residual_ctrl_budget),
        "latent_err_mean": float(err_mean),
        "latent_err_std": float(err_std),
    }


def lyapunov_bound_report(
    *,
    model,
    A,
    B_mat,
    lqr,
    Q,
    R_cost,
    m: float,
    max_tracking_error_x: float,
    max_displacement_x: float,
    err_mean: float,
    err_std: float,
    scale_B: bool,
    B_scale,
    optimize_lyapunov_P_flag: bool,
    aug_trajectories,
    device,
) -> dict:
    """Lyapunov-bound report (with optional SDP P optimization)."""
    print("\n" + "=" * 50)
    print("  Lyapunov Bound")
    print("=" * 50)

    gain_norm = lqr.gain_norm.item()
    P, kappa_P, rho_sq, P_eigvals = compute_lyapunov_params(lqr, Q, R_cost)
    rho_sq_lqr = rho_sq
    kappa_P_lqr = kappa_P

    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)

    sdp_result = None
    if optimize_lyapunov_P_flag:
        sdp_result = run_sdp_optimization(
            lqr, max_tracking_error_latent, eta,
            {"optimize_lyapunov_P": True},
        )
    if sdp_result is not None:
        gamma_lqr = lyapunov_gamma(max_tracking_error_latent, rho_sq_lqr, kappa_P_lqr, eta)
        rho_sq, kappa_P, sdp_gamma = sdp_result
        print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
        print(f"  κ(P) (LQR):                            {kappa_P_lqr:.6f}")
        print(f"  γ_max (LQR):                           {gamma_lqr:.6f}")
        print(f"  ρ² (SDP-optimized):                    {rho_sq:.6f}")
        print(f"  κ(P) (SDP-optimized):                  {kappa_P:.6f}")
        print(f"  γ_max (SDP):                           {sdp_gamma:.6f}")

    delta_max_lyap = lyapunov_gamma(max_tracking_error_latent, rho_sq, kappa_P, eta)
    A_norm = torch.linalg.norm(A, ord=2).item()
    BtPB = compute_BtPB(lqr, B_mat, P)
    residual_ctrl_budget = max_tracking_error_latent * gain_norm
    if scale_B:
        residual_ctrl_budget /= float(B_scale)

    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    print(f"  B^T P B:                               {BtPB:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    if residual_ctrl_budget > 0.5:
        print(f"{_GREEN}  Residual ctrl budget:                  {residual_ctrl_budget:.6f} N-m{_RESET}")
    else:
        print(f"  Residual ctrl budget:                  {residual_ctrl_budget:.6f} N-m")
    print(f"  κ(P) = λ_max(P)/λ_min(P):              {kappa_P:.6f}")
    print(f"  ρ² (Lyapunov):                         {rho_sq:.6f}")
    if delta_max_lyap < 0:
        print(f"{_RED}  δ_max (Lyapunov):                      {delta_max_lyap:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  δ_max (Lyapunov):                      {delta_max_lyap:.6f}{_RESET}")

    err_2sigma = err_mean + 2 * err_std
    print("  " + "-" * 48)
    print(f"  One-step latent error mean:            {err_mean:.6f}")
    print(f"  One-step latent error std:             {err_std:.6f}")
    if err_2sigma > delta_max_lyap:
        print(f"{_RED}  One-step latent error mean+2σ:         {err_2sigma:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  One-step latent error mean+2σ:         {err_2sigma:.6f}{_RESET}")
    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, delta_max_lyap, space="latent"
    )
    print(f"  Steps under δ_max:                     {count_under}/{total} ({fraction*100:.1f}%)")

    return {
        "kappa_P": float(kappa_P),
        "rho_sq_lyapunov": float(rho_sq),
        "max_tracking_error_latent": float(max_tracking_error_latent),
        "eta": float(eta),
        "delta_max_lyapunov": float(delta_max_lyap),
        "BtPB": float(BtPB),
        "residual_ctrl_budget": float(residual_ctrl_budget),
        "P": P.numpy().tolist(),
    }


def m_free_bound_report(
    *,
    model,
    A,
    B_mat,
    lqr,
    Q,
    R_cost,
    u_max: float,
    max_tracking_error_x: float,
    m: float,
    err_mean: float,
    err_std: float,
    optimize_lyapunov_P_flag: bool,
    aug_trajectories,
    device,
) -> dict:
    """Lipschitz-m-free bound: latent-space bounds derived from measured error
    and actuator budget rather than the encoder lower-Lipschitz constant."""
    print("\n" + "=" * 50)
    print("  Lipschitz-m-Free Bound")
    print("=" * 50)

    gain_norm = lqr.gain_norm.item()
    P, kappa_P, rho_sq, P_eigvals = compute_lyapunov_params(lqr, Q, R_cost)
    rho_sq_lqr = rho_sq
    kappa_P_lqr = kappa_P
    F = lqr.F

    lam_min_P_lqr = P_eigvals.min().item()
    lam_max_P_lqr = P_eigvals.max().item()
    BtPB_lqr = compute_BtPB(lqr, B_mat, P)
    A_norm = torch.linalg.norm(A, ord=2).item()
    R_val = err_mean + 2 * err_std
    max_latent_diff = compute_max_latent_diff_observed(model, aug_trajectories, device)

    print(f"  u_max:                                 {u_max:.6f}")
    print(f"  max latent space difference:           {max_latent_diff:.6f}")
    print(f"  One-step latent error mean:            {err_mean:.6f}")
    print(f"  One-step latent error std:             {err_std:.6f}")
    print(f"  R (mean + 2σ):                         {R_val:.6f}")
    if max_latent_diff > 0:
        print(f"  R / max_latent_diff:                   {R_val / max_latent_diff:.6f}")

    print(f"  ||A|| (spectral norm):                 {A_norm:.6f}")
    print(f"  B^T P B (LQR):                         {BtPB_lqr:.6f}")
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    print(f"  ρ² (LQR):                              {rho_sq_lqr:.6f}")
    print(f"  κ(P) (LQR):                            {kappa_P_lqr:.6f}")
    print(f"  λ_min(P) (LQR):                        {lam_min_P_lqr:.6f}")
    print(f"  λ_max(P) (LQR):                        {lam_max_P_lqr:.6f}")

    eps_for_opt = m * max_tracking_error_x
    BF_pre = B_mat @ F
    BF_norm_pre = torch.linalg.norm(BF_pre, ord=2).item()
    eta_for_opt = BF_norm_pre * (u_max / gain_norm) if gain_norm > 0 else 0.0
    gamma_lqr = lyapunov_gamma(eps_for_opt, rho_sq_lqr, kappa_P_lqr, eta_for_opt)
    print(f"  γ_max (LQR, m-derived ε):              {gamma_lqr:.6f}")

    if optimize_lyapunov_P_flag:
        sdp_result = run_sdp_optimization(
            lqr, eps_for_opt, eta_for_opt,
            {"optimize_lyapunov_P": True},
        )
        if sdp_result is not None:
            rho_sq, kappa_P, sdp_gamma = sdp_result
            print(f"  ρ² (SDP-optimized):                    {rho_sq:.6f}")
            print(f"  κ(P) (SDP-optimized):                  {kappa_P:.6f}")
            print(f"  γ_max (SDP, m-derived ε):              {sdp_gamma:.6f}")

    print("  " + "-" * 48)
    print(f"  ρ² (final):                            {rho_sq:.6f}")
    print(f"  κ(P) (final):                          {kappa_P:.6f}")

    z_ref_limit = u_max / gain_norm if gain_norm > 0 else 0.0
    BF = B_mat @ F
    BF_norm = torch.linalg.norm(BF, ord=2).item()
    eta = BF_norm * z_ref_limit
    print(f"  z_ref_limit = u_max / ||F||:           {z_ref_limit:.6f}")
    print(f"  ||B @ F||_2:                           {BF_norm:.6f}")
    print(f"  eta = ||BF|| * z_ref_limit:            {eta:.6f}")

    if rho_sq < 1.0:
        epsilon_max = (R_val + eta) * math.sqrt(kappa_P) / (1.0 - math.sqrt(rho_sq))
    else:
        epsilon_max = float("inf")
    if max_latent_diff > 0 and epsilon_max > max_latent_diff / 2:
        print(f"{_RED}  ε_max (max tracking error latent):     {epsilon_max:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  ε_max (max tracking error latent):     {epsilon_max:.6f}{_RESET}")
    if max_latent_diff > 0:
        print(f"  ε_max / max_latent_diff:               {epsilon_max / max_latent_diff:.6f}")

    gamma_max = lyapunov_gamma(epsilon_max, rho_sq, kappa_P, eta)
    if gamma_max < 0:
        print(f"{_RED}  γ_max (Lyapunov, m-free):              {gamma_max:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  γ_max (Lyapunov, m-free):              {gamma_max:.6f}{_RESET}")
    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, gamma_max, space="latent"
    )
    print(f"  Steps under γ_max:                     {count_under}/{total} ({fraction*100:.1f}%)")

    return {
        "u_max": float(u_max),
        "max_latent_diff": float(max_latent_diff),
        "kappa_P": float(kappa_P),
        "rho_sq_lyapunov": float(rho_sq),
        "R_model_error": float(R_val),
        "z_ref_limit": float(z_ref_limit),
        "eta": float(eta),
        "max_tracking_error_latent": float(epsilon_max),
        "gamma_max_lyapunov": float(gamma_max),
        "BF_norm": float(BF_norm),
        "steps_under_gamma_max": int(count_under),
        "total_steps": int(total),
    }


def alpha_bound_report(
    *,
    model,
    A,
    B_mat,
    lqr,
    real_state_dim: int,
    alpha_epsilon_x: float,
    alpha_eta: float,
    err_mean: float,
    err_std: float,
    aug_trajectories,
    device,
) -> dict:
    """Alpha-bound (state-extraction) stability report.

    Requires ``model.prepend_state == True`` so the latent state explicitly
    carries the real-state dims to extract via ``C = [I_p | 0_{p × q}]``.
    """
    print("\n" + "=" * 50)
    print("  Alpha Bound")
    print("=" * 50)

    if not getattr(model, "prepend_state", False):
        print(f"{_YELLOW}  Skipped: alpha bound requires prepend_state=True.{_RESET}")
        return {"skipped": True}

    n = A.shape[0]
    p = real_state_dim
    C = np.zeros((p, n), dtype=np.float64)
    C[:p, :p] = np.eye(p)

    A_cl = lqr.closed_loop.numpy().astype(np.float64)
    P_init = lqr.P.numpy().astype(np.float64)

    best_P, best_rho, best_alpha, lam_max_P, best_gamma = optimize_alpha_P(
        A_cl, C, alpha_epsilon_x, alpha_eta, P_init
    )

    if best_P is None:
        print(f"{_RED}  Alpha bound optimization failed.{_RESET}")
        return {"skipped": True, "reason": "optimization failed"}

    best_P_eigs = np.linalg.eigvalsh(best_P)
    lam_min_P = float(best_P_eigs.min())
    kappa_P = float(lam_max_P / lam_min_P)

    print(f"  ρ² (optimized):                        {best_rho**2:.6f}")
    print(f"  α = λ_max(P⁻¹CᵀC):                    {best_alpha:.6f}")
    print(f"  λ_min(P):                              {lam_min_P:.6f}")
    print(f"  λ_max(P):                              {lam_max_P:.6f}")
    print(f"  κ(P):                                  {kappa_P:.6f}")
    print(f"  ε_x:                                   {alpha_epsilon_x:.6f}")
    print(f"  η:                                     {alpha_eta:.6f}")

    gamma_base = alpha_epsilon_x * (1.0 - best_rho) / math.sqrt(best_alpha * lam_max_P)
    gamma_config = gamma_base - alpha_eta
    if gamma_config < 0:
        print(f"{_RED}  γ_max (config η):                      {gamma_config:.6f}{_RESET}")
    else:
        print(f"{_GREEN}  γ_max (config η):                      {gamma_config:.6f}{_RESET}")

    R_val = err_mean + 2 * err_std
    if R_val > 0:
        print(f"  γ_max / R:                             {gamma_config / R_val:.6f}")
    count_under, total, fraction = count_steps_under_threshold(
        model, aug_trajectories, device, gamma_config, space="latent"
    )
    print(f"  Steps under γ_max:                     {count_under}/{total} ({fraction*100:.1f}%)")

    return {
        "alpha_epsilon_x": float(alpha_epsilon_x),
        "alpha_eta": float(alpha_eta),
        "alpha": float(best_alpha),
        "rho_alpha": float(best_rho),
        "lambda_min_P": lam_min_P,
        "lambda_max_P": float(lam_max_P),
        "kappa_P": kappa_P,
        "gamma_max_alpha": float(gamma_config),
        "steps_under_gamma_max": int(count_under),
        "total_steps": int(total),
    }
