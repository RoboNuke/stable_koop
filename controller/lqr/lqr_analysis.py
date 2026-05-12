"""LQR-specific stability analysis — anything that uses ``F`` or ``P``.

Folded from the LQR-specific half of the legacy ``launch/stability_utils.py``
and the Lyapunov optimization helpers from ``model/utils.py``. Generic
analyses live in :mod:`controller.controller_analysis`.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

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
