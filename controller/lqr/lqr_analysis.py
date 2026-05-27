"""Kept LQR utilities used by the unified analysis in :mod:`lqr_analysisv2`.

Only construction (``setup_lqr``), the DARE-derived Lyapunov parameters,
and the two P-optimizers live here. The previous per-bound report
orchestrators (eigen / Lyapunov / m-free / alpha) were folded into the
single ``generate_P_and_bound`` + ``run_stability_report`` pipeline in
``lqr_analysisv2``.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch

from controller.lqr.lqr import LQR

if TYPE_CHECKING:
    from config.manager import LQRControllerCfg


# ---------------------------------------------------------------------------
# Config diagonal validation.
# ---------------------------------------------------------------------------


def _diag_to_tensor(diag, expected_len: int, name: str) -> torch.Tensor:
    """Validate ``diag`` and return ``torch.diag(diag)`` as a float32 tensor.

    Raises ``ValueError`` if ``diag`` is ``None`` or its length doesn't
    match ``expected_len``. ``name`` is the config-field name used in the
    error message.
    """
    if diag is None:
        raise ValueError(
            f"lqr_controller_cfg.{name} is required but was not set. "
            f"Provide a list of {expected_len} diagonal entries."
        )
    if len(diag) != expected_len:
        raise ValueError(
            f"lqr_controller_cfg.{name} has length {len(diag)} but the model "
            f"requires length {expected_len}."
        )
    return torch.diag(torch.tensor([float(v) for v in diag], dtype=torch.float64))


def build_C(C_mask, latent_dim: int) -> np.ndarray:
    """Build the rectangular C from a selector mask over latent dimensions.

    ``C_mask`` must be length ``latent_dim`` with entries in ``{0, 1}``. Each
    ``1`` at index ``i`` contributes a row ``eᵢᵀ`` to ``C``. Returns a
    ``(sum(mask), latent_dim)`` numpy array. Raises ``ValueError`` on missing
    input, wrong length, non-binary entries, or an all-zero mask.
    """
    if C_mask is None:
        raise ValueError(
            "lqr_controller_cfg.C_mask is required but was not set. "
            f"Provide a list of {latent_dim} entries (0/1) selecting which "
            "latent dims appear in the output."
        )
    if len(C_mask) != latent_dim:
        raise ValueError(
            f"lqr_controller_cfg.C_mask has length {len(C_mask)} but the "
            f"model requires length {latent_dim} (actual_latent)."
        )
    selected = []
    for i, v in enumerate(C_mask):
        iv = int(v)
        if iv not in (0, 1) or iv != v:
            raise ValueError(
                f"lqr_controller_cfg.C_mask[{i}] = {v!r} is not 0 or 1."
            )
        if iv == 1:
            selected.append(i)
    if not selected:
        raise ValueError(
            "lqr_controller_cfg.C_mask must select at least one latent dim "
            "(got all zeros)."
        )
    C = np.zeros((len(selected), latent_dim), dtype=np.float64)
    for row, col in enumerate(selected):
        C[row, col] = 1.0
    return C


# ---------------------------------------------------------------------------
# LQR construction.
# ---------------------------------------------------------------------------


def setup_lqr(
    A,
    B_mat,
    lqr_cfg: "LQRControllerCfg",
    *,
    action_dim: int,
):
    """Build and solve the LQR controller from an :class:`LQRControllerCfg`.

    Q and R diagonals are validated against the loaded model's
    ``actual_latent = A.shape[0]`` and ``action_dim`` respectively; either
    missing or mismatched length raises ``ValueError``. Returns
    ``(lqr, Q, R_cost, B_scale)``; ``B_scale`` is 1.0 unless
    ``lqr_cfg.scale_B`` normalizes B's spectral norm first.
    """
    # All LQR/Lyapunov calcs run in float64 — cast at the boundary so the
    # downstream code never has to track dtypes.
    A = A.to(torch.float64)
    B_mat = B_mat.to(torch.float64)
    actual_latent = A.shape[0]
    Q = _diag_to_tensor(lqr_cfg.Q_diag, actual_latent, "Q_diag")
    R_cost = _diag_to_tensor(lqr_cfg.R_diag, action_dim, "R_diag")

    if lqr_cfg.scale_B:
        print("  Scaling B")
        B_scale = torch.linalg.norm(B_mat, ord=2)
        B_for_lqr = B_mat / B_scale
    else:
        B_scale = 1.0
        B_for_lqr = B_mat

    lqr = LQR(A, B_for_lqr, Q, R_cost)
    return lqr, Q, R_cost, B_scale


# ---------------------------------------------------------------------------
# DARE-derived Lyapunov parameters (raw LQR P).
# ---------------------------------------------------------------------------


def compute_lyapunov_params(lqr, Q, R_cost):
    """Return ``(P, κ(P), ρ², λ(P))`` from the LQR Lyapunov certificate.

    ``ρ² = 1 − λ_min(Q + FᵀRF) / λ_max(P)`` — the DARE-derived Lyapunov
    decrease rate for the raw LQR P. (For optimized P, use whatever ρ²
    the optimizer returned.)
    """
    P = lqr.P
    F = lqr.F
    P_eigvals = torch.linalg.eigvalsh(P)
    kappa_P = (P_eigvals.max() / P_eigvals.min()).item()
    Q_plus_FRF = Q + F.T @ R_cost @ F
    Q_eigvals = torch.linalg.eigvalsh(Q_plus_FRF)
    rho_sq = 1.0 - Q_eigvals.min().item() / P_eigvals.max().item()
    return P, kappa_P, rho_sq, P_eigvals


# ---------------------------------------------------------------------------
# SDP P-optimization (no-prepend branch).
# ---------------------------------------------------------------------------


def optimize_lyapunov_P(A_cl_np, epsilon, eta, rho_grid_size=100):
    """SDP sweep over ρ to maximize ``γ = ε(1−ρ)/√κ − η``.

    Returns ``(P*, ρ*, κ*, γ*)`` or ``(None, None, None, -inf)`` on failure.
    """
    import cvxpy as cp

    n = A_cl_np.shape[0]
    A_cl = A_cl_np

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


# ---------------------------------------------------------------------------
# L-BFGS α-bound P-optimization (prepend branch).
# ---------------------------------------------------------------------------


def optimize_alpha_P(A_cl_np, C, epsilon_x, eta, P_init):
    """L-BFGS-B over ``P = LᵀL`` maximizing α-bound ``γ``.

    Returns ``(P*, ρ*, α*, λ_max(P*), γ*)`` or all-``None`` on failure.
    """
    from scipy.linalg import solve_discrete_lyapunov
    from scipy.optimize import minimize as scipy_minimize

    n = A_cl_np.shape[0]
    A_cl = A_cl_np
    CtC = C.T @ C

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
    print(f"  A_cl spectral radius: {rho_cl:.2f}")
    try:
        P_lyap = solve_discrete_lyapunov(A_cl.T, np.eye(n))
        lyap_eigvals = np.linalg.eigvalsh(P_lyap)
        print(
            f"  Lyapunov P: λ_min={lyap_eigvals.min():.2f}, "
            f"λ_max={lyap_eigvals.max():.2f}, "
            f"cond={lyap_eigvals.max() / lyap_eigvals.min():.2f}"
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
            L_c = np.linalg.cholesky(P_candidate)
            params_c = L_c[np.tril_indices(n)]
            val = neg_gamma_from_params(params_c)
            if val < 1e4:
                x0 = params_c
                print(f"  Initialized from {name} (γ={-val:.2f})")
                break
            else:
                print(f"  {name}: infeasible (obj={val:.2f})")
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


# ---------------------------------------------------------------------------
# Level-2 β-bound P-optimization (works for prepend and no-prepend).
# ---------------------------------------------------------------------------


def _print_C_observability(A_cl_np: np.ndarray, C: np.ndarray, threshold: float = 0.9) -> None:
    """Per-mode C-observability strength for near-unit-circle A_cl eigenvalues.

    For each right eigenvector ``v`` of A_cl with ``|λ| > threshold``, prints
    ``||Cv|| / ||v||`` — i.e. how much of that mode survives the output map.
    Modes with a small ratio are weakly observable through C and will tend
    to inflate ``λ_max(P)`` in the level-2 SDP.
    """
    eigvals, V = np.linalg.eig(A_cl_np)
    slow = [(i, lam) for i, lam in enumerate(eigvals) if abs(lam) > threshold]
    if not slow:
        return
    print(f"  --- C-observability of near-unit-circle A_cl modes (|λ| > {threshold}) ---")
    for i, lam in slow:
        v = V[:, i]
        v_norm = float(np.linalg.norm(v))
        Cv_over_v = float(np.linalg.norm(C @ v) / v_norm) if v_norm > 0 else float("nan")
        re, im = float(lam.real), float(lam.imag)
        print(
            f"    λ_{i:02d} = {re:+.4f}{im:+.4f}j  "
            f"|λ|={abs(lam):.4f}  ||Cv||/||v||={Cv_over_v:.3e}"
        )


def _check_detectability(A_cl_np: np.ndarray, C: np.ndarray, rho_target: float) -> None:
    """PBH detectability check: every A_cl mode with ``|λ| > rho_target`` must
    be detectable through ``C``. Raises ``ValueError`` on failure (no fallback).
    """
    n = A_cl_np.shape[0]
    for lam in np.linalg.eigvals(A_cl_np):
        if abs(lam) > rho_target:
            stacked = np.vstack([A_cl_np - lam * np.eye(n), C])
            r = int(np.linalg.matrix_rank(stacked))
            if r != n:
                raise ValueError(
                    f"undetectable mode at λ={lam} "
                    f"(rank([A_cl-λI; C])={r}, expected {n}); "
                    "the level-2 β-optimization requires (A_cl, C) to be "
                    "detectable above rho_target."
                )


def optimize_level2_P(
    A_cl_np: np.ndarray,
    C: np.ndarray,
    epsilon_x: float,
    eta: float,
    *,
    rho_target: float = 0.0,
    rho_grid_size: int = 100,
    delta: float = 0.001,
    rho_validation_slack: float = 1.01,
):
    """Level-2 SDP β-optimizer. Works for both prepend and no-prepend.

    Solves, for each ρ in a grid, ``minimize λ_max(P) s.t. P ⪰ 0,
    Aᵀ_cl P A_cl ⪯ ρ²P, P ⪰ CᵀC``. β is normalized to 1 by scaling P
    (legitimate since the SDP is scale-invariant in P), giving the bound
    ``γ = ε_x · (1−ρ) / √λ_max(P) − η`` — output-space (via C) directly.

    A PBH detectability pre-check runs *before* the grid sweep; any A_cl
    eigenvalue with ``|λ| > rho_target`` that is not detectable through C
    raises ``ValueError`` immediately.

    Solver: MOSEK if available (high-precision), else SCS with tight
    tolerances (``eps=1e-9, max_iters=100000``). The grid starts at
    ``σ(A_cl) + delta`` (default 0.02) to keep the feasible set away from
    the degenerate ``ρ ≈ σ(A_cl)`` regime where P goes near-singular.

    Returned P is also *validated*: the true P-norm contraction
    ``ρ_true = √max|eig(P⁻¹ Aᵀ_cl P A_cl)|`` is recomputed and the grid
    point is rejected if ``ρ_true > ρ · rho_validation_slack``, so any
    P whose claimed ρ is the solver's slack rather than truth gets thrown
    out.

    Returns ``(P*, ρ*, λ_max(P*), γ*)`` or ``(None, None, None, -inf)`` if
    every grid point was infeasible.
    """
    import cvxpy as cp

    _check_detectability(A_cl_np, C, rho_target)
    _print_C_observability(A_cl_np, C)

    n = A_cl_np.shape[0]
    sigma_Acl = float(np.max(np.abs(np.linalg.eigvals(A_cl_np))))
    if sigma_Acl >= 1.0:
        raise ValueError(
            f"level-2 β-optimization requires σ(A_cl) < 1 "
            f"(got σ={sigma_Acl:.4f}); the closed loop is not Schur-stable."
        )

    rho_grid_start = sigma_Acl + delta
    if rho_grid_start >= 0.999:
        raise ValueError(
            f"level-2 β-optimization: σ(A_cl)+delta = {rho_grid_start:.4f} ≥ 0.999; "
            f"no room for a ρ grid. Either reduce delta (currently {delta}) "
            f"or improve the LQR closed loop."
        )
    rho_grid = np.linspace(rho_grid_start, 0.999, rho_grid_size)
    CtC = C.T @ C

    # Build the SDP once with ρ² as a Parameter so cvxpy compiles a single
    # time and the grid sweep is solve-only. Same problem, same solver, same
    # answer — just no per-iteration canonicalization overhead.
    rho_sq_param = cp.Parameter(nonneg=True)
    P = cp.Variable((n, n), symmetric=True)
    t = cp.Variable(nonneg=True)
    constraints = [
        P >> 0,
        A_cl_np.T @ P @ A_cl_np << rho_sq_param * P,
        P >> CtC,
        P << t * np.eye(n),
    ]
    prob = cp.Problem(cp.Minimize(t), constraints)

    # Prefer MOSEK (interior-point, high precision); fall back to SCS with
    # tightened tolerances.
    use_mosek = "MOSEK" in cp.installed_solvers()
    if use_mosek:
        solve_kwargs = {"solver": cp.MOSEK, "verbose": False}
        solver_label = "MOSEK"
    else:
        solve_kwargs = {
            "solver": cp.SCS,
            "verbose": False,
            "eps": 1e-9,
            "max_iters": 100000,
        }
        solver_label = "SCS (eps=1e-9, max_iters=1e5)"

    best_gamma = -np.inf
    best_P = None
    best_rho = None
    best_lam_max_P = None
    rejected_validation = 0

    print(
        f"  Level-2 β-SDP sweep: {len(rho_grid)} grid points over "
        f"ρ ∈ [{rho_grid[0]:.4f}, {rho_grid[-1]:.4f}]  "
        f"(σ(A_cl)={sigma_Acl:.4f}, solver={solver_label})"
    )
    for i, rho in enumerate(rho_grid):
        rho_sq_param.value = float(rho) ** 2
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(**solve_kwargs)
        except cp.SolverError:
            continue

        if (i + 1) % 10 == 0 or i == len(rho_grid) - 1:
            print(
                f"    [{i + 1:3d}/{len(rho_grid)}] ρ={float(rho):.4f}  "
                f"status={prob.status}  best γ={best_gamma:.3e}  "
                f"rejected={rejected_validation}"
            )

        if prob.status not in ("optimal", "optimal_inaccurate") or t.value is None:
            continue
        lam_max_P = float(t.value)
        if lam_max_P <= 0:
            continue

        # Validate: the SDP may have certified ``rho`` only up to solver
        # slack. Recompute the true P-norm contraction and reject if it
        # exceeds the grid ρ by more than ``rho_validation_slack``.
        try:
            M = np.linalg.solve(P.value, A_cl_np.T @ P.value @ A_cl_np)
            rho_true = float(math.sqrt(max(np.max(np.abs(np.linalg.eigvals(M))), 0.0)))
        except np.linalg.LinAlgError:
            rejected_validation += 1
            continue
        if rho_true > float(rho) * rho_validation_slack:
            rejected_validation += 1
            continue

        gamma = epsilon_x * (1.0 - float(rho)) / math.sqrt(lam_max_P) - eta
        if gamma > best_gamma:
            best_gamma = gamma
            best_P = P.value.copy()
            best_rho = float(rho)
            best_lam_max_P = lam_max_P

    if rejected_validation:
        print(
            f"  Level-2 β-SDP: rejected {rejected_validation}/{len(rho_grid)} "
            f"grid points whose returned P violated ρ_true ≤ ρ · "
            f"{rho_validation_slack}."
        )
    return best_P, best_rho, best_lam_max_P, best_gamma
