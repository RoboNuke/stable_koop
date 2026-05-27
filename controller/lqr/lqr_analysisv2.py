"""LQR reporting/bound helpers (v2).

Small, composable helpers that replace the monolithic orchestrators in
``lqr_analysis.py``. Built up incrementally — each helper prints a
formatted report and returns a dict of stats for downstream use.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from controller.controller_analysis import (
    compute_encoder_lipschitz_bounds,
    control_analysis,
    count_steps_under_threshold,
)
from controller.lqr.lqr_analysis import (
    compute_lyapunov_params,
    optimize_alpha_P,
    optimize_level2_P,
    optimize_lyapunov_P,
)


_VALID_OPTIMIZERS = ("none", "matching", "beta")


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _color_ratio(value: float) -> str:
    if value >= 1.0:
        return _GREEN
    if value >= 0.75:
        return _YELLOW
    return _RED


def _color_pct(value: float) -> str:
    if value > 0.85:
        return _GREEN
    if value >= 0.75:
        return _YELLOW
    return _RED


def _stats_dict(errs_tensor) -> dict:
    """Compute mean / std / max / R = mean+2σ / pct ≤ R / total for a 1-D tensor."""
    mean = errs_tensor.mean().item()
    std = errs_tensor.std().item()
    max_err = errs_tensor.max().item()
    R = mean + 2.0 * std
    pct_under_R = (errs_tensor <= R).float().mean().item()
    total = int(errs_tensor.numel())
    return {
        "mean": mean,
        "std": std,
        "max": max_err,
        "R": R,
        "pct_under_R": pct_under_R,
        "total": total,
    }


def _print_stats_block(title: str, stats: dict) -> None:
    print(f"--- {title} ---")
    print(f"  transitions:      {stats['total']}")
    print(f"  mean:             {stats['mean']:.2e}")
    print(f"  std:              {stats['std']:.2e}")
    print(f"  max:              {stats['max']:.2e}")
    print(f"  R = mean + 2*std: {stats['R']:.2e}")
    print(f"  pct <= R:         {100.0 * stats['pct_under_R']:.2f}%")


def _print_identifiability_block(stats: dict) -> None:
    """Pretty-print the dataset-identifiability section from a stats dict."""
    print("--- Dataset identifiability ([Z; U_base] regression matrix) ---")
    print(f"  n_z (latent dim):                      {int(stats['n_z'])}")
    print(f"  n_u (input dim):                       {int(stats['n_u'])}")
    print(f"  T (total transitions):                 {int(stats['T'])}")
    rank = int(stats["rank"])
    expected = int(stats["expected_rank"])
    color = _GREEN if rank == expected else _RED
    tag = "identifiable" if rank == expected else "NOT identifiable — B is arbitrary"
    print(
        f"{color}  rank([Z; U_base]):                     "
        f"{rank} / {expected}  [{tag}]{_RESET}"
    )
    print(f"  largest  σ([Z; U_base]):               {float(stats['largest_singular_value']):.4e}")
    print(f"  smallest σ([Z; U_base]):               {float(stats['smallest_singular_value']):.4e}")
    cond = stats.get("condition_number")
    if cond is not None:
        print(f"  condition number σ_max / σ_min:        {float(cond):.4e}")


def dataset_identifiability_report(
    model, aug_trajectories, device, *, verbose: bool = True,
) -> dict:
    """Rank / smallest-σ check on the regression matrix ``M = [Z; U_base]``.

    ``Z`` is the encoded state at each transition and ``U_base`` is the
    Koopman input at that transition (the ``u`` that goes through ``B``
    in the regression ``z_{t+1} = A z_t + B u_t``). Stacked column-wise
    over every transition in ``aug_trajectories`` we get

        M = [Z; U_base] ∈ R^((n_z + n_u) × T).

    ``B`` is identifiable iff ``M`` has full row rank ``n_z + n_u``;
    equivalently iff the smallest singular value of ``M`` is bounded away
    from zero. When ``smallest σ ≈ 0``, ``u_base`` is (nearly) a linear
    function of ``z`` over the dataset and ``B`` is not uniquely
    determined by least squares.
    """
    model.to(device)
    model.eval()
    Z_cols: list = []
    U_cols: list = []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            if len(actions) == 0:
                continue
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z = model.encode(states_t[:T_act])                # (T_act, n_z)
            Z_cols.append(z.detach().cpu().numpy().T)         # (n_z, T_act)
            U_cols.append(actions_t.detach().cpu().numpy().T) # (n_u, T_act)
    Z = np.concatenate(Z_cols, axis=1)
    U = np.concatenate(U_cols, axis=1)
    M = np.vstack([Z, U])  # (n_z + n_u, T)

    n_z, n_u = int(Z.shape[0]), int(U.shape[0])
    expected_rank = n_z + n_u
    rank = int(np.linalg.matrix_rank(M))
    svs = np.linalg.svd(M, compute_uv=False)
    largest_sv = float(svs[0])
    smallest_sv = float(svs[-1])
    cond = (largest_sv / smallest_sv) if smallest_sv > 0 else float("inf")

    stats = {
        "n_z": n_z,
        "n_u": n_u,
        "T": int(M.shape[1]),
        "rank": rank,
        "expected_rank": int(expected_rank),
        "identifiable": bool(rank == expected_rank),
        "largest_singular_value": largest_sv,
        "smallest_singular_value": smallest_sv,
        "condition_number": float(cond),
        "singular_values": [float(s) for s in svs],
    }

    if verbose:
        _print_identifiability_block(stats)

    return stats


def _render_pred_error_report(pred_stats: dict) -> None:
    """Print the same body that :func:`one_step_pred_error_report` would, from a stats dict."""
    _print_stats_block("One-step latent prediction error (training data)", pred_stats["latent"])
    _print_stats_block("One-step state reconstruction error (training data)", pred_stats["state"])


def _render_controllability_fit_report(ctrl_stats: dict) -> None:
    """Print the same body that :func:`controllability_fit_report` would, from a stats dict."""
    print("--- Koopman controllability fit ---")
    m = ctrl_stats.get("m")
    if m is None:
        print("  m (encoder lower Lipschitz): n/a (fixed encoder)")
    else:
        print(f"  m (encoder lower Lipschitz):           {float(m):.2e}")
    latent_dim = ctrl_stats.get("latent_dim")
    ctrl_rank = ctrl_stats.get("ctrl_rank")
    if latent_dim is not None and ctrl_rank is not None:
        print(f"  Controllability rank:                  {int(ctrl_rank)} / {int(latent_dim)}")
    mode_info = ctrl_stats.get("mode_controllability")
    if mode_info:
        print("  --- All eigenvalues of A (PBH controllability) ---")
        for i, mi in enumerate(mode_info):
            re, im = mi["eigenvalue"]
            mag = mi["magnitude"]
            proj = mi["pbh_projection"]
            tag = "controllable" if mi["controllable"] else "UNCONTROLLABLE"
            print(
                f"    λ_{i:02d} = {float(re):+.4f}{float(im):+.4f}j  "
                f"|λ|={float(mag):.4f}  |wᵀB|={float(proj):.2e}  [{tag}]"
            )
        slow = [(i, mi) for i, mi in enumerate(mode_info) if float(mi["magnitude"]) > 0.9]
        if slow:
            print("  --- Near-unit-circle modes (|λ| > 0.9) ---")
            for i, mi in slow:
                re, im = mi["eigenvalue"]
                print(
                    f"    λ_{i:02d} = {float(re):+.4f}{float(im):+.4f}j  "
                    f"|λ|={float(mi['magnitude']):.4f}  "
                    f"|wᵀB|={float(mi['pbh_projection']):.2e}"
                )
            if any("mode_projections" in mi for _, mi in slow):
                print(
                    "  --- Near-unit-circle A modes (|λ| > 0.9): "
                    "Koopman mode projections ---"
                )
                for i, mi in slow:
                    projections = mi.get("mode_projections")
                    if not projections:
                        continue
                    re, im = mi["eigenvalue"]
                    parts = "  ".join(
                        f"{name}={float(val):.3f}"
                        for name, val in projections.items()
                    )
                    print(
                        f"    λ_{i:02d} = {float(re):+.4f}{float(im):+.4f}j  "
                        f"|λ|={float(mi['magnitude']):.4f}  {parts}"
                    )
    else:
        # Legacy YAML without PBH info — fall back to the |λ| extrema.
        evs = ctrl_stats.get("open_loop_eigvals") or []
        mags = [math.hypot(float(z[0]), float(z[1])) for z in evs if isinstance(z, (list, tuple))]
        if mags:
            print(f"  open-loop |λ| max:                     {max(mags):.2f}")
            print(f"  open-loop |λ| min:                     {min(mags):.2f}")
    svals = ctrl_stats.get("B_singular_values")
    if svals:
        print("  --- Singular values of B ---")
        for i, s in enumerate(svals):
            print(f"    σ_{i}(B) = {float(s):.4e}")
    print(f"  ||A||_2 (spectral norm):               {float(ctrl_stats['A_spec_norm']):.2e}")
    print(f"  ||B||_2 (spectral norm):               {float(ctrl_stats['B_spec_norm']):.2e}")


def one_step_pred_error_report(model, aug_trajectories, device, *, verbose: bool = True) -> dict:
    """One-step prediction error stats over training transitions.

    Computes per-transition errors in *both* spaces:
      * latent: ``||z_{t+1} − predict(z_t, u_t)||``
      * state:  ``||x_{t+1} − decode(predict(z_t, u_t))||``

    Prints a stats block for each and returns a dict with both sets nested
    under ``latent`` and ``state`` keys plus a top-level ``R_latent`` /
    ``R_state`` shortcut for downstream consumers.
    """
    model.to(device)
    model.eval()
    latent_errs, state_errs = [], []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_next = model.encode(states_t[1 : T_act + 1])
            z_pred = model.predict(z_all, actions_t[:T_act])
            latent_errs.append(torch.linalg.norm(z_next - z_pred, dim=-1).cpu())
            x_pred = model.decode(z_pred)
            x_next = states_t[1 : T_act + 1]
            state_errs.append(torch.linalg.norm(x_next - x_pred, dim=-1).cpu())
    latent_stats = _stats_dict(torch.cat(latent_errs))
    state_stats = _stats_dict(torch.cat(state_errs))

    if verbose:
        _print_stats_block("One-step latent prediction error (training data)", latent_stats)
        _print_stats_block("One-step state reconstruction error (training data)", state_stats)

    return {
        "latent": latent_stats,
        "state": state_stats,
        "R_latent": latent_stats["R"],
        "R_state": state_stats["R"],
    }


def controllability_fit_report(
    model, A, B_mat, aug_trajectories, device,
    *, verbose: bool = True, encoder_lipschitz_batch_size: int = 4096,
    mode_projection_groups: dict | None = None,
) -> dict:
    """Koopman controllability-fit diagnostics.

    Computes:
      * ``m`` — lower Lipschitz constant of the encoder (min singular value
        of the encoder Jacobian over the training states). When the encoder
        is fixed (e.g. trig encoder), ``m`` is ``None``.
      * controllability rank of ``(A, B)``.
      * open-loop eigenvalues of ``A``.
      * spectral norms ``||A||_2`` and ``||B||_2``.

    Prints a formatted block and returns a dict.
    """
    if verbose:
        print("--- Koopman controllability fit ---")

    m_gx, _L_gx, m_full, _L_full = compute_encoder_lipschitz_bounds(
        model, aug_trajectories, device,
        verbose=verbose, batch_size=encoder_lipschitz_batch_size,
    )
    m = m_full if m_full is not None else m_gx
    if verbose:
        if m is None:
            print("  m (encoder lower Lipschitz): n/a (fixed encoder)")
        else:
            print(f"  m (encoder lower Lipschitz):           {m:.2e}")

    ctrl_rank, mode_info, B_singular_values = control_analysis(
        A, B_mat, verbose=verbose, mode_projection_groups=mode_projection_groups,
    )

    A_norm = torch.linalg.norm(A, ord=2).item()
    B_norm = torch.linalg.norm(B_mat, ord=2).item()

    if verbose:
        print(f"  ||A||_2 (spectral norm):               {A_norm:.2e}")
        print(f"  ||B||_2 (spectral norm):               {B_norm:.2e}")

    return {
        "m": None if m is None else float(m),
        "ctrl_rank": int(ctrl_rank),
        "latent_dim": int(A.shape[0]),
        "open_loop_eigvals": [m["eigenvalue"] for m in mode_info],
        "mode_controllability": mode_info,
        "B_singular_values": B_singular_values,
        "A_spec_norm": float(A_norm),
        "B_spec_norm": float(B_norm),
    }


def lqr_fit_report(
    label: str,
    *,
    P,
    F,
    A_cl,
    Q,
    R_cost,
    rho_sq: float,
    alpha: float | None = None,
) -> dict:
    """LQR fit quality report.

    Designed to be called twice — once on the raw LQR solution and once
    on any optimized-P refinement — using ``label`` to disambiguate the
    two header lines (rendered as ``--- LQR fit (<label>) ---``).

    Accepts the already-computed quantities directly so the caller controls
    how ``rho_sq`` and ``P`` were derived (e.g. DARE-based vs SDP-optimized).
    ``A_cl``, ``Q``, and ``R_cost`` are used to print the closed-loop
    spectral radius and the two ρ estimates (true P-norm contraction vs
    DARE Rayleigh bound).
    """
    P_t = torch.as_tensor(P, dtype=torch.float64)
    F_t = torch.as_tensor(F, dtype=torch.float64)

    P_eigvals = torch.linalg.eigvalsh(P_t)
    lam_min_P = P_eigvals.min().item()
    lam_max_P = P_eigvals.max().item()
    kappa_P = lam_max_P / lam_min_P
    F_norm = torch.linalg.norm(F_t, ord=2).item()

    P_np = P_t.detach().cpu().numpy()
    A_cl_np = A_cl.detach().cpu().numpy() if hasattr(A_cl, "detach") else np.asarray(A_cl)
    Q_np = Q.detach().cpu().numpy() if hasattr(Q, "detach") else np.asarray(Q)
    R_np = R_cost.detach().cpu().numpy() if hasattr(R_cost, "detach") else np.asarray(R_cost)
    F_np = F_t.detach().cpu().numpy()

    sigma_A_cl = float(np.max(np.abs(np.linalg.eigvals(A_cl_np))))
    M = np.linalg.solve(P_np, A_cl_np.T @ P_np @ A_cl_np)
    rho_true = float(math.sqrt(max(np.max(np.abs(np.linalg.eigvals(M))), 0.0)))
    Q_plus_FRF = Q_np + F_np.T @ R_np @ F_np
    rayleigh_inner = 1.0 - float(np.linalg.eigvalsh(Q_plus_FRF).min()) / lam_max_P
    rho_rayleigh = float(math.sqrt(max(rayleigh_inner, 0.0)))

    # DARE residual: ||AᵀPA − P + Q + FᵀRF||_2 / ||P||_2. Should be at the
    # solver's float64 floor for a DARE-produced P; the Rayleigh bound is
    # only valid in that regime.
    dare_resid = A_cl_np.T @ P_np @ A_cl_np - P_np + Q_plus_FRF
    P_norm = float(np.linalg.norm(P_np, ord=2))
    dare_test = float(np.linalg.norm(dare_resid, ord=2)) / P_norm if P_norm > 0 else float("inf")
    rayleigh_color = _GREEN if dare_test < 1e-6 else _RED

    print(f"--- LQR fit ({label}) ---")
    print(f"  ρ²:                                    {float(rho_sq):.2f}")
    print(f"  κ(P) = λ_max(P)/λ_min(P):              {kappa_P:.2f}")
    print(f"  λ_min(P):                              {lam_min_P:.2e}")
    print(f"  λ_max(P):                              {lam_max_P:.2e}")
    print(f"  ||F||_2 (LQR gain norm):               {F_norm:.2e}")
    if alpha is not None:
        print(f"  α = λ_max(P⁻¹CᵀC):                     {float(alpha):.2e}")
    print(f"  σ(A_cl) (spectral radius):             {sigma_A_cl:.4f}")
    print(f"  ρ_true (P-norm contraction):           {rho_true:.4f}")
    print(f"  DARE Test ||resid||_2 / ||P||_2:       {dare_test:.2e}")
    print(f"{rayleigh_color}  ρ_Rayleigh (DARE bound):               {rho_rayleigh:.4f}{_RESET}")

    result = {
        "label": label,
        "rho_sq": float(rho_sq),
        "kappa_P": float(kappa_P),
        "lambda_min_P": float(lam_min_P),
        "lambda_max_P": float(lam_max_P),
        "F_norm": float(F_norm),
        "sigma_A_cl": sigma_A_cl,
        "rho_true": rho_true,
        "rho_rayleigh": rho_rayleigh,
        "dare_test": dare_test,
    }
    if alpha is not None:
        result["alpha"] = float(alpha)
    return result


def gamma_coverage_report(
    label: str,
    model,
    aug_trajectories,
    device,
    *,
    gamma: float,
    pred_error_stats: dict,
    space: str,
) -> dict:
    """Stats on how well a given max prediction error ``gamma`` covers training data.

    ``space`` selects which one-step-error stream to compare γ against:
      * ``"latent"`` — uses the latent-space R / max / count.
      * ``"state"`` — uses the state-space (decoded) R / max / count.

    Reports three colored stats:
      * ratio of γ to R (mean+2σ) — red <0.75, yellow [0.75, 1.0), green ≥1.0.
      * fraction of training transitions with error ≤ γ — red <0.75,
        yellow [0.75, 0.85], green >0.85.
      * γ as a fraction of the max observed error.
    """
    if space not in ("latent", "state"):
        raise ValueError(f"space must be 'latent' or 'state'; got {space!r}")
    stats = pred_error_stats[space]
    R = float(stats["R"])
    max_err = float(stats["max"])

    ratio_R = gamma / R if R > 0 else float("inf")
    ratio_max = gamma / max_err if max_err > 0 else float("inf")
    count_under, total, frac_under = count_steps_under_threshold(
        model, aug_trajectories, device, gamma, space=space
    )

    c_ratio = _color_ratio(ratio_R)
    c_pct = _color_pct(frac_under)

    print(f"--- γ coverage ({label}, {space} space) ---")
    print(f"  γ:                                     {gamma:.2e}")
    print(f"  R (mean + 2σ):                         {R:.2e}")
    print(f"  max {space} error:                       {max_err:.2e}")
    print(f"{c_ratio}  γ / R:                                 {ratio_R:.2f}{_RESET}")
    print(
        f"{c_pct}  transitions ≤ γ:                       "
        f"{count_under}/{total} ({100.0 * frac_under:.2f}%){_RESET}"
    )
    print(f"  γ / max {space} error:                  {ratio_max:.2f}")

    return {
        "label": label,
        "space": space,
        "gamma": float(gamma),
        "R": R,
        "max_error": max_err,
        "ratio_to_R": float(ratio_R),
        "ratio_to_max": float(ratio_max),
        "frac_transitions_under_gamma": float(frac_under),
        "count_under_gamma": int(count_under),
        "total_transitions": int(total),
    }


def action_budget_report(*, F, B_mat, u_max: float) -> dict:
    """Action-budget diagnostics for the closed-loop latent dynamics.

    Computes:
      * ``||F||_2`` (LQR gain spectral norm).
      * ``||B||_2`` (input matrix spectral norm).
      * ``||B||·||F||`` (the closed-loop input-feedback product).
      * ``u_max`` — actuator limit from the task.
      * ``z_ref_max = u_max / ||F||`` — largest latent reference the
        controller can chase without saturating.
      * ``eta_max = ||B|| · u_max`` — the per-step latent displacement
        induced by a saturated control. This is the η used by the bound
        computations downstream (replaces the legacy config-η).

    Prints a formatted block and returns a dict.
    """
    F_t = torch.as_tensor(F, dtype=torch.float64)
    B_t = torch.as_tensor(B_mat, dtype=torch.float64)

    F_norm = torch.linalg.norm(F_t, ord=2).item()
    B_norm = torch.linalg.norm(B_t, ord=2).item()
    BF_product = B_norm * F_norm
    z_ref_max = (u_max / F_norm) if F_norm > 0 else float("inf")
    eta_max = B_norm * float(u_max)

    print("--- Action budget ---")
    print(f"  ||B||_2:                               {B_norm:.2e}")
    print(f"  ||F||_2:                               {F_norm:.2e}")
    print(f"  ||B|| · ||F||:                         {BF_product:.2e}")
    print(f"  u_max:                                 {float(u_max):.2e}")
    print(f"  z_ref_max = u_max / ||F||:             {z_ref_max:.2e}")
    print(f"  eta_max = ||B|| · u_max:               {eta_max:.2e}")

    return {
        "B_norm": float(B_norm),
        "F_norm": float(F_norm),
        "BF_product": float(BF_product),
        "u_max": float(u_max),
        "z_ref_max": float(z_ref_max),
        "eta_max": float(eta_max),
    }


# ---------------------------------------------------------------------------
# Combined P-generation + γ_max bound.
# ---------------------------------------------------------------------------


def _print_cost_matrices(Q, R_cost, C: np.ndarray) -> None:
    """Pretty-print Q, R, and C as they enter the LQR fit."""
    Q_np = Q.detach().cpu().numpy() if hasattr(Q, "detach") else np.asarray(Q)
    R_np = R_cost.detach().cpu().numpy() if hasattr(R_cost, "detach") else np.asarray(R_cost)
    with np.printoptions(precision=4, suppress=True, linewidth=120):
        print("--- LQR cost matrices ---")
        print(f"  Q ({Q_np.shape[0]}×{Q_np.shape[1]}):")
        for row in Q_np:
            print("    " + np.array2string(row))
        print(f"  R ({R_np.shape[0]}×{R_np.shape[1]}):")
        for row in R_np:
            print("    " + np.array2string(row))
        print(f"  C ({C.shape[0]}×{C.shape[1]}):")
        for row in C:
            print("    " + np.array2string(row))


def _alpha_from_P(P_np: np.ndarray, C: np.ndarray) -> float:
    """α = λ_max(P⁻¹ CᵀC)."""
    CtC = C.T @ C
    return float(np.max(np.linalg.eigvalsh(np.linalg.solve(P_np, CtC))))


def _gamma_no_prepend(epsilon_x, eta, m, rho_sq, kappa_P) -> tuple[float, float]:
    """Return ``(gamma_no_eta, gamma_with_eta)`` for the no-prepend (Lyapunov) branch.

    ``γ_no_eta = (m·ε_x)(1−ρ)/√κ(P)`` is the control-free bound (η=0); the
    config-η version subtracts ``η``. ε is in x-space, scaled by encoder ``m``;
    η is in latent space.
    """
    rho = math.sqrt(max(rho_sq, 0.0))
    gamma_no_eta = (m * epsilon_x) * (1.0 - rho) / math.sqrt(kappa_P)
    return gamma_no_eta, gamma_no_eta - eta


def _gamma_prepend(epsilon_x, eta, rho_sq, alpha, lam_max_P) -> tuple[float, float]:
    """Return ``(gamma_no_eta, gamma_with_eta)`` for the prepend (α-bound) branch.

    ``γ_no_eta = ε_x(1−ρ)/√(α·λ_max(P))`` is the control-free bound; the
    config-η version subtracts ``η``.
    """
    rho = math.sqrt(max(rho_sq, 0.0))
    gamma_no_eta = epsilon_x * (1.0 - rho) / math.sqrt(alpha * lam_max_P)
    return gamma_no_eta, gamma_no_eta - eta


def generate_P_and_bound(
    *,
    model,
    lqr,
    Q,
    R_cost,
    C: np.ndarray,
    epsilon_x: float,
    eta: float,
    m: float,
    optimizer: str,
    beta_rho_target: float = 0.0,
) -> dict:
    """Compute the LQR P and γ_max bound, branching on ``model.prepend_state``.

    Always reports the raw DARE-derived P. ``optimizer`` selects the
    P-optimizer to run on top:

    * ``"none"`` — raw P only.
    * ``"matching"`` — branch-matched optimizer (SDP-Lyapunov for no-prepend,
      L-BFGS α-bound for prepend).
    * ``"beta"`` — level-2 SDP β-bound (both branches; ``γ = ε_x(1-ρ)/√λ_max(P) − η``
      via ``P ⪰ CᵀC``). ``beta_rho_target`` parameterizes the PBH
      detectability pre-check.

    Returns a dict with keys ``raw`` (dict), ``optimized`` (dict or None),
    ``gamma_max`` (float — from optimized if available, else raw),
    ``prepend_state`` (bool), and ``optimizer`` (the active mode).
    """
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"optimizer={optimizer!r} not in {_VALID_OPTIMIZERS}"
        )
    prepend = bool(getattr(model, "prepend_state", False))

    # --- raw LQR P (DARE) ---
    P_raw, kappa_P_raw, rho_sq_raw, P_eigvals_raw = compute_lyapunov_params(lqr, Q, R_cost)
    P_raw_np = P_raw.detach().cpu().numpy()
    lam_max_P_raw = P_eigvals_raw.max().item()

    raw = {
        "P": P_raw_np.tolist(),
        "rho_sq": float(rho_sq_raw),
        "kappa_P": float(kappa_P_raw),
        "lambda_max_P": float(lam_max_P_raw),
    }

    if prepend:
        alpha_raw = _alpha_from_P(P_raw_np, C)
        gamma_raw_no_eta, gamma_raw = _gamma_prepend(
            epsilon_x, eta, rho_sq_raw, alpha_raw, lam_max_P_raw
        )
        raw["alpha"] = alpha_raw
    else:
        gamma_raw_no_eta, gamma_raw = _gamma_no_prepend(
            epsilon_x, eta, m, rho_sq_raw, kappa_P_raw
        )
    raw["gamma_max_no_eta"] = float(gamma_raw_no_eta)
    raw["gamma_max"] = float(gamma_raw)

    optimized: dict | None = None
    gamma_max = gamma_raw
    gamma_max_no_eta = gamma_raw_no_eta

    if optimizer != "none":
        A_cl_np = lqr.closed_loop.detach().cpu().numpy()
        if optimizer == "matching":
            if prepend:
                P_opt, rho_opt, alpha_opt, lam_max_P_opt, gamma_opt = optimize_alpha_P(
                    A_cl_np, C, epsilon_x, eta, P_raw_np
                )
                if P_opt is not None:
                    P_opt_eigs = np.linalg.eigvalsh(P_opt)
                    kappa_P_opt = float(P_opt_eigs.max() / P_opt_eigs.min())
                    gamma_opt_no_eta = float(gamma_opt) + float(eta)
                    optimized = {
                        "P": P_opt.tolist(),
                        "rho_sq": float(rho_opt ** 2),
                        "kappa_P": kappa_P_opt,
                        "lambda_max_P": float(lam_max_P_opt),
                        "alpha": float(alpha_opt),
                        "gamma_max_no_eta": gamma_opt_no_eta,
                        "gamma_max": float(gamma_opt),
                    }
                    gamma_max = float(gamma_opt)
                    gamma_max_no_eta = gamma_opt_no_eta
            else:
                # No-prepend: SDP optimizes over P. ε passed to the optimizer
                # must be in latent space, so apply the m-scaling here.
                P_opt, rho_opt, kappa_P_opt, gamma_opt = optimize_lyapunov_P(
                    A_cl_np, m * epsilon_x, eta
                )
                if P_opt is not None:
                    P_opt_eigs = np.linalg.eigvalsh(P_opt)
                    gamma_opt_no_eta = float(gamma_opt) + float(eta)
                    optimized = {
                        "P": P_opt.tolist(),
                        "rho_sq": float(rho_opt ** 2),
                        "kappa_P": float(kappa_P_opt),
                        "lambda_max_P": float(P_opt_eigs.max()),
                        "gamma_max_no_eta": gamma_opt_no_eta,
                        "gamma_max": float(gamma_opt),
                    }
                    gamma_max = float(gamma_opt)
                    gamma_max_no_eta = gamma_opt_no_eta
        elif optimizer == "beta":
            # Level-2 β-bound: works in both branches. ``epsilon_x`` is
            # treated as output-space (||Ce||) directly; the level-2 bound
            # is ``γ = ε_x(1-ρ)/√λ_max(P) - η`` after β-normalization
            # (P ⪰ CᵀC pins β=1). The detectability pre-check inside
            # optimize_level2_P will raise if any A_cl mode with
            # |λ| > beta_rho_target is not detectable through C.
            P_opt, rho_opt, lam_max_P_opt, gamma_opt = optimize_level2_P(
                A_cl_np, C, epsilon_x, eta, rho_target=beta_rho_target,
            )
            if P_opt is not None:
                P_opt_eigs = np.linalg.eigvalsh(P_opt)
                kappa_P_opt = (
                    float(P_opt_eigs.max() / P_opt_eigs.min())
                    if P_opt_eigs.min() > 0 else float("inf")
                )
                gamma_opt_no_eta = float(gamma_opt) + float(eta)
                optimized = {
                    "P": P_opt.tolist(),
                    "rho_sq": float(rho_opt ** 2),
                    "kappa_P": kappa_P_opt,
                    "lambda_max_P": float(lam_max_P_opt),
                    "alpha": 1.0,  # β-normalized: α_eff = 1/β = 1
                    "gamma_max_no_eta": gamma_opt_no_eta,
                    "gamma_max": float(gamma_opt),
                }
                gamma_max = float(gamma_opt)
                gamma_max_no_eta = gamma_opt_no_eta

    return {
        "prepend_state": prepend,
        "optimizer": optimizer,
        "raw": raw,
        "optimized": optimized,
        "gamma_max": float(gamma_max),
        "gamma_max_no_eta": float(gamma_max_no_eta),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def run_stability_report(
    *,
    model,
    lqr,
    A,
    B_mat,
    Q,
    R_cost,
    C: np.ndarray,
    u_max: float,
    aug_trajectories,
    device,
    epsilon_x: float,
    ctrl_percentages: list[float],
    optimizer: str,
    beta_rho_target: float = 0.0,
    quiet_diagnostics: bool = False,
    encoder_lipschitz_batch_size: int = 4096,
    cached_pred_stats: dict | None = None,
    cached_ctrl_stats: dict | None = None,
    cached_identifiability: dict | None = None,
    n_dense_sweep_points: int = 100,
    mode_projection_groups: dict | None = None,
) -> dict:
    """Full LQR stability + bound report.

    Drives the small helpers in order and returns a consolidated dict
    suitable for YAML persistence.

    * ``quiet_diagnostics=True`` runs the one-step and controllability
      helpers silently (still computes their values).
    * ``cached_pred_stats`` / ``cached_ctrl_stats``: when provided, skip
      the corresponding (re)computation and use the cached values
      directly. Used by the controller-fit step to reuse the dicts
      already saved into ``model_performance.yaml`` during training.
    """
    # Dataset identifiability — purely a function of the dataset + encoder,
    # so it's reused from the koopman-level YAML when present and computed
    # otherwise. Printed before the one-step error block.
    if cached_identifiability is not None:
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  Dataset identifiability (loaded from model_performance.yaml)")
            print("=" * 50)
            _print_identifiability_block(cached_identifiability)
        identifiability_stats = cached_identifiability
    else:
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  Dataset identifiability")
            print("=" * 50)
        identifiability_stats = dataset_identifiability_report(
            model, aug_trajectories, device, verbose=not quiet_diagnostics,
        )

    if cached_pred_stats is not None:
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  One-step prediction error (loaded from model_performance.yaml)")
            print("=" * 50)
            _render_pred_error_report(cached_pred_stats)
        pred_stats = cached_pred_stats
    else:
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  One-step prediction error")
            print("=" * 50)
        pred_stats = one_step_pred_error_report(
            model, aug_trajectories, device, verbose=not quiet_diagnostics
        )

    if cached_ctrl_stats is not None:
        need_recompute = "mode_controllability" not in cached_ctrl_stats or (
            mode_projection_groups is not None
            and not any(
                "mode_projections" in mi
                for mi in cached_ctrl_stats.get("mode_controllability", [])
            )
        )
        if need_recompute:
            # Recompute PBH per-mode info (also picks up the new
            # ``mode_projections`` field when groups are configured).
            _, mode_info, B_singular_values = control_analysis(
                A, B_mat, verbose=False,
                mode_projection_groups=mode_projection_groups,
            )
            cached_ctrl_stats["mode_controllability"] = mode_info
            cached_ctrl_stats.setdefault("B_singular_values", B_singular_values)
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  Controllability fit (loaded from model_performance.yaml)")
            print("=" * 50)
            _render_controllability_fit_report(cached_ctrl_stats)
        ctrl_stats = cached_ctrl_stats
    else:
        if not quiet_diagnostics:
            print("\n" + "=" * 50)
            print("  Controllability fit")
            print("=" * 50)
        ctrl_stats = controllability_fit_report(
            model, A, B_mat, aug_trajectories, device,
            verbose=not quiet_diagnostics,
            encoder_lipschitz_batch_size=encoder_lipschitz_batch_size,
            mode_projection_groups=mode_projection_groups,
        )
    m = ctrl_stats["m"] if ctrl_stats["m"] is not None else 1.0

    # Action budget runs first so its derived quantities (z_ref_max, eta_max)
    # are available for the γ_max bound below.
    print("\n" + "=" * 50)
    print("  Action budget")
    print("=" * 50)
    action_stats = action_budget_report(F=lqr.F, B_mat=B_mat, u_max=u_max)
    eta_max = action_stats["eta_max"]  # ||B|| · u_max

    # γ_max is parameterized by ``η = ctrl_pct · ||B|| · u_max``. We pass
    # η = 0 to the optimizer (its objective subtracts η as a constant, so
    # the optimal P doesn't depend on η) — the per-percentage γ is then
    # just ``gamma_no_eta − η``.
    print("\n" + "=" * 50)
    print("  P and γ_max bound")
    print("=" * 50)
    bound = generate_P_and_bound(
        model=model,
        lqr=lqr,
        Q=Q,
        R_cost=R_cost,
        C=C,
        epsilon_x=epsilon_x,
        eta=0.0,
        m=m,
        optimizer=optimizer,
        beta_rho_target=beta_rho_target,
    )

    print()
    _print_cost_matrices(Q, R_cost, C)
    print()
    lqr_fit_report(
        "raw",
        P=bound["raw"]["P"],
        F=lqr.F,
        A_cl=lqr.closed_loop,
        Q=Q,
        R_cost=R_cost,
        rho_sq=bound["raw"]["rho_sq"],
        alpha=bound["raw"].get("alpha"),
    )
    if bound["optimized"] is not None:
        print()
        lqr_fit_report(
            "optimized",
            P=bound["optimized"]["P"],
            F=lqr.F,
            A_cl=lqr.closed_loop,
            Q=Q,
            R_cost=R_cost,
            rho_sq=bound["optimized"]["rho_sq"],
            alpha=bound["optimized"].get("alpha"),
        )

    # Coverage. The bound is naturally state-space in the prepend branch
    # (ε_x is x-space) and latent-space otherwise. Per request, the
    # latent-space cross-reference printouts are disabled (commented below);
    # natural-space coverage remains.
    gamma_max_no_eta = bound["gamma_max_no_eta"]
    prepend = bound["prepend_state"]
    coverage_space = "state" if prepend else "latent"
    # other_space = "latent" if prepend else "state"  # cross-reference disabled

    # Precompute per-transition errors in the natural coverage space so the
    # dense ctrl_percentage sweep doesn't re-iterate the dataset 100 times.
    errors = _precompute_one_step_errors(
        model, aug_trajectories, device, space=coverage_space
    )
    nat_stats = pred_stats[coverage_space]
    R_nat = float(nat_stats["R"])
    max_nat = float(nat_stats["max"])

    # Per-percentage coverage (the user's selected list).
    ctrl_pct_results: list[dict] = []
    for pct in ctrl_percentages:
        eta_at = float(pct) * float(eta_max)
        gamma_at = float(gamma_max_no_eta) - eta_at
        print()
        print(
            f"=== ctrl_pct = {float(pct):.2f}  →  "
            f"η = pct · ||B||·u_max = {eta_at:.2e}  ===  γ = {gamma_at:.2e}"
        )
        cov = _coverage_from_errors(
            f"γ_ctrl_pct={float(pct):.2f}",
            errors,
            gamma=gamma_at,
            R=R_nat,
            max_err=max_nat,
            space=coverage_space,
        )
        # _coverage_from_errors(  # latent cross-reference (disabled per user request)
        #     f"γ_ctrl_pct={float(pct):.2f}",
        #     _precompute_one_step_errors(model, aug_trajectories, device, space=other_space),
        #     gamma=gamma_at, R=float(pred_stats[other_space]["R"]),
        #     max_err=float(pred_stats[other_space]["max"]), space=other_space,
        # )
        ctrl_pct_results.append({
            "ctrl_percentage": float(pct),
            "eta": eta_at,
            "gamma_max": gamma_at,
            "coverage": cov,
        })

    # ---- Prepend models: also report the *latent-space* alternative bound.
    # When the state is prepended both bounds apply: the standard α-bound
    # already certifies real-state error (above), and γ_latent =
    # ε_x(1−ρ)/√κ(P) certifies the full latent-vector error via the same
    # Lyapunov certificate (m=1 since the prepended state slice is identity).
    ctrl_pct_results_latent: list[dict] = []
    latent_alt_sweep: dict | None = None
    if prepend:
        source = bound["optimized"] if bound["optimized"] is not None else bound["raw"]
        source_label = "optimized" if bound["optimized"] is not None else "raw"
        rho_sq_alt = float(source["rho_sq"])
        kappa_P_alt = float(source["kappa_P"])
        gamma_latent_no_eta, _ = _gamma_no_prepend(
            epsilon_x, 0.0, 1.0, rho_sq_alt, kappa_P_alt
        )

        print()
        print("=" * 50)
        print("  Latent-space alternative bound (prepend models)")
        print("=" * 50)
        print(
            "  Using ε_x as a latent-vector target with the no-prepend "
            "formula γ_latent = ε_x(1−ρ)/√κ(P) (m=1 because the prepended "
            "state slice is identity)."
        )
        print(f"  source P:                              {source_label}")
        print(f"  ρ² (from {source_label}):              {rho_sq_alt:.4f}")
        print(f"  κ(P) (from {source_label}):            {kappa_P_alt:.4e}")
        print(f"  γ_latent (η=0):                        {gamma_latent_no_eta:.4e}")

        # Precompute latent prediction errors and the latent R for coverage.
        latent_errors = _precompute_one_step_errors(
            model, aug_trajectories, device, space="latent"
        )
        latent_stats = pred_stats["latent"]
        R_latent = float(latent_stats["R"])
        max_latent = float(latent_stats["max"])

        for pct in ctrl_percentages:
            eta_at = float(pct) * float(eta_max)
            gamma_at = float(gamma_latent_no_eta) - eta_at
            print()
            print(
                f"=== [latent alt] ctrl_pct = {float(pct):.2f}  →  "
                f"η = {eta_at:.2e}  ===  γ_latent = {gamma_at:.2e}"
            )
            cov_lat = _coverage_from_errors(
                f"γ_latent_ctrl_pct={float(pct):.2f}",
                latent_errors,
                gamma=gamma_at,
                R=R_latent,
                max_err=max_latent,
                space="latent",
            )
            ctrl_pct_results_latent.append({
                "ctrl_percentage": float(pct),
                "eta": eta_at,
                "gamma_max": gamma_at,
                "coverage": cov_lat,
            })

        # Dense latent sweep — mirrors the main sweep below but in latent space.
        sweep_pcts_lat = np.linspace(0.0, 1.0, n_dense_sweep_points)
        sweep_etas_lat: list[float] = []
        sweep_gammas_lat: list[float] = []
        sweep_fracs_lat: list[float] = []
        sweep_ratios_lat: list[float] = []
        total_n_lat = int(latent_errors.numel())
        for pct in sweep_pcts_lat:
            eta_at = float(pct) * float(eta_max)
            gamma_at = float(gamma_latent_no_eta) - eta_at
            count_under = (
                int((latent_errors < gamma_at).sum().item()) if gamma_at > 0 else 0
            )
            frac = count_under / total_n_lat if total_n_lat > 0 else 0.0
            ratio = gamma_at / R_latent if R_latent > 0 else float("inf")
            sweep_etas_lat.append(eta_at)
            sweep_gammas_lat.append(gamma_at)
            sweep_fracs_lat.append(frac)
            sweep_ratios_lat.append(ratio)
        latent_alt_sweep = {
            "ctrl_percentages": [float(p) for p in sweep_pcts_lat],
            "etas": sweep_etas_lat,
            "gamma_max_values": sweep_gammas_lat,
            "frac_transitions_covered": sweep_fracs_lat,
            "gamma_over_R": sweep_ratios_lat,
            "R": R_latent,
            "max_error": max_latent,
            "u_max": float(u_max),
            "eta_max": float(eta_max),
            "space": "latent",
            "source_P": source_label,
            "rho_sq": rho_sq_alt,
            "kappa_P": kappa_P_alt,
            "gamma_max_no_eta": float(gamma_latent_no_eta),
        }

    # Dense sweep for the summary plot — 100 points across the full
    # control-budget range. Uses precomputed errors (no extra iteration).
    sweep_pcts = np.linspace(0.0, 1.0, n_dense_sweep_points)
    sweep_etas: list[float] = []
    sweep_gammas: list[float] = []
    sweep_fracs: list[float] = []
    sweep_ratios: list[float] = []
    total_n = int(errors.numel())
    for pct in sweep_pcts:
        eta_at = float(pct) * float(eta_max)
        gamma_at = float(gamma_max_no_eta) - eta_at
        count_under = int((errors < gamma_at).sum().item()) if gamma_at > 0 else 0
        frac = count_under / total_n if total_n > 0 else 0.0
        ratio = gamma_at / R_nat if R_nat > 0 else float("inf")
        sweep_etas.append(eta_at)
        sweep_gammas.append(gamma_at)
        sweep_fracs.append(frac)
        sweep_ratios.append(ratio)
    sweep = {
        "ctrl_percentages": [float(p) for p in sweep_pcts],
        "etas": sweep_etas,
        "gamma_max_values": sweep_gammas,
        "frac_transitions_covered": sweep_fracs,
        "gamma_over_R": sweep_ratios,
        "R": R_nat,
        "max_error": max_nat,
        "u_max": float(u_max),
        "eta_max": float(eta_max),
        "space": coverage_space,
    }

    Q_list = (
        Q.detach().cpu().numpy() if hasattr(Q, "detach") else np.asarray(Q)
    ).astype(float).tolist()
    R_list = (
        R_cost.detach().cpu().numpy() if hasattr(R_cost, "detach") else np.asarray(R_cost)
    ).astype(float).tolist()
    C_list = C.astype(float).tolist()

    return {
        "dataset_identifiability": identifiability_stats,
        "pred_error_stats": pred_stats,
        "controllability_fit": ctrl_stats,
        "bound": bound,
        "action_budget": action_stats,
        "ctrl_pct_results": ctrl_pct_results,
        "ctrl_pct_sweep": sweep,
        "ctrl_pct_results_latent_alt": ctrl_pct_results_latent if prepend else None,
        "ctrl_pct_sweep_latent_alt": latent_alt_sweep,
        "epsilon_x": float(epsilon_x),
        "m": float(m),
        "optimizer": str(optimizer),
        "prepend_state": bound["prepend_state"],
        "F": lqr.F.detach().cpu().numpy().tolist(),
        "A": A.detach().cpu().numpy().tolist(),
        "B": B_mat.detach().cpu().numpy().tolist(),
        "Q": Q_list,
        "R": R_list,
        "C": C_list,
    }


def _coverage_or_infeasible(
    label, model, aug_trajectories, device, *, gamma, pred_error_stats, space
):
    """Wrap :func:`gamma_coverage_report` with an infeasible-γ short-circuit."""
    if gamma > 0:
        return gamma_coverage_report(
            label, model, aug_trajectories, device,
            gamma=gamma, pred_error_stats=pred_error_stats, space=space,
        )
    print(f"--- γ coverage ({label}, {space} space) ---")
    print(f"{_RED}  γ = {gamma:.2e} ≤ 0 — bound is infeasible.{_RESET}")
    return {
        "label": label,
        "space": space,
        "gamma": float(gamma),
        "R": float(pred_error_stats[space]["R"]),
        "max_error": float(pred_error_stats[space]["max"]),
        "infeasible": True,
    }


def _precompute_one_step_errors(model, aug_trajectories, device, *, space):
    """Return a 1-D tensor of per-transition errors in the given space.

    Used to avoid re-iterating the dataset for every γ check when we sweep
    ``ctrl_percentage`` over many values.
    """
    if space not in ("latent", "state"):
        raise ValueError(f"space must be 'latent' or 'state'; got {space!r}")
    model.to(device)
    model.eval()
    all_errs: list = []
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
    return torch.cat(all_errs)


def _coverage_from_errors(
    label: str,
    errors,
    *,
    gamma: float,
    R: float,
    max_err: float,
    space: str,
    verbose: bool = True,
) -> dict:
    """Same shape as :func:`gamma_coverage_report` but on precomputed errors."""
    total = int(errors.numel())
    if gamma > 0:
        count_under = int((errors < gamma).sum().item())
    else:
        count_under = 0
    frac_under = count_under / total if total > 0 else 0.0
    ratio_R = gamma / R if R > 0 else float("inf")
    ratio_max = gamma / max_err if max_err > 0 else float("inf")

    if verbose:
        if gamma > 0:
            c_ratio = _color_ratio(ratio_R)
            c_pct = _color_pct(frac_under)
            print(f"--- γ coverage ({label}, {space} space) ---")
            print(f"  γ:                                     {gamma:.2e}")
            print(f"  R (mean + 2σ):                         {R:.2e}")
            print(f"  max {space} error:                       {max_err:.2e}")
            print(f"{c_ratio}  γ / R:                                 {ratio_R:.2f}{_RESET}")
            print(
                f"{c_pct}  transitions ≤ γ:                       "
                f"{count_under}/{total} ({100.0 * frac_under:.2f}%){_RESET}"
            )
            print(f"  γ / max {space} error:                  {ratio_max:.2f}")
        else:
            print(f"--- γ coverage ({label}, {space} space) ---")
            print(f"{_RED}  γ = {gamma:.2e} ≤ 0 — bound is infeasible.{_RESET}")

    return {
        "label": label,
        "space": space,
        "gamma": float(gamma),
        "R": float(R),
        "max_error": float(max_err),
        "ratio_to_R": float(ratio_R),
        "ratio_to_max": float(ratio_max),
        "frac_transitions_under_gamma": float(frac_under),
        "count_under": int(count_under),
        "total": int(total),
        "infeasible": bool(gamma <= 0),
    }
