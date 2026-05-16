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
    optimize_lyapunov_P,
)


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


def one_step_pred_error_report(model, aug_trajectories, device) -> dict:
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

    _print_stats_block("One-step latent prediction error (training data)", latent_stats)
    _print_stats_block("One-step state reconstruction error (training data)", state_stats)

    return {
        "latent": latent_stats,
        "state": state_stats,
        "R_latent": latent_stats["R"],
        "R_state": state_stats["R"],
    }


def controllability_fit_report(model, A, B_mat, aug_trajectories, device) -> dict:
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
    print("--- Koopman controllability fit ---")

    m_gx, _L_gx, m_full, _L_full = compute_encoder_lipschitz_bounds(
        model, aug_trajectories, device
    )
    m = m_full if m_full is not None else m_gx
    if m is None:
        print("  m (encoder lower Lipschitz): n/a (fixed encoder)")
    else:
        print(f"  m (encoder lower Lipschitz):           {m:.2e}")

    ctrl_rank = control_analysis(A, B_mat)

    eigvals = torch.linalg.eigvals(A)
    eig_abs = eigvals.abs()
    A_norm = torch.linalg.norm(A, ord=2).item()
    B_norm = torch.linalg.norm(B_mat, ord=2).item()

    print(f"  open-loop |λ| max:                     {eig_abs.max().item():.2f}")
    print(f"  open-loop |λ| min:                     {eig_abs.min().item():.2f}")
    print(f"  ||A||_2 (spectral norm):               {A_norm:.2e}")
    print(f"  ||B||_2 (spectral norm):               {B_norm:.2e}")

    return {
        "m": None if m is None else float(m),
        "ctrl_rank": int(ctrl_rank),
        "latent_dim": int(A.shape[0]),
        "open_loop_eigvals": np.asarray(eigvals.detach().cpu().numpy()).tolist(),
        "A_spec_norm": float(A_norm),
        "B_spec_norm": float(B_norm),
    }


def lqr_fit_report(
    label: str,
    *,
    P,
    F,
    q_scale: float,
    r_scale: float,
    rho_sq: float,
    alpha: float | None = None,
) -> dict:
    """LQR fit quality report.

    Designed to be called twice — once on the raw LQR solution and once
    on any optimized-P refinement — using ``label`` to disambiguate the
    two header lines (rendered as ``--- LQR fit (<label>) ---``).

    Accepts the already-computed quantities directly so the caller controls
    how ``rho_sq`` and ``P`` were derived (e.g. DARE-based vs SDP-optimized).
    ``q_scale`` / ``r_scale`` are the cost-matrix scales straight from
    config (``Q = q_scale * I``, ``R = r_scale * I``).
    """
    P_t = torch.as_tensor(P, dtype=torch.float64)
    F_t = torch.as_tensor(F, dtype=torch.float64)

    P_eigvals = torch.linalg.eigvalsh(P_t)
    lam_min_P = P_eigvals.min().item()
    lam_max_P = P_eigvals.max().item()
    kappa_P = lam_max_P / lam_min_P
    F_norm = torch.linalg.norm(F_t, ord=2).item()

    print(f"--- LQR fit ({label}) ---")
    print(f"  ρ²:                                    {float(rho_sq):.2f}")
    print(f"  κ(P) = λ_max(P)/λ_min(P):              {kappa_P:.2f}")
    print(f"  λ_min(P):                              {lam_min_P:.2e}")
    print(f"  λ_max(P):                              {lam_max_P:.2e}")
    print(f"  ||F||_2 (LQR gain norm):               {F_norm:.2e}")
    print(f"  q_scale:                               {float(q_scale):.2g}")
    print(f"  r_scale:                               {float(r_scale):.2g}")
    if alpha is not None:
        print(f"  α = λ_max(P⁻¹CᵀC):                     {float(alpha):.2e}")

    result = {
        "label": label,
        "rho_sq": float(rho_sq),
        "kappa_P": float(kappa_P),
        "lambda_min_P": float(lam_min_P),
        "lambda_max_P": float(lam_max_P),
        "F_norm": float(F_norm),
        "q_scale": float(q_scale),
        "r_scale": float(r_scale),
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


def action_budget_report(*, F, u_max: float, A) -> dict:
    """Action-budget diagnostics for the closed-loop latent dynamics.

    Computes:
      * ``||F||_2`` (LQR gain spectral norm).
      * ``u_max`` — actuator limit from the task.
      * ``z_ref_limit = u_max / ||F||`` — largest latent reference the
        controller can chase without saturating.
      * ``eta_max = z_ref_limit * ||I + A||`` — maximum possible single-step
        latent displacement induced by tracking a reference at the limit.

    Prints a formatted block and returns a dict.
    """
    F_t = torch.as_tensor(F, dtype=torch.float64)
    A_t = torch.as_tensor(A, dtype=torch.float64)

    F_norm = torch.linalg.norm(F_t, ord=2).item()
    z_ref_limit = (u_max / F_norm) if F_norm > 0 else float("inf")
    I_plus_A_norm = torch.linalg.norm(
        torch.eye(A_t.shape[0], dtype=A_t.dtype) + A_t, ord=2
    ).item()
    eta_max = z_ref_limit * I_plus_A_norm

    print("--- Action budget ---")
    print(f"  ||F||_2:                               {F_norm:.2e}")
    print(f"  u_max:                                 {float(u_max):.2e}")
    print(f"  z_ref_limit = u_max / ||F||:           {z_ref_limit:.2e}")
    print(f"  ||I + A||_2:                           {I_plus_A_norm:.2e}")
    print(f"  eta_max = z_ref_limit * ||I + A||:     {eta_max:.2e}")

    return {
        "F_norm": float(F_norm),
        "u_max": float(u_max),
        "z_ref_limit": float(z_ref_limit),
        "I_plus_A_norm": float(I_plus_A_norm),
        "eta_max": float(eta_max),
    }


# ---------------------------------------------------------------------------
# Combined P-generation + γ_max bound.
# ---------------------------------------------------------------------------


def _alpha_from_P(P_np: np.ndarray, real_state_dim: int) -> float:
    """α = λ_max(P⁻¹ CᵀC) with C = [I_p | 0]."""
    n = P_np.shape[0]
    C = np.zeros((real_state_dim, n), dtype=np.float64)
    C[: real_state_dim, : real_state_dim] = np.eye(real_state_dim)
    CtC = C.T @ C
    return float(np.max(np.linalg.eigvalsh(np.linalg.solve(P_np.astype(np.float64), CtC))))


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
    real_state_dim: int,
    epsilon_x: float,
    eta: float,
    m: float,
    use_optimization: bool,
) -> dict:
    """Compute the LQR P and γ_max bound, branching on ``model.prepend_state``.

    Always reports the raw DARE-derived P. When ``use_optimization`` is set,
    additionally runs the optimizer matching the active branch (SDP for
    no-prepend, L-BFGS α-optimizer for prepend) and reports those results.

    Returns a dict with keys ``raw`` (dict), ``optimized`` (dict or None),
    ``gamma_max`` (float — from optimized if available, else raw), and
    ``prepend_state`` (bool).
    """
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
        alpha_raw = _alpha_from_P(P_raw_np, real_state_dim)
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

    if use_optimization:
        A_cl_np = lqr.closed_loop.detach().cpu().numpy().astype(np.float64)
        if prepend:
            n = A_cl_np.shape[0]
            C = np.zeros((real_state_dim, n), dtype=np.float64)
            C[: real_state_dim, : real_state_dim] = np.eye(real_state_dim)
            P_opt, rho_opt, alpha_opt, lam_max_P_opt, gamma_opt = optimize_alpha_P(
                A_cl_np, C, epsilon_x, eta, P_raw_np.astype(np.float64)
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

    return {
        "prepend_state": prepend,
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
    real_state_dim: int,
    u_max: float,
    aug_trajectories,
    device,
    epsilon_x: float,
    eta: float,
    q_scale: float,
    r_scale: float,
    use_optimization: bool,
) -> dict:
    """Full LQR stability + bound report.

    Drives the small helpers in order and returns a consolidated dict
    suitable for YAML persistence.
    """
    print("\n" + "=" * 50)
    print("  One-step prediction error")
    print("=" * 50)
    pred_stats = one_step_pred_error_report(model, aug_trajectories, device)

    print("\n" + "=" * 50)
    print("  Controllability fit")
    print("=" * 50)
    ctrl_stats = controllability_fit_report(model, A, B_mat, aug_trajectories, device)
    m = ctrl_stats["m"] if ctrl_stats["m"] is not None else 1.0

    print("\n" + "=" * 50)
    print("  P and γ_max bound")
    print("=" * 50)
    bound = generate_P_and_bound(
        model=model,
        lqr=lqr,
        Q=Q,
        R_cost=R_cost,
        real_state_dim=real_state_dim,
        epsilon_x=epsilon_x,
        eta=eta,
        m=m,
        use_optimization=use_optimization,
    )

    print()
    lqr_fit_report(
        "raw",
        P=bound["raw"]["P"],
        F=lqr.F,
        q_scale=q_scale,
        r_scale=r_scale,
        rho_sq=bound["raw"]["rho_sq"],
        alpha=bound["raw"].get("alpha"),
    )
    if bound["optimized"] is not None:
        print()
        lqr_fit_report(
            "optimized",
            P=bound["optimized"]["P"],
            F=lqr.F,
            q_scale=q_scale,
            r_scale=r_scale,
            rho_sq=bound["optimized"]["rho_sq"],
            alpha=bound["optimized"].get("alpha"),
        )

    print()
    action_stats = action_budget_report(F=lqr.F, u_max=u_max, A=A)

    # γ values: control-free (η=0) and config-η. For the prepend branch, the
    # bound is naturally a state-space ε_x quantity, so coverage is computed
    # against the state-space one-step error (legacy ``alpha_r_space="state"``);
    # we also print the latent-space coverage for visibility. The no-prepend
    # branch is naturally latent, so we report latent there.
    gamma_max = bound["gamma_max"]
    gamma_max_no_eta = bound["gamma_max_no_eta"]
    prepend = bound["prepend_state"]
    coverage_space = "state" if prepend else "latent"
    other_space = "latent" if prepend else "state"

    print()
    print(f"=== γ control-free (η=0) coverage ===  γ = {gamma_max_no_eta:.2e}")
    coverage_no_eta = _coverage_or_infeasible(
        "γ_no_eta",
        model,
        aug_trajectories,
        device,
        gamma=gamma_max_no_eta,
        pred_error_stats=pred_stats,
        space=coverage_space,
    )
    # Also show the other space for cross-reference (informational).
    _coverage_or_infeasible(
        "γ_no_eta",
        model,
        aug_trajectories,
        device,
        gamma=gamma_max_no_eta,
        pred_error_stats=pred_stats,
        space=other_space,
    )

    print()
    print(f"=== γ_max (config η) coverage ===  γ = {gamma_max:.2e}")
    coverage = _coverage_or_infeasible(
        "γ_max",
        model,
        aug_trajectories,
        device,
        gamma=gamma_max,
        pred_error_stats=pred_stats,
        space=coverage_space,
    )
    _coverage_or_infeasible(
        "γ_max",
        model,
        aug_trajectories,
        device,
        gamma=gamma_max,
        pred_error_stats=pred_stats,
        space=other_space,
    )

    return {
        "pred_error_stats": pred_stats,
        "controllability_fit": ctrl_stats,
        "bound": bound,
        "action_budget": action_stats,
        "gamma_coverage": coverage,
        "gamma_coverage_no_eta": coverage_no_eta,
        "epsilon_x": float(epsilon_x),
        "eta": float(eta),
        "m": float(m),
        "use_optimization": bool(use_optimization),
        "prepend_state": bound["prepend_state"],
        "F": lqr.F.detach().cpu().numpy().tolist(),
        "A": A.detach().cpu().numpy().tolist(),
        "B": B_mat.detach().cpu().numpy().tolist(),
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
