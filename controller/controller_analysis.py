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


def control_analysis(
    A,
    B_mat,
    *,
    verbose: bool = True,
    mode_projection_groups: dict | None = None,
):
    """Controllability diagnostics for ``(A, B)``.

    Returns ``(ctrl_rank, mode_info, B_singular_values)``:

    * ``ctrl_rank`` — rank of the Kalman controllability matrix.
    * ``mode_info`` — list of dicts (one per A eigenvalue, ordered as
      returned by ``scipy.linalg.eig``) with::

          {
              "eigenvalue":      [real, imag],
              "magnitude":       |λ|,
              "pbh_projection":  ||wᵀ B||  (PBH controllability test),
              "controllable":    bool (pbh_projection > tol · ||w||),
              "mode_projections": {group_name: ||v[idx]||/||v||, ...}
                  (only when ``mode_projection_groups`` is set; ``v`` is
                  the right eigenvector of A — shows Koopman mode
                  structure).
          }

      Per the PBH test, mode ``i`` is uncontrollable iff its left eigenvector
      ``w_i`` (satisfying ``w_iᵀ A = λ_i w_iᵀ``) annihilates ``B``.
    * ``B_singular_values`` — full SVD of B (the input-coupling "modes";
      B is rectangular so it doesn't have eigenvalues directly).

    ``mode_projection_groups``: optional ``{name: [indices]}`` map. When
    provided, the right-eigenvector projection ``||v[indices]|| / ||v||``
    is computed for each group and stored under ``mode_projections``; the
    near-unit-circle block also prints the breakdown.

    When ``verbose``, prints rank, every eigenvalue with its controllability
    flag, and the B singular values.
    """
    from scipy.linalg import eig as scipy_eig

    A_np = np.asarray(A.detach().cpu().numpy() if hasattr(A, "detach") else A)
    B_np = np.asarray(B_mat.detach().cpu().numpy() if hasattr(B_mat, "detach") else B_mat)

    # Kalman controllability matrix [B, AB, A²B, ...]
    C_mat = np.hstack(
        [np.linalg.matrix_power(A_np, i) @ B_np for i in range(A_np.shape[0])]
    )
    ctrl_rank = int(np.linalg.matrix_rank(C_mat))

    # PBH test per eigenvalue; keep right eigenvectors for the Koopman
    # mode-projection breakdown.
    eigvals, V_left, V_right = scipy_eig(A_np, left=True, right=True)
    mode_info = []
    for i, lam in enumerate(eigvals):
        w = V_left[:, i].conj()                       # left eigenvector (row)
        w_norm = float(np.linalg.norm(w))
        proj_norm = float(np.linalg.norm(w @ B_np))
        # Relative threshold guards against scale of w (eigenvectors aren't
        # unique up to scale, but scipy normalizes; 1e-8 · ||w|| is generous).
        controllable = proj_norm > 1e-8 * max(w_norm, 1.0)
        entry = {
            "eigenvalue": [float(lam.real), float(lam.imag)],
            "magnitude": float(abs(lam)),
            "pbh_projection": proj_norm,
            "controllable": bool(controllable),
        }
        if mode_projection_groups:
            v = V_right[:, i]
            v_norm = float(np.linalg.norm(v))
            projections = {}
            for name, idx in mode_projection_groups.items():
                projections[str(name)] = (
                    float(np.linalg.norm(v[idx]) / v_norm) if v_norm > 0 else float("nan")
                )
            entry["mode_projections"] = projections
        mode_info.append(entry)

    B_singular_values = [float(s) for s in np.linalg.svd(B_np, compute_uv=False)]

    if verbose:
        print(f"  Controllability rank:                  {ctrl_rank} / {A_np.shape[0]}")
        print("  --- All eigenvalues of A (PBH controllability) ---")
        for i, m in enumerate(mode_info):
            re, im = m["eigenvalue"]
            mag = m["magnitude"]
            proj = m["pbh_projection"]
            tag = "controllable" if m["controllable"] else "UNCONTROLLABLE"
            print(
                f"    λ_{i:02d} = {re:+.4f}{im:+.4f}j  "
                f"|λ|={mag:.4f}  |wᵀB|={proj:.2e}  [{tag}]"
            )
        slow = [(i, m) for i, m in enumerate(mode_info) if m["magnitude"] > 0.9]
        if slow:
            print("  --- Near-unit-circle modes (|λ| > 0.9) ---")
            for i, m in slow:
                re, im = m["eigenvalue"]
                print(
                    f"    λ_{i:02d} = {re:+.4f}{im:+.4f}j  "
                    f"|λ|={m['magnitude']:.4f}  |wᵀB|={m['pbh_projection']:.2e}"
                )
            if mode_projection_groups:
                print(
                    "  --- Near-unit-circle A modes (|λ| > 0.9): "
                    "Koopman mode projections ---"
                )
                for i, m in slow:
                    re, im = m["eigenvalue"]
                    parts = "  ".join(
                        f"{name}={val:.3f}" for name, val in m["mode_projections"].items()
                    )
                    print(
                        f"    λ_{i:02d} = {re:+.4f}{im:+.4f}j  |λ|={m['magnitude']:.4f}  {parts}"
                    )
        print("  --- Singular values of B ---")
        for i, s in enumerate(B_singular_values):
            print(f"    σ_{i}(B) = {s:.4e}")

    return ctrl_rank, mode_info, B_singular_values


# ---------------------------------------------------------------------------
# Encoder Lipschitz bounds.
# ---------------------------------------------------------------------------


def compute_encoder_lipschitz(
    encoder,
    training_data,
    *,
    verbose: bool = True,
    batch_size: int = 4096,
):
    """Lower / upper Lipschitz constants of ``encoder`` over ``training_data``.

    Computes per-point Jacobians with ``vmap(jacrev(...))`` and keeps the min
    and max singular values across all points. The Jacobian batch is sized
    naïvely ``len(training_data) × output_dim × hidden_dim`` floats, which can
    easily exceed RAM for large training sets — so we chunk the inputs into
    ``batch_size`` slices and accumulate the per-point singular values.
    """
    X = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in training_data])

    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)

    jac_fn = vmap(jacrev(encode_single))

    sigma_min_chunks: list[torch.Tensor] = []
    sigma_max_chunks: list[torch.Tensor] = []
    for start in range(0, len(X), batch_size):
        chunk = X[start : start + batch_size]
        J = jac_fn(chunk)                              # (b, out, in)
        svd = torch.linalg.svdvals(J)                  # (b, min(out, in))
        sigma_min_chunks.append(svd[:, -1].detach())
        sigma_max_chunks.append(svd[:, 0].detach())
    sigma_mins = torch.cat(sigma_min_chunks)
    sigma_maxs = torch.cat(sigma_max_chunks)

    m = float(sigma_mins.min())
    L = float(sigma_maxs.max())
    if verbose:
        print(
            f"  σ_min distribution ({len(sigma_mins)} points): "
            f"min={sigma_mins.min():.2f}  p1={sigma_mins.quantile(0.01):.2f}  "
            f"p5={sigma_mins.quantile(0.05):.2f}  median={sigma_mins.median():.2f}  "
            f"mean={sigma_mins.mean():.2f}"
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


def compute_encoder_lipschitz_bounds(
    model, aug_trajectories, device, *, verbose: bool = True, batch_size: int = 4096
):
    """Compute Lipschitz bounds for both ``g(x)`` and the full ``encode`` map."""
    if getattr(model, "_trig_encoder", False) or model.encoder is None:
        if verbose:
            print("  Lipschitz bounds: skipped (fixed encoder)")
        return None, None, None, None

    model_cpu = model.cpu()
    training_states = []
    for states, actions in aug_trajectories:
        for s in states:
            training_states.append(s)

    if model_cpu.prepend_state:
        if verbose:
            print("  --- g(x) encoder only ---")
        m_gx, L_gx = compute_encoder_lipschitz(
            model_cpu.encoder, training_states, verbose=verbose, batch_size=batch_size
        )
        if verbose:
            print("  --- full encode [x; g(x)] ---")
        m_full, L_full = compute_encoder_lipschitz(
            model_cpu.encode, training_states, verbose=verbose, batch_size=batch_size
        )
    else:
        m_gx, L_gx = compute_encoder_lipschitz(
            model_cpu.encode, training_states, verbose=verbose, batch_size=batch_size
        )
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


def compute_max_latent_diff_observed(model, aug_trajectories, device) -> float:
    """Latent-space diameter approximated from the encoded training data.

    Encodes every state in ``aug_trajectories``, takes the per-dimension
    min and max across the resulting latent vectors, and returns the
    Euclidean distance between those corners. Env-agnostic — no hand-picked
    extremes.
    """
    model.to(device)
    model.eval()
    z_min = None
    z_max = None
    with torch.no_grad():
        for states, _actions in aug_trajectories:
            z = model.encode(
                torch.tensor(states, dtype=torch.float32, device=device)
            )
            if z_min is None:
                z_min = z.min(dim=0).values
                z_max = z.max(dim=0).values
            else:
                z_min = torch.minimum(z_min, z.min(dim=0).values)
                z_max = torch.maximum(z_max, z.max(dim=0).values)
    return torch.linalg.norm(z_max - z_min).item()


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


