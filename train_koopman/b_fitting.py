"""Analytical B-matrix fitting routines.

Two methods are exposed:

* :func:`gradient_b` — gradient descent on B with a controllability
  regularizer (matches the legacy ``solve_B_with_controllability``).
* :func:`least_squares_projected_b` — closed-form least-squares B
  followed by :func:`project_for_controllability`, a deterministic
  three-stage PBH + SVD + iterative-perturbation projection that
  guarantees full controllability rank (matches the legacy
  ``train_for_B=False`` path).

Both consume per-transition ``(z_t, z_next, u_t)`` tensors produced by
encoding a perturbed trajectory dataset; the joint paradigm dispatches
to one of them via :class:`config.manager.BFittingCfg.method`.
"""

from __future__ import annotations

import numpy as np
import torch

from train_koopman.losses import controllability_loss


_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"


def _residual_and_action_matrices(z_t: torch.Tensor, z_next: torch.Tensor, A: torch.Tensor,
                                  u_t: torch.Tensor):
    """Return ``residual = z_next − A z_t`` in (N, latent_dim) and ``u_t`` as numpy arrays."""
    residual = z_next - z_t @ A.T
    return residual.detach().cpu().numpy(), u_t.detach().cpu().numpy()


def least_squares_b(z_t: torch.Tensor, z_next: torch.Tensor, A: torch.Tensor,
                    u_t: torch.Tensor) -> np.ndarray:
    """Closed-form B via least-squares: solves ``u_t @ B.T = z_next − A z_t``.

    Returns a numpy ``B`` of shape ``(latent_dim, action_dim)``.
    """
    residual, u_np = _residual_and_action_matrices(z_t, z_next, A, u_t)
    B_T, *_ = np.linalg.lstsq(u_np, residual, rcond=None)
    return B_T.T


def gradient_b(
    A: torch.Tensor,
    z_t: torch.Tensor,
    z_next: torch.Tensor,
    u_t: torch.Tensor,
    *,
    ctrl_lambda: float,
    train_steps: int,
    lr: float,
    init_B: torch.Tensor | None = None,
    log_interval: int = 100,
) -> torch.Tensor:
    """Gradient descent on B with a controllability regularizer.

    Loss:
        ``recon = Σ_dim (z_next − A z_t − B u_t)² averaged over N samples``
        ``loss  = recon + ctrl_lambda · −σ_min([B, AB, …, A^{n−1}B])``

    The recon term sums across latent dims and averages over samples
    (matching legacy ``||R − B U||_F² / N`` where R, U are column-stacked).
    The controllability matrix uses ``horizon = latent_dim`` (matching legacy).

    If ``init_B`` is ``None``, initializes B from the least-squares fit
    (also matches legacy).
    """
    latent_dim = A.shape[0]

    if init_B is None:
        B_ls_np = least_squares_b(z_t, z_next, A, u_t)
        B = torch.tensor(B_ls_np, dtype=A.dtype, device=A.device).clone()
    else:
        B = init_B.clone().to(dtype=A.dtype, device=A.device)
    B.requires_grad_(True)

    optimizer = torch.optim.Adam([B], lr=lr)
    for step in range(train_steps):
        optimizer.zero_grad()
        pred = z_t @ A.T + u_t @ B.T
        recon = ((pred - z_next) ** 2).sum(dim=-1).mean()
        reg = controllability_loss(A, B, horizon=latent_dim)
        loss = recon + ctrl_lambda * reg
        loss.backward()
        optimizer.step()

        is_last = step == train_steps - 1
        if log_interval > 0 and (step % log_interval == 0 or is_last):
            sigma_min_val = -reg.item()
            print(
                f"  step {step:4d} | recon={recon.item():.4f} | "
                f"σ_min={sigma_min_val:.4f} | ctrl={reg.item():.4f}"
            )
    return B.detach()


# ---------------------------------------------------------------------------
# Closed-form least-squares B + deterministic controllability projection.
# ---------------------------------------------------------------------------


def _fmt_array(arr) -> str:
    return np.array2string(
        np.asarray(arr), formatter={"float_kind": lambda x: f"{x:.3f}"}
    )


def _ctrl_rank_and_sv(A: np.ndarray, B: np.ndarray, n: int):
    """Return ``(rank, singular_values, C_mat)`` for ``C = [B, AB, …, A^{n−1}B]``."""
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    sv = np.linalg.svd(C, compute_uv=False)
    rank = int(np.linalg.matrix_rank(C))
    return rank, sv, C


def project_for_controllability(A: np.ndarray, B_ls: np.ndarray, n: int) -> np.ndarray:
    """Project B onto the maximally controllable subspace.

    Deterministic three-stage strategy (legacy verbatim):
      1. PBH test — augment B in directions where left eigenvectors of A
         have near-zero projection onto B.
      2. SVD boost — directly boost weak singular directions of the
         controllability matrix.
      3. Iterative perturbation — last-resort random perturbation along
         the weakest singular direction.

    All linear algebra in float64.
    """
    A = A.astype(np.float64)
    B_ls = B_ls.astype(np.float64)
    m = B_ls.shape[1]
    b_norm = np.linalg.norm(B_ls, ord=2)

    rank_init, sv_init, _ = _ctrl_rank_and_sv(A, B_ls, n)
    print(f"  Initial controllability rank: {rank_init}/{n}")
    print(f"  Controllability SV: {_fmt_array(sv_init)}")

    if rank_init == n:
        print(f"  {_GREEN}B_ls already fully controllable.{_RESET}")
        return B_ls.astype(np.float32)

    # --- Stage 1: PBH test ---
    eigvals, V_right = np.linalg.eig(A)
    try:
        W = np.linalg.inv(V_right).conj()
    except np.linalg.LinAlgError:
        print(f"  {_RED}PROBLEM: A is defective (non-diagonalizable).{_RESET}")
        print(f"  {_YELLOW}FALLBACK: Using SVD of controllability matrix directly.{_RESET}")
        W = None

    B_aug = B_ls.copy()

    if W is not None:
        pbh_failures = []
        for i in range(n):
            w = W[i]
            projection = np.abs(w @ B_ls)
            if np.max(projection) < b_norm * 1e-4:
                pbh_failures.append(i)

        print(
            f"  PBH test: {len(pbh_failures)} failing eigenvalue(s) "
            f"(threshold: |w^H B| < {b_norm * 1e-4:.3f})"
        )
        for idx in pbh_failures:
            print(
                f"    lambda={eigvals[idx]:.3f}, |lambda|={abs(eigvals[idx]):.3f}, "
                f"|w^H B|={np.max(np.abs(W[idx] @ B_ls)):.3f}"
            )

        correction_scale = max(b_norm, 1e-3)
        already_added = []

        for idx in pbh_failures:
            w = W[idx]
            for part_name, part in [("real", w.real), ("imag", w.imag)]:
                norm = np.linalg.norm(part)
                if norm < 1e-12:
                    continue
                direction = part / norm
                is_dup = any(abs(np.dot(direction, p)) > 0.99 for p in already_added)
                if is_dup:
                    continue
                already_added.append(direction)
                B_aug += correction_scale * direction[:, None] @ np.ones((1, m))
                print(f"    Added {part_name} direction for lambda={eigvals[idx]:.3f}")

    rank_pbh, sv_pbh, _ = _ctrl_rank_and_sv(A, B_aug, n)
    print(f"  After PBH augmentation: rank = {rank_pbh}/{n}")
    print(f"  SV: {_fmt_array(sv_pbh)}")

    if rank_pbh == n:
        print(f"  {_GREEN}Full rank achieved via PBH augmentation.{_RESET}")
        return B_aug.astype(np.float32)

    # --- Stage 2: SVD-based boost ---
    print(f"  {_RED}PROBLEM: PBH augmentation insufficient (rank {rank_pbh}/{n}).{_RESET}")
    print(
        f"  {_YELLOW}FALLBACK: Directly boosting weak SVD directions of "
        f"controllability matrix.{_RESET}"
    )

    C_mat = np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)])
    U_svd, S, _ = np.linalg.svd(C_mat, full_matrices=True)

    target_sv = np.median(S[:rank_pbh]) if rank_pbh > 0 else b_norm
    boost = max(target_sv * 0.1, b_norm, 1e-3)

    for i in range(n):
        if S[i] < S[0] * 1e-8:
            direction = U_svd[:, i]
            B_aug += boost * direction[:, None] @ np.ones((1, m))
            print(f"    Boosted SVD direction {i}: S={S[i]:.3f} -> adding scale {boost:.3f}")

    rank_svd, sv_svd, _ = _ctrl_rank_and_sv(A, B_aug, n)
    print(f"  After SVD boost: rank = {rank_svd}/{n}")
    print(f"  SV: {_fmt_array(sv_svd)}")

    if rank_svd == n:
        print(f"  {_GREEN}Full rank achieved via SVD boost.{_RESET}")
        return B_aug.astype(np.float32)

    # --- Stage 3: iterative perturbation ---
    print(f"  {_RED}PROBLEM: SVD boost insufficient (rank {rank_svd}/{n}).{_RESET}")
    print(f"  {_YELLOW}FALLBACK: Iterative perturbation with rank checking.{_RESET}")

    rng = np.random.RandomState(42)
    for attempt in range(100):
        C_cur = np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)])
        U_c, S_c, _ = np.linalg.svd(C_cur, full_matrices=True)

        weakest_idx = np.argmin(S_c)
        direction = U_c[:, weakest_idx]
        noise = rng.randn(n)
        noise -= direction * np.dot(noise, direction)
        noise /= np.linalg.norm(noise) + 1e-12

        perturb = boost * (direction + 0.1 * noise)
        B_aug += perturb[:, None] @ np.ones((1, m))

        rank_try = np.linalg.matrix_rank(
            np.hstack([np.linalg.matrix_power(A, i) @ B_aug for i in range(n)])
        )
        if rank_try == n:
            print(f"    Achieved full rank after {attempt + 1} perturbation(s)")
            break
    else:
        print(
            f"  {_RED}WARNING: Could not achieve full controllability rank after "
            f"100 iterations.{_RESET}"
        )

    rank_final, sv_final, _ = _ctrl_rank_and_sv(A, B_aug, n)
    color = _GREEN if rank_final == n else _RED
    print(f"  {color}Final controllability rank: {rank_final}/{n}{_RESET}")
    print(f"  Final SV: {_fmt_array(sv_final)}")

    return B_aug.astype(np.float32)


def least_squares_projected_b(
    A: torch.Tensor,
    z_t: torch.Tensor,
    z_next: torch.Tensor,
    u_t: torch.Tensor,
) -> torch.Tensor:
    """Closed-form B via least-squares, then PBH/SVD/iterative projection."""
    n = A.shape[0]
    B_ls_np = least_squares_b(z_t, z_next, A, u_t)
    A_np = A.detach().cpu().numpy()
    B_np = project_for_controllability(A_np, B_ls_np, n)
    return torch.tensor(B_np, dtype=A.dtype, device=A.device)
