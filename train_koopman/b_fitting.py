"""B-matrix fitting routines used independently of end-to-end training.

The two-phase and joint paradigms train B via the shared ``train()`` loop in
:mod:`train_koopman.training_loop`. These standalone fitters are used when you
have a pre-trained Koopman A and need B without re-running the full pipeline
(e.g. from the LQR-controller entry point).
"""

from __future__ import annotations

import numpy as np
import torch

from train_koopman.losses import controllability_loss


def least_squares_b(
    z_t: np.ndarray, z_next: np.ndarray, A: np.ndarray, u_t: np.ndarray
) -> np.ndarray:
    """Closed-form B via pseudoinverse: ``B = (Z_{t+1} - A Z_t) · U_t^+``.

    All inputs and the returned B are numpy arrays.
    """
    residual = z_next - z_t @ A.T  # (N, latent_dim)
    B_T, *_ = np.linalg.lstsq(u_t, residual, rcond=None)
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
) -> torch.Tensor:
    """Gradient descent on B with controllability regularization.

    Loss: ``MSE(z_t @ A.T + u_t @ B.T, z_next) + ctrl_lambda · controllability_loss(A, B)``.
    Returns the optimized B tensor.
    """
    latent_dim = A.shape[0]
    action_dim = u_t.shape[-1]
    if init_B is None:
        B = torch.zeros(latent_dim, action_dim, dtype=A.dtype, device=A.device)
    else:
        B = init_B.clone().to(dtype=A.dtype, device=A.device)
    B.requires_grad_(True)

    optimizer = torch.optim.Adam([B], lr=lr)
    for step in range(train_steps):
        optimizer.zero_grad()
        pred = z_t @ A.T + u_t @ B.T
        mse = ((pred - z_next) ** 2).mean()
        reg = controllability_loss(A, B)
        loss = mse + ctrl_lambda * reg
        loss.backward()
        optimizer.step()
    return B.detach()
