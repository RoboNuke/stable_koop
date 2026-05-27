"""Smoke test for the unified LQR stability analysis (both branches).

Run:
    python -m tests.smoke_lqr_analysisv2

Builds a toy Koopman-like model with a Linear encoder + linear latent
dynamics, generates a tiny synthetic trajectory dataset, then drives the
full ``run_stability_report`` pipeline so every helper prints:
``one_step_pred_error_report`` → ``controllability_fit_report`` →
``generate_P_and_bound`` → ``lqr_fit_report`` (raw [+ optimized]) →
``action_budget_report`` → ``gamma_coverage_report``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from config.manager import LQRControllerCfg
from controller.lqr.lqr import LQR
from controller.lqr.lqr_analysis import build_C, setup_lqr
from controller.lqr.lqr_analysisv2 import run_stability_report


class ToyKoopman(nn.Module):
    """Tiny Koopman-like model: Linear encoder, linear latent dynamics."""

    def __init__(self, state_dim: int, latent_dim: int, action_dim: int, prepend_state: bool):
        super().__init__()
        self.prepend_state = prepend_state
        self._trig_encoder = False

        # encoder: state_dim -> latent_dim (with prepend_state, full encode is [x; g(x)])
        g_dim = latent_dim - state_dim if prepend_state else latent_dim
        self.encoder = nn.Linear(state_dim, g_dim, bias=False)
        nn.init.orthogonal_(self.encoder.weight)

        # latent dynamics: z_{t+1} = A z_t + B u_t  (stable A by construction)
        rng = torch.Generator().manual_seed(0)
        A = torch.randn(latent_dim, latent_dim, generator=rng) * 0.2
        eig = torch.linalg.eigvals(A).abs().max().item()
        if eig >= 0.9:
            A = A * (0.85 / eig)
        self.register_buffer("_A", A)
        self.register_buffer("_B", torch.randn(latent_dim, action_dim, generator=rng) * 0.5)

    @property
    def A(self):
        return self._A

    @property
    def B_matrix(self):
        return self._B

    def encode(self, x):
        gx = self.encoder(x)
        if self.prepend_state:
            return torch.cat([x, gx], dim=-1)
        return gx

    def predict(self, z, u):
        return z @ self._A.T + u @ self._B.T

    def decode(self, z):
        # Trivial: take the first state_dim entries (works because prepend_state).
        # For non-prepend, this won't be exact, but decode is unused by the helpers we test.
        return z[..., : self.encoder.in_features]


def _synthetic_trajectories(
    n_traj: int = 4,
    horizon: int = 20,
    state_dim: int = 2,
    action_dim: int = 1,
    seed: int = 0,
):
    """List of (states, actions) numpy pairs in the shape expected by the helpers."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traj):
        states = rng.standard_normal((horizon + 1, state_dim)).astype(np.float32) * 0.3
        actions = rng.standard_normal((horizon, action_dim)).astype(np.float32) * 0.2
        out.append((states, actions))
    return out


def _run_branch(*, prepend_state: bool) -> None:
    state_dim = 2
    latent_dim = 4
    action_dim = 1
    device = torch.device("cpu")

    model = ToyKoopman(state_dim, latent_dim, action_dim, prepend_state=prepend_state).to(device)
    aug_trajectories = _synthetic_trajectories(state_dim=state_dim, action_dim=action_dim)

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    # Q regularized to full rank (ρ² < 1) by giving every latent dim a positive
    # weight; R is a 0.1 diagonal on a 1-D action. C_mask picks the first
    # state_dim latent dims as the output (= the "real" state slice).
    c_mask = [1] * state_dim + [0] * (latent_dim - state_dim)
    lqr_cfg = LQRControllerCfg(
        Q_diag=[1.0] * latent_dim,
        R_diag=[0.1] * action_dim,
        C_mask=c_mask,
        scale_B=False,
    )

    label = "prepend" if prepend_state else "no-prepend"
    print("\n" + "#" * 70)
    print(f"#  BRANCH: {label}")
    print("#" * 70)

    lqr, Q, R_cost, B_scale = setup_lqr(
        A, B_mat, lqr_cfg, action_dim=action_dim
    )
    C_np = build_C(lqr_cfg.C_mask, latent_dim=A.shape[0])

    run_stability_report(
        model=model,
        lqr=lqr,
        A=A,
        B_mat=B_mat,
        Q=Q,
        R_cost=R_cost,
        C=C_np,
        u_max=2.0,
        aug_trajectories=aug_trajectories,
        device=device,
        epsilon_x=5.0,   # generous so γ_max comes out positive for the toy
        ctrl_percentages=[0.0, 0.5, 1.0],
        optimizer="none",
    )


def main() -> None:
    _run_branch(prepend_state=False)
    _run_branch(prepend_state=True)


if __name__ == "__main__":
    main()
