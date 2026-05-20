"""LQR policy for InvertedPendulum-v4/v5.

Solves the discrete-time DARE around the upright equilibrium using the
rod-pole linearization of the cartpole dynamics, with MuJoCo's physical
parameters hardcoded (read from ``env.unwrapped.model``; see
``eval/inverted_pendulum_metrics.py`` for matching values).

Dynamics, rod pole (capsule), state ``x = [cart_x, cart_xdot, θ, θ̇]``::

    ẍ_c  = -(3 m g)/(4 M_eff) · θ      + (1/M_eff) · F
    θ̈    =  (3 g (M+m))/(4 L M_eff) · θ + (-3/(4 L M_eff)) · F

with ``M_eff = M + m/4``. ZOH discretization at ``Δt = 0.04`` s (the env's
control timestep) gives ``A_d, B_d``; the discrete LQR solve is delegated
to :class:`controller.lqr.lqr.LQR` to reuse our existing DARE machinery.

Configurable Q and R are passed at construction. Defaults are a sane
starting point:

    Q = diag(1, 1, 10, 1)   # state cost (heavier on θ)
    R = 1                   # control cost

Action is clipped to ``±action_clip`` (default 3.0 = env actuator limit).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import expm

from controller.lqr.lqr import LQR


# MuJoCo InvertedPendulum-v4/v5 physical parameters.
_M_CART = 10.471976    # cart body mass [kg]
_M_POLE = 5.018592     # pole body mass [kg]
_POLE_HALF_LEN = 0.3   # hinge → pole COM [m] (capsule half-length)
_G = 9.81              # gravitational acceleration [m/s²]
_DT = 0.04             # control timestep = physics_dt × frame_skip


def _continuous_dynamics() -> tuple[np.ndarray, np.ndarray]:
    """Linearized rod-pole cartpole dynamics around upright."""
    M, m, L, g = _M_CART, _M_POLE, _POLE_HALF_LEN, _G
    M_eff = M + m / 4.0
    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -3.0 * m * g / (4.0 * M_eff), 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0 * g * (M + m) / (4.0 * L * M_eff), 0.0],
        ],
        dtype=np.float64,
    )
    B_c = np.array(
        [
            [0.0],
            [1.0 / M_eff],
            [0.0],
            [-3.0 / (4.0 * L * M_eff)],
        ],
        dtype=np.float64,
    )
    return A_c, B_c


def _zoh_discretize(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero-order-hold discretization via the augmented matrix-exponential."""
    n = A_c.shape[0]
    aug = np.zeros((n + 1, n + 1), dtype=np.float64)
    aug[:n, :n] = A_c
    aug[:n, n : n + 1] = B_c
    expm_aug = expm(aug * dt)
    A_d = expm_aug[:n, :n]
    B_d = expm_aug[:n, n : n + 1]
    return A_d, B_d


class LQRPolicy:
    """Discrete-time LQR feedback around upright for InvertedPendulum-v4/v5.

    ``Q`` may be a 4-list (diagonal) or a 4×4 matrix. ``R`` may be a scalar,
    a 1-list, or a 1×1 matrix. The constructor solves the DARE once via
    :class:`controller.lqr.lqr.LQR` and stores the gain ``F``; ``__call__``
    just reorders the obs and applies ``u = -F · x``.
    """

    def __init__(
        self,
        Q=None,
        R=None,
        action_clip: float = 3.0,
        **_unused,
    ):
        Q_arr = np.diag([1.0, 1.0, 10.0, 1.0]) if Q is None else np.asarray(Q, dtype=np.float64)
        if Q_arr.ndim == 1:
            Q_arr = np.diag(Q_arr)
        if Q_arr.shape != (4, 4):
            raise ValueError(
                f"LQR_policy Q must be a 4-list or 4×4 matrix, got shape {Q_arr.shape}"
            )

        if R is None:
            R_arr = np.array([[1.0]], dtype=np.float64)
        else:
            R_arr = np.asarray(R, dtype=np.float64)
            if R_arr.ndim == 0:
                R_arr = R_arr.reshape(1, 1)
            elif R_arr.ndim == 1:
                R_arr = np.diag(R_arr)
        if R_arr.shape != (1, 1):
            raise ValueError(
                f"LQR_policy R must be a scalar or 1×1 matrix, got shape {R_arr.shape}"
            )

        A_c, B_c = _continuous_dynamics()
        A_d, B_d = _zoh_discretize(A_c, B_c, _DT)

        # Delegate the DARE solve to our existing LQR helper.
        lqr = LQR(
            torch.from_numpy(A_d),
            torch.from_numpy(B_d),
            torch.from_numpy(Q_arr),
            torch.from_numpy(R_arr),
        )
        # F is (1, 4) in state order [cart_x, cart_xdot, θ, θ̇].
        self.F = lqr.F.detach().cpu().numpy()
        self.action_clip = float(action_clip)

    def __call__(self, obs):
        # gym obs is [cart_x, θ, cart_xdot, θ̇]; LQR state order is
        # [cart_x, cart_xdot, θ, θ̇] — reorder here once.
        cart_x, theta, cart_xdot, theta_dot = obs
        state = np.array([cart_x, cart_xdot, theta, theta_dot], dtype=np.float64)
        u = float(-(self.F @ state)[0])
        return np.array([np.clip(u, -self.action_clip, self.action_clip)], dtype=np.float32)
