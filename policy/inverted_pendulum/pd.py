"""PD controller for InvertedPendulum-v4 (MuJoCo) with optional cart-state feedback.

PD on pole angle ``θ`` and angular velocity ``θ̇``. With pure pole-only
feedback the cart drifts to the ±1 track wall in ~270 steps and the pole
falls; adding cart-position and cart-velocity terms lets the policy
regulate the cart around the origin so balance can be sustained.

Control law::

    u = k_theta · θ + k_theta_dot · θ̇ + k_x · cart_x + k_xdot · cart_xdot

To stabilize the cart around ``cart_x = 0`` with ``cart_xdot = 0``, use
**negative** ``k_x`` and ``k_xdot`` (force points opposite the cart's
position / velocity error). ``k_x = k_xdot = 0`` (the default) recovers
the pure pole-only PD.

Obs layout (gym ``InvertedPendulum-v4`` — qpos then qvel)::

    [cart_x, theta, cart_xdot, theta_dot]

Action: continuous, shape ``(1,)`` in ``[-3.0, 3.0]`` (force on the cart).
"""

from __future__ import annotations

import numpy as np


class PDPolicy:
    """4-gain linear feedback on ``(cart_x, θ, cart_xdot, θ̇)``.

    Sign convention follows the gym InvertedPendulum-v4 obs layout: positive
    ``k_theta``/``k_theta_dot`` stabilize the pole; **negative**
    ``k_x``/``k_xdot`` stabilize the cart. The defaults
    (``k_x = k_xdot = 0``) collapse to a pole-only PD for backward
    compatibility with the original undertuned-CartPole formulation.
    """

    def __init__(
        self,
        k_theta: float = 4.0,
        k_theta_dot: float = 0.5,
        k_x: float = 0.0,
        k_xdot: float = 0.0,
        action_clip: float = 3.0,
        **_unused,
    ):
        # ``**_unused`` swallows extra YAML fields so a single BasePolicyCfg
        # dataclass can populate every policy class without errors.
        self.k_theta = k_theta
        self.k_theta_dot = k_theta_dot
        self.k_x = k_x
        self.k_xdot = k_xdot
        self.action_clip = action_clip

    def __call__(self, obs):
        cart_x, theta, cart_xdot, theta_dot = obs
        u = (
            self.k_theta * theta
            + self.k_theta_dot * theta_dot
            + self.k_x * cart_x
            + self.k_xdot * cart_xdot
        )
        return np.array([np.clip(u, -self.action_clip, self.action_clip)], dtype=np.float32)
