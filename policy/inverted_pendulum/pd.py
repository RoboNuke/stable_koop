"""Undertuned PD controller for InvertedPendulum-v4 (MuJoCo).

PD on pole angle ``θ`` and angular velocity ``θ̇``, ignoring cart position
and velocity. Gains are intentionally low so the controller is slow to
react and fails on harder initial conditions.

Obs layout (gym ``InvertedPendulum-v4``)::

    [cart_x, cart_xdot, theta, theta_dot]

Action: continuous, shape ``(1,)`` in ``[-3.0, 3.0]`` (force on the cart).
"""

from __future__ import annotations

import numpy as np


class PDPolicy:
    """``u = k_theta · θ + k_theta_dot · θ̇`` clipped to ``±action_clip``.

    Sign convention matches the original discrete-CartPole formulation: when
    the pole falls in the +θ direction, the controller pushes the cart in
    the +x direction to catch it. The discrete ``return 1 if u > 0 else 0``
    is adapted to continuous by clipping ``u`` directly to the actuator
    range, so the "undertuned" defaults produce a weak corrective force.
    """

    def __init__(
        self,
        k_theta: float = 4.0,
        k_theta_dot: float = 0.5,
        action_clip: float = 3.0,
        **_unused,
    ):
        # ``**_unused`` swallows extra YAML fields so a single BasePolicyCfg
        # dataclass can populate every policy class without errors.
        self.k_theta = k_theta
        self.k_theta_dot = k_theta_dot
        self.action_clip = action_clip

    def __call__(self, obs):
        _, _, theta, theta_dot = obs
        u = self.k_theta * theta + self.k_theta_dot * theta_dot
        return np.array([np.clip(u, -self.action_clip, self.action_clip)], dtype=np.float32)
