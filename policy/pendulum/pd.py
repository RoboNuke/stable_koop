"""PD-controller policy for Pendulum-v1 targeting the upright position."""

from __future__ import annotations

import numpy as np


def _parse_obs(obs):
    """Extract (theta, cos_th, sin_th, thdot) from either obs format.

    Handles both ``[cos_th, sin_th, thdot]`` (3D) and ``[theta, thdot]`` (2D).
    """
    if len(obs) == 3:
        cos_th, sin_th, thdot = obs
        theta = np.arctan2(sin_th, cos_th)
    else:
        theta, thdot = obs
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
    return theta, cos_th, sin_th, thdot


class PDPolicy:
    """Proportional-derivative policy ``u = -kp·θ − kd·θ̇``, clipped to ±2."""

    def __init__(self, kp: float, kd: float, **_unused):
        # ``**_unused`` swallows any extra YAML fields (ke, gamma, switch_angle)
        # so a single BasePolicyCfg dataclass can populate every policy class.
        self.kp = kp
        self.kd = kd

    def __call__(self, obs):
        theta, _, _, thdot = _parse_obs(obs)
        u = -self.kp * theta - self.kd * thdot
        return np.array([np.clip(u, -2.0, 2.0)])
