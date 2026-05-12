"""Energy-shaping swing-up policies for Pendulum-v1.

Pendulum-v1 physics: ``m = 1, l = 1, g = 10, I = m·l²/3 = 1/3, max_torque = 2``.
Energy convention: ``E = θ̇²/6 + 5·cos θ`` (continuous), ``E_target = 5`` (upright).

Two variants are provided:

* :class:`EnergyShapingPolicy` — continuous proportional pumping.
* :class:`BangEnergyShapingPolicy` — bang-bang pumping with brake at half gain.

Both switch to PD balance when ``|θ| < switch_angle``.
"""

from __future__ import annotations

import numpy as np

from policy.pendulum.pd import _parse_obs


class ZeroPolicy:
    """Always emits zero action — used when the YAML names ``base_policy: "none"``."""

    def __init__(self, **_unused):
        pass

    def __call__(self, obs):
        return np.array([0.0], dtype=np.float32)


class EnergyShapingPolicy:
    """Energy-shaping swing-up + PD balance for Pendulum-v1."""

    def __init__(
        self,
        kp: float = 10.0,
        kd: float = 3.0,
        ke: float = 2.0,
        switch_angle: float = 0.7854,
        **_unused,
    ):
        # ``switch_angle`` here is in **radians** to preserve the historical
        # function signature; the YAML ``switch_angle`` field is in degrees and
        # the caller is responsible for the conversion (matches the legacy
        # ``sa * π / 180`` pattern in ``launch/eval_policy.py``).
        self.kp = kp
        self.kd = kd
        self.ke = ke
        self.switch_angle = switch_angle

    def __call__(self, obs):
        theta, cos_th, sin_th, thdot = _parse_obs(obs)
        E = thdot ** 2 / 6.0 + 5.0 * cos_th
        E_target = 5.0
        u_swing = self.ke * thdot * (E_target - E)
        u_balance = -self.kp * theta - self.kd * thdot
        near_top = abs(theta) < self.switch_angle
        u = u_balance if near_top else u_swing
        return np.array([np.clip(u, -2.0, 2.0)])


class BangEnergyShapingPolicy:
    """Bang-bang energy-shaping swing-up + PD balance for Pendulum-v1.

    When ``E < E_target`` pumps energy via bang-bang along ``sign(θ̇ · cos θ)``;
    when ``E > E_target`` brakes at half gain to shed excess energy.
    """

    def __init__(
        self,
        kp: float = 10.0,
        kd: float = 3.0,
        ke: float = 2.0,
        switch_angle: float = 1.0472,
        **_unused,
    ):
        self.kp = kp
        self.kd = kd
        self.ke = ke
        self.switch_angle = switch_angle

    def __call__(self, obs):
        theta, cos_th, sin_th, thdot = _parse_obs(obs)
        E = thdot ** 2 / 6.0 - 5.0 * cos_th
        E_target = -5.0
        if E < E_target:
            u_swing = self.ke * np.sign(thdot * cos_th)
        else:
            u_swing = -self.ke * np.sign(thdot * cos_th) * 0.5
        u_balance = -self.kp * theta - self.kd * thdot
        near_top = abs(theta) < self.switch_angle
        u = u_balance if near_top else u_swing
        return np.array([np.clip(u, -2.0, 2.0)])
