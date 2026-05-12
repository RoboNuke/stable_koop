"""Pendulum-specific gym wrappers + the per-env wrapper builder.

The wrapper builder is the single function that the generic
:func:`data.env_builder.make_single_env` dispatcher calls when ``env_name`` is
a pendulum variant. Adding a new pendulum option means editing this builder,
not the framework.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from wrappers.pendulum.pendulum import FrictionPendulumWrapper, PendulumWrapper
from wrappers.pendulum.theta_obs import ThetaObsWrapper
from wrappers.pendulum.limited_spawn import LimitedSpawnWrapper


def apply_pendulum_wrappers(
    env: gym.Env,
    *,
    obs_type: str = "cos_sin",
    limited_spawn: bool = False,
    spawn_angle_range: float = 180.0,
    friction_coeff: float = 0.0,
    **_unused,
) -> gym.Env:
    """Apply pendulum-specific wrappers given the env-kwargs from a YAML.

    Order is significant: spawn-limiting and friction operate on raw state, so
    they wrap the env before the obs transform.
    """
    if limited_spawn:
        max_angle = spawn_angle_range * np.pi / 180.0
        env = LimitedSpawnWrapper(env, max_angle=max_angle)
    if friction_coeff != 0.0:
        env = FrictionPendulumWrapper(env, friction_coeff=friction_coeff)
    if obs_type == "theta":
        env = ThetaObsWrapper(env)
    return env


__all__ = [
    "FrictionPendulumWrapper",
    "LimitedSpawnWrapper",
    "PendulumWrapper",
    "ThetaObsWrapper",
    "apply_pendulum_wrappers",
]
