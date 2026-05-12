"""Gym environment construction with stable_koop's wrapper stack.

Single source of truth for ``make_single_env`` / ``make_eval_env``. The pendulum
wrappers (``ThetaObsWrapper``, ``LimitedSpawnWrapper``, ``FrictionPendulumWrapper``)
are applied conditionally based on the gather-data config.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np

from wrappers.pendulum import (
    FrictionPendulumWrapper,
    LimitedSpawnWrapper,
    ThetaObsWrapper,
)


def make_single_env(
    env_name: str,
    obs_type: str = "cos_sin",
    limited_spawn: bool = False,
    spawn_angle_range: float = 180.0,
    friction_coeff: float = 0.0,
    render_mode: Optional[str] = None,
) -> gym.Env:
    """Build one env and apply the configured wrapper stack."""
    env = gym.make(env_name, render_mode=render_mode) if render_mode else gym.make(env_name)
    if limited_spawn:
        max_angle = spawn_angle_range * np.pi / 180.0
        env = LimitedSpawnWrapper(env, max_angle=max_angle)
    if friction_coeff != 0.0:
        env = FrictionPendulumWrapper(env, friction_coeff=friction_coeff)
    if obs_type == "theta":
        env = ThetaObsWrapper(env)
    return env


def make_eval_env(
    env_name: str,
    num_parallel_evals: int = 1,
    **kwargs,
):
    """Build a vectorized env if ``num_parallel_evals > 1``, else a single env."""
    if num_parallel_evals > 1:
        return gym.vector.SyncVectorEnv(
            [lambda: make_single_env(env_name, **kwargs) for _ in range(num_parallel_evals)]
        )
    return make_single_env(env_name, **kwargs)
