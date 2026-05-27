"""Inverted-pendulum base-goal getter.

InvertedPendulum-v4/-v5 use the stock gymnasium env with no custom wrapper
stack, so we don't register a wrapper builder — only the base goal.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def inverted_pendulum_base_goal(env: gym.Env) -> np.ndarray:
    """Upright pole, cart at origin, zero velocities: ``[0, 0, 0, 0]``."""
    obs_dim = int(env.observation_space.shape[-1])
    if obs_dim != 4:
        raise ValueError(f"Unexpected inverted-pendulum obs dim {obs_dim}")
    return np.zeros(4, dtype=np.float32)


__all__ = ["inverted_pendulum_base_goal"]
