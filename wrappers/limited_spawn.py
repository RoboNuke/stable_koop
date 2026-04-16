"""Wrapper to limit pendulum spawn angle range."""
import gymnasium as gym
import numpy as np


class LimitedSpawnWrapper(gym.Wrapper):
    """Resets pendulum with angle uniformly in [-max_angle, max_angle]."""

    def __init__(self, env, max_angle=np.pi, max_thdot=8.0):
        super().__init__(env)
        self.max_angle = max_angle
        self.max_thdot = max_thdot

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        theta = np.random.uniform(-self.max_angle, self.max_angle)
        thdot = np.random.uniform(-self.max_thdot, self.max_thdot)
        self.env.unwrapped.state = np.array([theta, thdot])
        obs = self.env.unwrapped._get_obs()
        return obs, info
