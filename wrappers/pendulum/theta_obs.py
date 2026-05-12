import gymnasium as gym
import numpy as np


class ThetaObsWrapper(gym.ObservationWrapper):
    """Convert Pendulum observation from [cos_th, sin_th, thdot] to [theta, thdot].

    theta = arctan2(sin_th, cos_th), so theta=0 is upright.
    Reduces state_dim from 3 to 2.
    """

    def __init__(self, env):
        super().__init__(env)
        # New observation space: [theta, thdot]
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -env.observation_space.high[2]], dtype=np.float32),
            high=np.array([np.pi, env.observation_space.high[2]], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        cos_th, sin_th, thdot = obs
        theta = np.arctan2(sin_th, cos_th)
        return np.array([theta, thdot], dtype=np.float32)
