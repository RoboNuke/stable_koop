import gymnasium as gym
import numpy as np


class ResidualPolicyEnv(gym.Wrapper):
    """Wraps an env so the agent outputs z_ref in latent space.

    The residual control is u_res = z_ref @ F^T (LQR gain), added to the
    base policy action.

    Observation: [cos_th, sin_th, thdot, base_action]
    Action: z_ref vector of size latent_dim
    Applied action: clip(base_action + F @ z_ref, env_low, env_high)
    """

    def __init__(self, env, base_policy, lqr, latent_dim, z_ref_limit=1.0):
        super().__init__(env)
        self.base_policy = base_policy
        self.F = lqr.F.numpy().astype(np.float32)  # (action_dim, latent_dim)
        self.latent_dim = latent_dim
        self.z_ref_limit = z_ref_limit

        # Augmented observation: env obs + base action
        obs_high = env.observation_space.high
        obs_low = env.observation_space.low
        act_high = env.action_space.high
        act_low = env.action_space.low
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_low, act_low]),
            high=np.concatenate([obs_high, act_high]),
            dtype=np.float32,
        )

        # Agent action space: z_ref in latent space
        self.action_space = gym.spaces.Box(
            low=-np.ones(latent_dim, dtype=np.float32) * z_ref_limit,
            high=np.ones(latent_dim, dtype=np.float32) * z_ref_limit,
            dtype=np.float32,
        )

        self._env_act_low = act_low
        self._env_act_high = act_high

    def _augment_obs(self, obs):
        base_action = self.base_policy(obs)
        self._last_base_action = base_action
        return np.concatenate([obs, base_action]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, z_ref):
        # u_res = F @ z_ref -> shape (action_dim,)
        u_res = self.F @ z_ref
        total_action = np.clip(
            self._last_base_action + u_res,
            self._env_act_low, self._env_act_high,
        )
        obs, reward, terminated, truncated, info = self.env.step(total_action)
        return self._augment_obs(obs), reward, terminated, truncated, info
