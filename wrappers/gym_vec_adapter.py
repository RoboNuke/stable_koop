"""Adapters between gymnasium VectorEnv (numpy) and the batched-tensor
interface expected by :class:`wrappers.residual.ResidualPolicyEnv`.

Two directions:

* :class:`GymVectorAdapter` — wraps a ``gym.vector.VectorEnv`` (numpy in/out)
  and exposes the IsaacLab-shaped batched-tensor interface. Place this
  *between* the raw gymnasium ``SyncVectorEnv`` and ``ResidualPolicyEnv``.

* :class:`TensorToGymAdapter` — wraps a batched-tensor env and exposes a
  gymnasium ``VectorEnv``-shaped numpy interface. Used by the evaluator,
  which iterates per-sample numpy arrays.

Conventions matched (IsaacLab/skrl):
* observation tensor:  (N, obs_dim), float32, on device
* reward tensor:       (N, 1),       float32, on device
* terminated/truncated:(N, 1),       bool,    on device
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class GymVectorAdapter:
    """Numpy gym VectorEnv -> batched-tensor interface."""

    def __init__(self, vec_env, *, device: torch.device):
        self.vec_env = vec_env
        self.device = torch.device(device)
        self.num_envs = int(vec_env.num_envs)
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space

    def reset(self, **kwargs):
        obs, info = self.vec_env.reset(**kwargs)
        # Preserve env-native obs dtype (gym InvertedPendulum returns float64,
        # for instance). Downcasting to float32 here loses precision on the
        # policy input — the koopman model handles its own float32 conversion.
        return torch.as_tensor(obs, device=self.device), info

    def step(self, action: torch.Tensor):
        a = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self.vec_env.step(a)
        return (
            torch.as_tensor(obs, device=self.device),
            torch.as_tensor(reward, device=self.device, dtype=torch.float32).reshape(self.num_envs, 1),
            torch.as_tensor(terminated, device=self.device, dtype=torch.bool).reshape(self.num_envs, 1),
            torch.as_tensor(truncated, device=self.device, dtype=torch.bool).reshape(self.num_envs, 1),
            info,
        )

    def close(self):
        return self.vec_env.close()


class TensorToGymAdapter:
    """Batched-tensor env -> numpy gym VectorEnv-shaped interface.

    Mimics enough of ``gym.vector.VectorEnv`` for the evaluator
    (``num_envs``, ``single_observation_space``, ``single_action_space``,
    ``reset``/``step`` returning numpy arrays).
    """

    def __init__(self, tensor_env):
        self.tensor_env = tensor_env
        self.num_envs = int(tensor_env.num_envs)
        self.single_observation_space = tensor_env.observation_space
        self.single_action_space = tensor_env.action_space
        self.observation_space = self._batched_space(tensor_env.observation_space)
        self.action_space = self._batched_space(tensor_env.action_space)

    @staticmethod
    def _batched_space(space):
        low = np.broadcast_to(space.low, (1, *space.low.shape)).copy()
        high = np.broadcast_to(space.high, (1, *space.high.shape)).copy()
        return gym.spaces.Box(low=low, high=high, dtype=space.dtype)

    def reset(self, **kwargs):
        obs, info = self.tensor_env.reset(**kwargs)
        return obs.detach().cpu().numpy(), info

    def step(self, action_np):
        action_t = torch.as_tensor(
            action_np, device=self.tensor_env.device, dtype=torch.float32
        )
        obs, reward, terminated, truncated, info = self.tensor_env.step(action_t)
        return (
            obs.detach().cpu().numpy(),
            reward.detach().cpu().numpy().reshape(self.num_envs),
            terminated.detach().cpu().numpy().reshape(self.num_envs),
            truncated.detach().cpu().numpy().reshape(self.num_envs),
            info,
        )

    def close(self):
        return self.tensor_env.close()
