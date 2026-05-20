"""Stability-aware residual wrapper (batched, tensor-resident).

The wrapper expects an inner env with the IsaacLab/Forge-shaped batched-tensor
interface::

    env.num_envs            : int (N)
    env.device              : torch.device
    env.observation_space   : gym.spaces.Box (per-env shape)
    env.action_space        : gym.spaces.Box (per-env shape)
    env.reset(**kwargs)     -> (obs: (N, O) tensor, info: dict)
    env.step(action)        -> (obs: (N, O) tensor,
                                reward: (N, 1) tensor,
                                terminated: (N, 1) bool tensor,
                                truncated: (N, 1) bool tensor,
                                info: dict)
    env.close()

All math runs in torch on ``env.device`` — no numpy round trips per step. For
gymnasium ``SyncVectorEnv`` use, place :class:`GymVectorAdapter` (from
``wrappers.gym_vec_adapter``) between the raw vec env and this wrapper.

The base controller (LQR in the Koopman latent space) is baked in: with no
residual, the env runs ``u = F @ z_t``. With the residual ``z_ref_t``,
``u = F @ (z_t - z_ref_t)``. The wrapper also augments the observation with
stability features and the reward with ``w * (gamma_max - gamma_t - eta_t)``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


def _matrix_spec_norm(M: torch.Tensor, *, latent_dim: int) -> torch.Tensor:
    """Spectral norm of M treated as a 2-D matrix (latent_dim columns)."""
    if M.ndim == 1:
        M = M.reshape(-1, latent_dim)
    elif M.ndim != 2:
        M = M.reshape(-1, latent_dim)
    return torch.linalg.matrix_norm(M, ord=2)


class ResidualPolicyEnv:
    """Batched, tensor-resident residual wrapper. See module docstring."""

    metadata: dict = {}

    def __init__(
        self,
        env,
        koopman_model,
        lqr,
        *,
        gamma_max: float,
        gamma_worst_case: float,
        reward_weight: float,
        pred_error_space: str,
        z_ref_max_mode: str,
        obs_augmentation: str,
        disable_action_augmentation: bool,
        device: torch.device,
    ):
        if pred_error_space not in ("latent", "state_decoded", "state_reconstruction"):
            raise ValueError(
                f"pred_error_space must be one of 'latent' / 'state_decoded' / "
                f"'state_reconstruction', got {pred_error_space!r}"
            )
        if obs_augmentation not in ("both", "raw", "none"):
            raise ValueError(
                f"obs_augmentation must be 'both', 'raw', or 'none', "
                f"got {obs_augmentation!r}"
            )

        self.env = env
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)

        self.obs_augmentation = obs_augmentation
        self._n_extra_obs = {"both": 2, "raw": 1, "none": 0}[obs_augmentation]
        self.disable_action_augmentation = bool(disable_action_augmentation)
        self.pred_error_space = pred_error_space
        self.reward_weight = float(reward_weight)

        self.koopman = koopman_model.to(self.device).eval()
        self.latent_dim = int(self.koopman.encoder_latent_dim) + int(
            self.koopman.prepend_dim
        )

        F = lqr.F.detach().to(self.device).float()  # (A, L)
        self.F = F
        self.F_norm = torch.linalg.matrix_norm(F, ord=2)
        B = self.koopman.B_matrix.detach().to(self.device).float()
        self.B_norm = _matrix_spec_norm(B, latent_dim=self.latent_dim)
        self._inv_BF_norm = 1.0 / (self.B_norm * self.F_norm)

        self.gamma_max = torch.tensor(float(gamma_max), device=self.device)
        self.gamma_worst_case = torch.tensor(
            float(gamma_worst_case), device=self.device
        )

        # z_ref_max — kept as a device scalar.
        if z_ref_max_mode == "action_bound":
            high = torch.as_tensor(
                np.asarray(env.action_space.high), device=self.device, dtype=torch.float32
            )
            low = torch.as_tensor(
                np.asarray(env.action_space.low), device=self.device, dtype=torch.float32
            )
            u_max = torch.maximum(
                torch.linalg.vector_norm(high), torch.linalg.vector_norm(low)
            )
            self.z_ref_max = u_max / self.B_norm
        elif z_ref_max_mode == "stability":
            self.z_ref_max = self.gamma_max / (self.B_norm * self.F_norm)
        else:
            raise ValueError(
                f"z_ref_max_mode must be 'action_bound' or 'stability', got {z_ref_max_mode!r}"
            )

        # Observation / action spaces in per-env (single env) shape.
        env_obs_low = np.asarray(env.observation_space.low, dtype=np.float32)
        env_obs_high = np.asarray(env.observation_space.high, dtype=np.float32)
        extra_low = np.full(self._n_extra_obs, -np.inf, dtype=np.float32)
        extra_high = np.full(self._n_extra_obs, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([env_obs_low, extra_low]),
            high=np.concatenate([env_obs_high, extra_high]),
            dtype=np.float32,
        )
        if self.disable_action_augmentation:
            self.action_space = env.action_space
        else:
            self.action_space = gym.spaces.Box(
                low=-np.ones(self.latent_dim, dtype=np.float32),
                high=np.ones(self.latent_dim, dtype=np.float32),
                dtype=np.float32,
            )

        self._env_act_low = torch.as_tensor(
            np.asarray(env.action_space.low), device=self.device, dtype=torch.float32
        )
        self._env_act_high = torch.as_tensor(
            np.asarray(env.action_space.high), device=self.device, dtype=torch.float32
        )

        # Per-row state lives on device.
        self._z_prev: torch.Tensor | None = None  # (N, L)
        self._gamma_prev = torch.zeros(self.num_envs, device=self.device)

    # ---- per-row state reset helpers ---------------------------------

    def _reset_rows(self, mask: torch.Tensor) -> None:
        """Zero out gamma_prev for rows in ``mask`` (post-auto-reset rows)."""
        if mask.any():
            self._gamma_prev = torch.where(
                mask, torch.zeros_like(self._gamma_prev), self._gamma_prev
            )

    # ---- gym-shaped API (batched) ------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.to(self.device)  # preserve env-native dtype (e.g. float64)
        with torch.no_grad():
            self._z_prev = self.koopman.encode(obs.float())
        self._gamma_prev = torch.zeros(self.num_envs, device=self.device)
        return self._augment_obs(obs), info

    @torch.no_grad()
    def step(self, raw_action: torch.Tensor):
        raw_action = raw_action.to(self.device).float()
        z_t = self._z_prev  # (N, L)

        if self.disable_action_augmentation:
            # Passthrough mode: no latent-reference. z_ref is conceptually
            # zero for downstream diagnostics (kept on info for shape
            # consistency across modes).
            z_ref = torch.zeros(
                (self.num_envs, self.latent_dim),
                device=self.device,
                dtype=raw_action.dtype,
            )
            delta = None
            u = torch.clamp(raw_action, self._env_act_low, self._env_act_high)
        else:
            z_ref = self.z_ref_max * raw_action  # (N, L)
            delta = z_t - z_ref
            u = delta @ self.F.T  # (N, A)
            u = torch.clamp(u, self._env_act_low, self._env_act_high)

        obs_next, env_reward, terminated, truncated, info = self.env.step(u)
        obs_next = obs_next.to(self.device)  # keep env-native dtype
        env_reward = env_reward.to(self.device).float()
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)

        # Float32 view used only for the koopman model; obs_next itself is
        # passed through to the policy in its native dtype below.
        obs_next_f32 = obs_next.float()
        z_next = self.koopman.encode(obs_next_f32)

        # Always-on diagnostics (exposed via info for downstream eval). z_pred
        # / x_pred are also reused for the gamma_t computation below depending
        # on pred_error_space, so this is at most one extra decode per step
        # compared to the minimal path.
        z_pred = self.koopman.predict(z_t, u)
        x_pred = self.koopman.decode(z_pred)

        if self.pred_error_space == "latent":
            gamma_t = torch.linalg.vector_norm(z_next - z_pred, dim=-1)
        elif self.pred_error_space == "state_decoded":
            gamma_t = torch.linalg.vector_norm(obs_next_f32 - x_pred, dim=-1)
        else:  # state_reconstruction
            x_recon = self.koopman.decode(z_next)
            gamma_t = torch.linalg.vector_norm(obs_next_f32 - x_recon, dim=-1)

        if delta is None:
            eta_t = torch.zeros_like(gamma_t)
        else:
            eta_t = self.B_norm * self.F_norm * torch.linalg.vector_norm(delta, dim=-1)

        stability_term = self.gamma_max - gamma_t - eta_t
        # env_reward may be (N,) or (N, 1); match its shape for the augmentation.
        if env_reward.ndim == 2:
            aug_reward = env_reward + self.reward_weight * stability_term.unsqueeze(-1)
        else:
            aug_reward = env_reward + self.reward_weight * stability_term

        # Update caches; clear gamma_prev for envs that just terminated (the
        # inner env auto-resets, so obs_next is post-reset for those rows).
        self._z_prev = z_next
        self._gamma_prev = gamma_t
        done_mask = (terminated.bool().reshape(self.num_envs) |
                     truncated.bool().reshape(self.num_envs))
        self._reset_rows(done_mask)

        info = dict(info)
        info["env_reward"] = env_reward
        info["gamma_t"] = gamma_t
        info["eta_t"] = eta_t
        info["stability_term"] = stability_term
        info["applied_action"] = u       # env-native action sent downstream
        info["z_ref_t"] = z_ref          # scaled latent reference (zero in passthrough)
        info["z_t"] = z_t
        info["z_next"] = z_next
        info["z_pred"] = z_pred
        info["x_pred"] = x_pred

        return self._augment_obs(obs_next), aug_reward, terminated, truncated, info

    # ---- obs augmentation -------------------------------------------

    def _augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.obs_augmentation == "none":
            return obs
        # Match the obs dtype so cat doesn't silently upcast / mix precisions.
        gamma_col = self._gamma_prev.to(obs.dtype).unsqueeze(-1)
        if self.obs_augmentation == "raw":
            return torch.cat([obs, gamma_col], dim=-1)
        normalized = ((self._gamma_prev - self.gamma_worst_case) * self._inv_BF_norm).to(obs.dtype)
        return torch.cat([obs, gamma_col, normalized.unsqueeze(-1)], dim=-1)

    def close(self):
        return self.env.close()
