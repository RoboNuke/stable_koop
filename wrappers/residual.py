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

The base controller (LQR in the Koopman latent space) is baked in. The
latent reference splits into a fixed env-specific base (encoded from
``base_goal_fn(env)``, re-queried per step) and the actor's residual:

    z_ref_t = z_ref_t_base + z_ref_t_res
    u       = -F @ (z_t - z_ref_t_base - z_ref_t_res)

With ``z_ref_t_res = 0`` this reduces to ``u = -F @ (z_t - z_ref_t_base)``,
i.e. LQR-to-goal (the DARE-derived stabilizing law). The wrapper also augments the observation with stability
features and the reward with ``w * (gamma_max - gamma_t - eta_t)``.

Normalization
-------------
The koopman model and LQR ``F`` were fit in the **normalized** space produced
by ``data.augmentation.augment_trajectories``. To match that, the wrapper
takes ``obs_scale`` and ``act_scale`` and applies them transparently:

* obs flowing into ``koopman.encode``  is divided by ``obs_scale``
* action flowing into ``koopman.predict`` is divided by ``act_scale``
* ``F @ delta`` lives in normalized action space, so it's multiplied by
  ``act_scale`` before being sent to ``env.step`` (and re-normalized after
  env-side clipping for an accurate ``gamma_t``)

``gamma_t`` therefore lives in the same space as
:func:`controller.controller_analysis.count_steps_under_threshold`, so the
training-data coverage report and the live-rollout violation rate use the
same metric.

Pass ``obs_scale=None`` / ``act_scale=None`` if the koopman's augmentation cfg
has ``obs_scale_source="none"`` / ``act_scale_source="none"`` (i.e. the model
was trained on raw inputs). ``prepend_base_action`` / ``use_action_delta``
training options are not supported by the wrapper yet — pass them via
``aug_cfg`` and the wrapper will assert they're disabled.
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
        base_goal_fn,
        obs_scale=None,
        act_scale=None,
        aug_cfg=None,
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

        # Guard against unsupported augmentation modes — these change the
        # koopman input layout and require state-channel concatenation /
        # action-channel arithmetic the wrapper isn't set up for yet.
        if aug_cfg is not None:
            if bool(getattr(aug_cfg, "prepend_base_action", False)):
                raise NotImplementedError(
                    "ResidualPolicyEnv does not yet support koopman models trained "
                    "with augmentation.prepend_base_action=True (the encoder expects "
                    "a [obs; base_action] input that the wrapper has no source for "
                    "at inference)."
                )
            if bool(getattr(aug_cfg, "use_action_delta", False)):
                raise NotImplementedError(
                    "ResidualPolicyEnv does not yet support koopman models trained "
                    "with augmentation.use_action_delta=True."
                )

        self.env = env
        self.device = torch.device(device)
        self.num_envs = int(env.num_envs)

        if base_goal_fn is None:
            raise ValueError(
                "ResidualPolicyEnv requires a base_goal_fn. Register one for "
                "this env via data.env_builder.register_env_base_goal and "
                "resolve it with get_base_goal_fn(env_name)."
            )
        self._base_goal_fn = base_goal_fn
        self._base_goal_logged = False

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

        # Per-dim normalization scales that match the training-time
        # ``augment_trajectories`` projection. obs flowing into ``encode`` is
        # divided by ``obs_scale``; ``F @ delta`` is multiplied by ``act_scale``
        # before being sent to ``env.step`` (and re-normalized after env-side
        # clipping for an accurate ``gamma_t``). ``None`` means no scaling.
        self.obs_scale = (
            torch.as_tensor(np.asarray(obs_scale, dtype=np.float32), device=self.device)
            if obs_scale is not None else None
        )
        self.act_scale = (
            torch.as_tensor(np.asarray(act_scale, dtype=np.float32), device=self.device)
            if act_scale is not None else None
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

    # ---- normalization helpers (env-scale <-> koopman-scale) ---------

    def _obs_to_koop(self, obs: torch.Tensor) -> torch.Tensor:
        """Raw env obs -> normalized obs the koopman encoder was trained on."""
        if self.obs_scale is None:
            return obs
        return obs / self.obs_scale

    def _decoded_to_env(self, x_norm: torch.Tensor) -> torch.Tensor:
        """koopman.decode output (normalized) -> raw env-obs scale."""
        if self.obs_scale is None:
            return x_norm
        return x_norm * self.obs_scale

    def _action_to_koop(self, u_env: torch.Tensor) -> torch.Tensor:
        """Raw env action -> normalized action the koopman predict was trained on."""
        if self.act_scale is None:
            return u_env
        return u_env / self.act_scale

    def _action_to_env(self, u_norm: torch.Tensor) -> torch.Tensor:
        """Normalized action (F @ z output) -> raw env action."""
        if self.act_scale is None:
            return u_norm
        return u_norm * self.act_scale

    @torch.no_grad()
    def _compute_z_ref_base(self) -> torch.Tensor:
        """Per-step base latent reference encoded from ``base_goal_fn(env)``."""
        x_base = self._base_goal_fn(self.env)
        x_base = torch.as_tensor(x_base, device=self.device, dtype=torch.float32)
        if x_base.ndim == 1:
            x_base = x_base.unsqueeze(0).expand(self.num_envs, -1)
        elif x_base.ndim != 2 or x_base.shape[0] != self.num_envs:
            raise ValueError(
                f"base_goal_fn returned shape {tuple(x_base.shape)}; "
                f"expected (x_dim,) or ({self.num_envs}, x_dim)."
            )
        z_base = self.koopman.encode(self._obs_to_koop(x_base))
        if not self._base_goal_logged:
            print(
                f"[ResidualPolicyEnv] base goal (first call):\n"
                f"  x-space (row 0): {x_base[0].detach().cpu().numpy()}\n"
                f"  z-space (row 0): {z_base[0].detach().cpu().numpy()}"
            )
            self._base_goal_logged = True
        return z_base

    # ---- gym-shaped API (batched) ------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = obs.to(self.device)  # preserve env-native dtype (e.g. float64)
        with torch.no_grad():
            self._z_prev = self.koopman.encode(self._obs_to_koop(obs.float()))
        self._gamma_prev = torch.zeros(self.num_envs, device=self.device)
        return self._augment_obs(obs), info

    @torch.no_grad()
    def step(self, raw_action: torch.Tensor):
        raw_action = raw_action.to(self.device).float()
        z_t = self._z_prev  # (N, L) in normalized latent space

        if self.disable_action_augmentation:
            # Passthrough mode: ``raw_action`` is already in env-action space.
            # Both reference components are zero (kept on info for shape
            # consistency so downstream consumers don't have to branch).
            z_ref_base = torch.zeros(
                (self.num_envs, self.latent_dim),
                device=self.device,
                dtype=raw_action.dtype,
            )
            z_ref_res = torch.zeros_like(z_ref_base)
            delta = None
            u_env = torch.clamp(raw_action, self._env_act_low, self._env_act_high)
        else:
            z_ref_res = self.z_ref_max * raw_action  # (N, L)
            z_ref_base = self._compute_z_ref_base().to(z_ref_res.dtype)
            delta = z_t - z_ref_base - z_ref_res
            # ``F`` was fit against the koopman model's (normalized) A and B
            # via DARE, so the stabilizing law is ``u = -F·(z - z_ref)`` (the
            # closed loop is ``A - BF``; see controller/lqr/lqr.py). The
            # result lives in normalized action space; convert to raw env
            # scale before clipping against env action bounds.
            u_norm_unclipped = -(delta @ self.F.T)  # (N, A)
            u_env = torch.clamp(
                self._action_to_env(u_norm_unclipped),
                self._env_act_low, self._env_act_high,
            )

        # Re-normalize the post-clip action so ``predict`` sees the same units
        # the koopman model was trained on (and ``gamma_t`` lands in the same
        # space as the LQR analysis's coverage report).
        u_norm = self._action_to_koop(u_env)

        obs_next, env_reward, terminated, truncated, info = self.env.step(u_env)
        obs_next = obs_next.to(self.device)  # keep env-native dtype
        env_reward = env_reward.to(self.device).float()
        terminated = terminated.to(self.device)
        truncated = truncated.to(self.device)

        # Normalized float32 view used by the koopman model; the env-scale
        # obs flows through to the policy in its native dtype via _augment_obs.
        obs_next_koop = self._obs_to_koop(obs_next.float())
        z_next = self.koopman.encode(obs_next_koop)

        # Always-on diagnostics (exposed via info for downstream eval). z_pred
        # / x_pred are also reused for the gamma_t computation below; the
        # decode is at most one extra forward pass compared to the minimal
        # path. Everything lives in normalized space — same as
        # ``controller.controller_analysis.count_steps_under_threshold``.
        z_pred = self.koopman.predict(z_t, u_norm)
        x_pred = self.koopman.decode(z_pred)

        if self.pred_error_space == "latent":
            gamma_t = torch.linalg.vector_norm(z_next - z_pred, dim=-1)
        elif self.pred_error_space == "state_decoded":
            gamma_t = torch.linalg.vector_norm(obs_next_koop - x_pred, dim=-1)
        else:  # state_reconstruction
            x_recon = self.koopman.decode(z_next)
            gamma_t = torch.linalg.vector_norm(obs_next_koop - x_recon, dim=-1)

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
        info["applied_action"] = u_env   # env-native action sent downstream
        info["z_ref_t_base"] = z_ref_base  # encoded base goal (zero in passthrough)
        info["z_ref_t_res"] = z_ref_res    # actor residual (zero in passthrough)
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
