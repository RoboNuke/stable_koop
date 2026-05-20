"""Residual-policy training config (SAC)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class TrainResidualCfg:
    """Top-level residual-policy SAC config."""

    # Stability-aware residual settings.
    gamma_worst_case: float
    """User-supplied worst-case Koopman one-step error (env-specific)."""
    reward_weight: float = 1.0
    """Weight on the stability reward term ``gamma_max - gamma_t - eta_t``."""
    pred_error_space: str = "latent"
    """One of 'latent' / 'state_decoded' / 'state_reconstruction'."""
    z_ref_max_mode: str = "action_bound"
    """One of 'action_bound' (u_max / ||B||) or 'stability' (gamma_max / (||B|| ||F||))."""
    obs_augmentation: str = "both"
    """Which stability features to append to the observation:
    'both' (raw gamma_{t-1} + normalized), 'raw' (gamma_{t-1} only), or 'none'."""
    disable_action_augmentation: bool = False
    """If True, the wrapper exposes the base env action space directly: agent
    actions pass through to the env unchanged (no z_ref / F·delta machinery).
    In this mode eta_t = 0 so the stability reward reduces to
    ``reward_weight * (gamma_max - gamma_t)``. Use for baseline comparisons."""

    # SAC hyperparameters.
    num_envs: int = 8
    total_timesteps: int = 100_000
    eval_interval: int = 5_000
    batch_size: int = 256
    lr: float = 3.0e-4
    actor_hidden_size: int = 64
    actor_hidden_layers: int = 2
    critic_hidden_size: int = 256
    critic_hidden_layers: int = 2
    gamma: float = 0.99
    tau: float = 0.005
    initial_entropy_value: float = 1.0
    random_timesteps: int = 1_000
    learning_starts: int = 1_000
    memory_size: int = 100_000

    # Source artifacts: filesystem paths.
    koopman_path: str
    """Path to the Koopman experiment dir (containing ``koopman_ckpt.pt``)
    or directly to a ``koopman_ckpt.pt`` file."""
    controller_path: str
    """Path to the LQR controller dir (containing ``lqr.pt`` and
    ``ctrl_performance.yaml``)."""

    experiment_name: str = "pendulum_default"
    """Output subdir under ``train_residual/weights/``."""
    seed: int = 42
