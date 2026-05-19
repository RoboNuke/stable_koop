"""Residual-policy training config (SAC).

Field defaults mirror ``config/pendulum.yaml`` exactly.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class TrainResidualCfg:
    """Top-level residual-policy SAC config."""

    z_ref_limit: float = 1.0
    """Max ‖z_ref‖ output by the residual actor (latent-space action bound)."""
    num_envs: int = 8
    """Parallel envs."""
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

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman experiment subdir under ``results/``."""
    lqr_name: str = "pendulum_default"
    """LQR fit subdir under ``results/<koopman_experiment_name>/lqr/``."""
    experiment_name: str = "pendulum_default"
    """Output subdir under ``train_residual/weights/``."""
    seed: int = 42
