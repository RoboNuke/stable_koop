"""Evaluation config: rollout counts, success criteria, output paths.

Field defaults mirror ``config/pendulum.yaml`` exactly.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class EvalCfg:
    """Top-level evaluation config."""

    num_parallel_evals: int = 16
    """Parallel envs used during rollout-based eval."""
    eval_num_trajectories: int = 200
    eval_max_steps: int = 200
    eval_seed: int = 42

    success_angle_deg: float = 15.0
    """Max angle (degrees from upright) considered successful (pendulum)."""
    success_max_thdot: float = 1.0
    """Max angular velocity for success (pendulum)."""
    success_hold_steps: int = 20
    """Number of consecutive steps within the success region required."""

    eval_koopman_accuracy: bool = True
    """Run multi-step Koopman model accuracy eval."""
    eval_policy_rollout: bool = True
    """Run base / combined policy success-rate rollout."""

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman weights subdir under ``train_koopman/weights/``."""
    residual_experiment_name: str | None = None
    """Source residual weights subdir; ``None`` evaluates base policy only."""
    results_name: str = "pendulum_default"
    """Output subdir under ``eval/results/``."""
