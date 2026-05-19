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
    """Max angle (degrees from upright) considered successful."""
    success_max_thdot: float = 1.0
    """Max angular velocity for success."""
    success_max_cart_vel: float = 0.5
    """Max cart linear velocity for success (InvertedPendulum-v4 only;
    ignored by envs that don't have a cart in their scorer)."""
    success_hold_steps: int = 20
    """Number of consecutive steps within the success region required."""

    eval_koopman_accuracy: bool = True
    """Run multi-step Koopman model accuracy eval."""
    eval_policy_rollout: bool = True
    """Run base / combined policy success-rate rollout."""

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman experiment subdir under ``results/``."""
    residual_experiment_name: str | None = None
    """Source residual weights subdir; ``None`` evaluates base policy only."""
    results_name: str = "pendulum_default"
    """Output subdir under ``eval/results/``."""
