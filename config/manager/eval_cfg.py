"""Evaluation config: rollout counts, success criteria, mode triggers, output paths.

``EvalCfg`` drives ``python -m eval --config ...`` and the auto-eval calls
issued at the end of ``controller.lqr`` and ``train_residual``. The set of
modes that run is implied by which artifact fields are populated:

* ``lqr_output_name`` empty                          → base policy only
* ``lqr_output_name`` set                            → base + LQR
* ``lqr_output_name`` + ``residual_experiment_name`` → base + LQR + residual
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class VideoCfg:
    """Optional rollout-video rendering controlled by ``eval/gym_rollout_video.py``.

    Gymnasium-only — the script uses ``gym.make(..., render_mode='rgb_array')``
    and ``env.render()`` for frame capture; not compatible with Isaac Lab /
    Forge envs. When ``enabled`` is False the script is a no-op. When True
    it writes one file per entry in ``formats`` (each must be one of
    ``"gif"`` or ``"mp4"``).
    """

    enabled: bool = False
    formats: list[str] = dataclasses.field(default_factory=lambda: ["gif"])
    fps: int = 30
    num_videos: int = 1
    """Number of independent rollouts to record. Each uses a distinct seed
    (``seed``, ``seed + 1``, …) so trajectories don't repeat. With
    ``num_videos == 1`` outputs are named ``rollout.{gif,mp4}``; otherwise
    ``rollout_000.{gif,mp4}``, ``rollout_001.{gif,mp4}``, …"""
    max_steps: int | None = None
    """Override ``EvalCfg.eval_max_steps`` for the recorded rollout."""
    seed: int | None = None
    """Override ``EvalCfg.eval_seed`` for the recorded rollout (base seed
    when ``num_videos > 1``)."""
    bar_pct_cap: float = 1.5
    """Bar visually saturates at this ratio of ``gamma_t / gamma_max`` so values
    above ``gamma_max`` overflow visibly without growing without bound."""
    use_gather_base_policy: bool = False
    """If True (and no residual is set), drive the rollout with the gather-data
    base policy from the dataset's ``gather_data_cfg`` snapshot instead of the
    Koopman LQR. Useful for long, stable demo rollouts on envs whose
    Koopman LQR doesn't hold upright."""


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
    capture_per_step: bool = True
    """Capture every per-env, per-step value the residual wrapper writes to
    ``info`` (``gamma_t``, ``eta_t``, ``stability_term``, ``env_reward``,
    ``z_t``, ``z_next``, ``z_pred``, ``z_ref_t``, ``x_pred``, …) into the
    per-mode ``_eval_traj.npz`` under the wrapper's own key names.
    ``states``, ``applied_action``, total ``reward`` and ``success`` are
    captured regardless. Set False to skip the heavier latent / prediction
    arrays."""

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman experiment subdir under ``results/``."""
    lqr_output_name: str = ""
    """Subdir under ``results/<koopman_experiment_name>/lqr/`` holding
    ``lqr.pt`` + ``ctrl_performance.yaml``. Required by
    ``eval/gym_rollout_video.py`` (needed to locate ``F`` and ``gamma_max``)."""
    residual_experiment_name: str | None = None
    """Source residual weights subdir; ``None`` evaluates base policy only."""
    residual_train_cfg_path: str | None = None
    """Path to the ``train_residual_cfg`` YAML used to train the residual.
    ``eval/gym_rollout_video.py`` reads it to mirror the wrapper kwargs the
    actor was trained against. Required when ``residual_experiment_name`` is
    set and ``video.enabled`` is True."""
    results_name: str = "pendulum_default"
    """Output subdir under ``eval/results/``."""

    video: VideoCfg = dataclasses.field(default_factory=VideoCfg)
