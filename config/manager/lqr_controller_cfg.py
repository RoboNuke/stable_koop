"""LQR-controller-fitting config: cost matrices, stability bound, output paths.

Field defaults mirror ``config/pendulum.yaml`` exactly.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class StabilityAnalysisCfg:
    """Inputs to the unified γ_max bound.

    The bound formula is picked automatically from the Koopman model's
    ``prepend_state`` flag:

    * no-prepend → ``γ_max = (m·ε_x)(1−ρ)/√κ(P) − η``
    * prepend    → ``γ_max = ε_x(1−ρ)/√(α·λ_max(P)) − η``
    """

    epsilon_x: float = 0.3
    """Max tracking error in x-space. Scaled by encoder m for the no-prepend branch."""
    eta: float = 0.1
    """Disturbance / displacement bound in latent space (both branches)."""
    use_optimization: bool = True
    """Run the P-optimizer that matches the active branch (SDP for no-prepend,
    L-BFGS α-optimizer for prepend)."""


@dataclasses.dataclass(kw_only=True)
class LQRControllerCfg:
    """Top-level LQR controller config."""

    q_scale: float = 1.0
    """Multiplier on identity Q."""
    q_epsilon_scale: float = 0.0
    """Additive ε·I term on Q (regularization)."""
    r_scale: float = 1.0
    """Multiplier on identity R."""
    scale_B: bool = False
    """Normalize B to unit spectral norm before solving LQR."""

    stability_analysis: StabilityAnalysisCfg = dataclasses.field(
        default_factory=StabilityAnalysisCfg
    )

    eval_cfg_path: str = ""
    """Path to the eval YAML used at the end of the controller fit to roll
    out base policy + (optional) residual policy and record per-step
    diagnostics (``z_t``, ``z_pred``, ``x_pred``, etc.) to
    ``eval_traj.npz`` in the controller's output directory. Required when
    the env has a registered :class:`eval.EnvScorer`."""

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman experiment subdir under ``results/``."""
    output_name: str = "pendulum_default"
    """Output subdir under ``results/<koopman_experiment_name>/lqr/``."""
