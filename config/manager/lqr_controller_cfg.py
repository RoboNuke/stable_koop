"""LQR-controller-fitting config: cost matrices, stability bounds, output paths.

Field defaults mirror ``config/pendulum.yaml`` exactly.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class StabilityAnalysisCfg:
    """Which closed-loop stability bounds to compute on the fitted LQR."""

    use_eigen_bound: bool = True
    """Spectral-radius / transient-constant bound."""
    use_lyapunov_bound: bool = True
    """Lyapunov-equation-based bound."""
    use_m_free_bound: bool = False
    """Lipschitz-m-free bound (does not assume encoder lower-Lipschitz)."""
    optimize_lyapunov_P: bool = True
    """Run SDP optimization to tighten the Lyapunov P matrix."""

    use_alpha_bound: bool = False
    """Alpha-bound stability analysis (requires ``prepend_state`` in Koopman cfg)."""
    alpha_epsilon_x: float = 0.3
    """Max tracking error in state space (alpha bound)."""
    alpha_eta: float = 0.1
    """Disturbance bound in latent space (alpha bound)."""


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
    max_tracking_error_x: float = 0.3
    """Max state-space tracking error tolerated (defines success bounds)."""
    max_displacement_x: float = 0.57
    """Max possible state displacement (defines normalization)."""
    controllable_subspace: bool = False
    """Solve LQR only in the controllable subspace."""
    ctrl_threshold: float | None = None
    """Controllability singular-value threshold (null = machine epsilon default)."""

    stability_analysis: StabilityAnalysisCfg = dataclasses.field(
        default_factory=StabilityAnalysisCfg
    )

    koopman_experiment_name: str = "pendulum_default"
    """Source Koopman weights subdir under ``train_koopman/weights/``."""
    output_name: str = "pendulum_default"
    """Output subdir under ``controller/lqr/weights/`` (or equivalent)."""
