"""LQR-controller-fitting config: cost matrices, stability bound, output paths."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class StabilityAnalysisCfg:
    """Inputs to the unified γ_max bound.

    The bound formula is picked automatically from the Koopman model's
    ``prepend_state`` flag:

    * no-prepend → ``γ_max = (m·ε_x)(1−ρ)/√κ(P) − η``
    * prepend    → ``γ_max = ε_x(1−ρ)/√(α·λ_max(P)) − η``

    ``η`` is parameterized by the *fraction of control budget consumed*,
    ``η = ctrl_pct · u_max / ||F||``. The :attr:`ctrl_percentages` list
    selects which fractions get a printed coverage block and a compliance
    heatmap. A separate dense sweep (100 points in [0, 1]) drives the
    summary plot.
    """

    epsilon_x: float = 0.3
    """Max tracking error in x-space. Scaled by encoder m for the no-prepend branch."""
    ctrl_percentages: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    """Control-budget fractions to print + heatmap. η = ctrl_pct · u_max / ||F||."""
    optimizer: str = "matching"
    """P-optimizer selection:
    * ``"none"`` — skip optimization; use the raw DARE-derived P only.
    * ``"matching"`` — run the optimizer that matches the active branch
      (SDP-Lyapunov for no-prepend, L-BFGS α-bound for prepend).
    * ``"beta"`` — level-2 SDP β-bound (works in both branches; bound is
      ``γ = ε_x(1-ρ)/√λ_max(P) - η`` with α≡1 via the constraint
      ``P ⪰ CᵀC``). Requires (A_cl, C) detectable above ``beta_rho_target``."""
    beta_rho_target: float = 0.0
    """Detectability threshold for the β-optimizer. Every A_cl mode with
    ``|λ| > beta_rho_target`` must satisfy ``rank([A_cl-λI; C]) = n`` or
    the run raises ``ValueError``. Ignored when ``optimizer != "beta"``."""
    mode_projection_groups: dict[str, list[int]] | None = None
    """Optional ``{group_name: [indices]}`` map. When set, the controllability
    fit prints, for each A eigenvalue with ``|λ| > 0.9``, the right-eigenvector
    projection ``||v[indices]||/||v||`` per group — i.e. how each Koopman
    mode is distributed across the configured latent groups (e.g. cart vs.
    pole vs. encoder dims)."""


@dataclasses.dataclass(kw_only=True)
class LQRControllerCfg:
    """Top-level LQR controller config.

    ``Q_diag``, ``R_diag``, ``C_diag`` give the diagonals of Q, R, and the
    state-extraction matrix C. All three are required; the runtime validates
    their lengths against the loaded model's actual_latent, action_dim, and
    real_state_dim, raising ``ValueError`` on mismatch or absence.
    """

    Q_diag: list[float] | None = None
    """Diagonal of the LQR state cost ``Q`` (length = actual_latent)."""
    R_diag: list[float] | None = None
    """Diagonal of the LQR control cost ``R`` (length = action_dim)."""
    C_mask: list[int] | None = None
    """Selector mask of which latent dims appear in the output. Length must
    equal ``actual_latent``; entries are 0 or 1. Each ``1`` at index ``i``
    adds a row ``eᵢᵀ`` to ``C``, so ``C`` is ``(sum(mask), actual_latent)``.
    Example: ``[1, 0, 1]`` → ``[[1,0,0],[0,0,1]]``."""
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
