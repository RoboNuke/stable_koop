"""Koopman-training config: paradigm, model parameterization, B-fitting, losses.

Field defaults mirror ``config/pendulum.yaml`` exactly. Enum-like choices
(``approach``, ``k_type``, ``encoder_type``, ``b_fitting_method``) are required
(no default) so a YAML must name them explicitly.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class LossesCfg:
    """Per-loss enable flags, weights, and targets — every field copied from
    ``config/pendulum.yaml``."""

    recon_weight: float = 1.0
    """Reconstruction MSE weight (always on)."""
    pred_weight: float = 1.0
    """One-step latent prediction MSE weight (always on)."""
    latent_consistency_weight: float = 0.5
    """Consistency between sequential and direct latent prediction routes."""

    latent_rollout_loss: bool = False
    latent_rollout_weight: float = 1.0
    x_pred_weight: float = 1.0
    """State-space multi-step prediction weight (only when
    ``latent_rollout_loss`` is true)."""

    l_inf_pred_loss: bool = False
    l_inf_pred_weight: float = 1.0

    controllability_loss: bool = False
    ctrl_weight: float = 0.1

    bi_lipschitz_loss: bool = True
    bilip_weight: float = 2.0
    bilip_m_target: float = 1.0

    spectral_loss: bool = False
    spectral_weight: float = 0.01

    normality_loss: bool = False
    normality_weight: float = 2.0

    cl_normality_loss: bool = False
    cl_normality_weight: float = 1.0

    b_eigen_loss: bool = True
    b_eigen_weight: float = 1.0
    b_eigen_min_scale: float = 0.3

    b_scale_loss: bool = False
    b_scale_weight: float = 1.0
    b_scale_target: float = 1.0

    b_min_sv_loss: bool = False
    b_min_sv_weight: float = 1.0
    b_min_sv_target: float = 0.1

    eig_spread_loss: bool = True
    eig_spread_weight: float = 1.0
    eig_spread_min_gap: float = 0.1

    unstable_ctrl_loss: bool = True
    unstable_ctrl_weight: float = 0.05
    unstable_ctrl_threshold: float = 1.0

    unit_circle_gap_loss: bool = False
    unit_circle_gap_weight: float = 1.0
    unit_circle_gap: float = 0.05

    upper_lipschitz_loss: bool = False
    upper_lip_weight: float = 1.0
    upper_lip_m_max: float = 1.25


@dataclasses.dataclass(kw_only=True)
class BFittingCfg:
    """B-matrix fitting method + hyperparameters."""

    method: str
    """``least_squares`` (closed-form pseudoinverse) or ``gradient`` (gradient
    descent with controllability regularization). Required."""

    ctrl_lambda: float = 0.1
    """Controllability regularization weight (gradient method only)."""
    train_steps: int = 500
    """Optimization steps (gradient method only)."""
    lr: float = 1.0e-3
    """Learning rate (gradient method only)."""


@dataclasses.dataclass(kw_only=True)
class TrainKoopmanCfg:
    """Top-level Koopman-training config."""

    approach: str
    """Training paradigm: ``two_phase`` (fixed encoder then train A+B) or
    ``joint`` (encoder + A + B trained together). Required."""
    k_type: str
    """A-matrix parameterization: ``cayley``, ``schur``, ``normalized``,
    ``complex_normal``, or ``unbounded``. Required."""
    encoder_type: str
    """Encoder family: ``linear`` or ``cayley``. Required."""

    # Architecture
    # state_dim and action_dim are derived from the dataset at training time
    # (env.observation_space and env.action_space, captured by gather_data).
    latent_dim: int = 8
    rho: float = 1.2
    """Spectral-radius bound for stable A parameterizations."""
    encoder_spec_norm: bool = False
    encoder_latent: int = 16
    init_b_in_a: bool = True
    """Initialize B in A's eigenbasis (two-phase only)."""
    prepend_state: bool = False
    """Prepend raw state x to latent: z = [x; g(x)]."""
    prepend_control: bool = False
    """Include u_base in prepended dims."""

    # B fitting
    b_fitting: BFittingCfg = dataclasses.field(
        default_factory=lambda: BFittingCfg(method="least_squares")
    )

    # Training
    horizon: int = 5
    batch_size: int = 2048
    num_workers: int = 4
    lr: float = 1.0e-2
    weight_decay: float = 1.0e-5
    num_epochs: int = 100
    recon_pretrain_epochs: int = 0
    """Encoder/decoder reconstruction-only pretrain epochs (0 = disabled)."""
    recon_pretrain_lr: float = 1.0e-3
    recon_pretrain_bilip: bool = False
    recon_pretrain_bilip_weight: float = 1.0

    loss_threshold: float = -10.0
    """Early-stopping threshold on training loss."""

    scheduler_step: int = 100
    scheduler_gamma: float = 0.5
    cosine_scheduler: bool = False
    cosine_T_max: int = 200
    cosine_eta_min: float = 1.0e-6
    riemannian_optimizer: bool = True
    """Use geoopt RiemannianAdam (required by orthogonal layers)."""
    vectorize_rollout: bool = True
    """Closed-form vectorized rollout (faster, more memory)."""
    torch_compile: bool = False
    grad_clip: bool = False
    grad_clip_type: str = "norm"
    """``norm`` (clip_grad_norm_) or ``value`` (clip_grad_value_)."""
    grad_clip_value: float = 1.0

    losses: LossesCfg = dataclasses.field(default_factory=LossesCfg)

    seed: int = 42
    log_interval: int = 10
    experiment_name: str = "pendulum_default"
    """Subdirectory under ``train_koopman/weights/``."""
    dataset_name: str = "pendulum_default"
    """Dataset to load from ``data/datasets/<dataset_name>.npz``."""
