"""Controller utilities + analysis (controller-type-agnostic and per-type)."""

from controller.controller_analysis import (
    compute_encoder_lipschitz,
    compute_encoder_lipschitz_bounds,
    compute_latent_errors,
    compute_max_latent_diff,
    compute_state_recon_errors,
    control_analysis,
    count_steps_under_threshold,
    latent_error_to_state_error,
    max_tolerable_model_error,
    spectral_radius,
    state_error_to_latent_error,
    transient_constant,
)

__all__ = [
    "compute_encoder_lipschitz",
    "compute_encoder_lipschitz_bounds",
    "compute_latent_errors",
    "compute_max_latent_diff",
    "compute_state_recon_errors",
    "control_analysis",
    "count_steps_under_threshold",
    "latent_error_to_state_error",
    "max_tolerable_model_error",
    "spectral_radius",
    "state_error_to_latent_error",
    "transient_constant",
]
