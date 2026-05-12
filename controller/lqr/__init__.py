"""LQR controller + analysis utilities."""

from controller.lqr.lqr import LQR
from controller.lqr.lqr_analysis import (
    compute_BtPB,
    compute_lyapunov_params,
    lyapunov_gamma,
    optimize_alpha_P,
    optimize_lyapunov_P,
    run_sdp_optimization,
    setup_lqr,
)

__all__ = [
    "LQR",
    "compute_BtPB",
    "compute_lyapunov_params",
    "lyapunov_gamma",
    "optimize_alpha_P",
    "optimize_lyapunov_P",
    "run_sdp_optimization",
    "setup_lqr",
]
