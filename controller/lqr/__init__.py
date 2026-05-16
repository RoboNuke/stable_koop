"""LQR controller + analysis utilities."""

from controller.lqr.lqr import LQR
from controller.lqr.lqr_analysis import (
    compute_lyapunov_params,
    optimize_alpha_P,
    optimize_lyapunov_P,
    setup_lqr,
)
from controller.lqr.lqr_analysisv2 import (
    action_budget_report,
    controllability_fit_report,
    gamma_coverage_report,
    generate_P_and_bound,
    lqr_fit_report,
    one_step_pred_error_report,
    run_stability_report,
)

__all__ = [
    "LQR",
    "action_budget_report",
    "compute_lyapunov_params",
    "controllability_fit_report",
    "gamma_coverage_report",
    "generate_P_and_bound",
    "lqr_fit_report",
    "one_step_pred_error_report",
    "optimize_alpha_P",
    "optimize_lyapunov_P",
    "run_stability_report",
    "setup_lqr",
]
