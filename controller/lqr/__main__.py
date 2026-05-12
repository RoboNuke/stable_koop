"""CLI: ``python -m controller.lqr --config <yaml>``.

Loads a trained Koopman model from ``train_koopman/weights/<exp>/``, fits the
LQR controller per the LQRControllerCfg, runs the configured stability bounds,
and writes results to ``controller/lqr/weights/<output_name>/``.

This replaces the legacy ``launch/tune_koop_model.py`` entry point — it is the
controller-fitting stage of the pipeline.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import yaml

from config.manager import ConfigManager, LQRControllerCfg
from controller.controller_analysis import (
    compute_encoder_lipschitz_bounds,
    compute_latent_errors,
    control_analysis,
    latent_error_to_state_error,
    max_tolerable_model_error,
    spectral_radius,
    state_error_to_latent_error,
    transient_constant,
)
from controller.lqr.lqr_analysis import (
    compute_BtPB,
    compute_lyapunov_params,
    lyapunov_gamma,
    run_sdp_optimization,
    setup_lqr,
)
from data.dataloader import load_dataset
from data.gather_data import augment_perturbed_trajectories
from train_koopman.checkpointing import build_koopman_model, load_checkpoint, make_device


def _out_dir(name: str) -> Path:
    out = Path("controller") / "lqr" / "weights" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_koopman_artifacts(experiment_name: str, device):
    """Load the Phase-2 Koopman ``(model, koopman_cfg, dataset_name)`` tuple."""
    ckpt_path = Path("train_koopman") / "weights" / experiment_name / "koopman_ckpt.pt"
    raw = torch.load(ckpt_path, map_location=device)
    koop_cfg_flat = raw["config"]
    model, _ = build_koopman_model(koop_cfg_flat, augment=True, device=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded Koopman weights from {ckpt_path}")
    return model, koop_cfg_flat


def run(cfg: LQRControllerCfg) -> str:
    device = make_device()
    out_dir = _out_dir(cfg.output_name)

    model, koop_cfg = _load_koopman_artifacts(cfg.koopman_experiment_name, device)

    flat = {
        **koop_cfg,
        **dataclasses.asdict(cfg),
        **dataclasses.asdict(cfg.stability_analysis),
    }

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    rho_A = spectral_radius(A)
    B_sigma_max = torch.linalg.norm(B_mat, ord=2).item()
    print(f"  Spectral radius of A (open):           {rho_A:.6f}")
    print(f"  B largest singular value:              {B_sigma_max:.6f}")
    ctrl_rank = control_analysis(A, B_mat)

    # Encoder Lipschitz against the dataset used for training.
    ds = load_dataset(koop_cfg["dataset_name"])
    import numpy as np

    koopman_obs_scale = np.concatenate([ds.obs_scale, ds.act_scale])
    aug_trajectories = augment_perturbed_trajectories(
        ds.perturbed_trajectories,
        augment=True,
        obs_scale=koopman_obs_scale,
        act_scale=ds.act_scale,
    )
    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(
        model, aug_trajectories, device
    )
    m = m_full if m_full is not None else 1.0

    # LQR + closed-loop diagnostics.
    lqr, Q, R_cost, B_scale = setup_lqr(A, B_mat, flat)
    gain_norm = lqr.gain_norm.item()
    closed_loop = lqr.closed_loop
    rho_cl = spectral_radius(closed_loop)
    C = transient_constant(closed_loop)
    P, kappa_P, rho_sq, P_eigvals = compute_lyapunov_params(lqr, Q, R_cost)
    BtPB = compute_BtPB(lqr, B_mat, P)
    print(f"  LQR gain norm (||F||):                 {gain_norm:.6f}")
    print(f"  Spectral radius A-BF (closed):         {rho_cl:.6f}")
    print(f"  Transient constant (C):                {C:.6f}")
    print(f"  κ(P):                                  {kappa_P:.6f}")
    print(f"  B^T P B:                               {BtPB:.6f}")

    max_tracking_error_x = cfg.max_tracking_error_x
    max_displacement_x = cfg.max_displacement_x
    max_tracking_error_latent = state_error_to_latent_error(max_tracking_error_x, m)
    eta = state_error_to_latent_error(max_displacement_x, m)
    max_runtime_error_latent = max_tolerable_model_error(
        rho_cl, C, max_tracking_error_latent, eta
    )
    residual_ctrl_budget = max_tracking_error_latent * gain_norm

    err_mean, err_std = compute_latent_errors(model, aug_trajectories, device, None)

    sdp_result = run_sdp_optimization(lqr, max_tracking_error_latent, eta, flat) if cfg.stability_analysis.use_lyapunov_bound else None
    gamma_lyapunov = (
        lyapunov_gamma(max_tracking_error_latent, rho_sq, kappa_P, eta)
        if cfg.stability_analysis.use_lyapunov_bound
        else None
    )

    variables = {
        "m": float(m),
        "gain_norm": float(gain_norm),
        "rho": float(rho_cl),
        "C": float(C),
        "kappa_P": float(kappa_P),
        "BtPB": float(BtPB),
        "max_tracking_error_x": float(max_tracking_error_x),
        "max_displacement_x": float(max_displacement_x),
        "max_tracking_error_latent": float(max_tracking_error_latent),
        "eta": float(eta),
        "max_runtime_error_latent": float(max_runtime_error_latent),
        "residual_ctrl_budget": float(residual_ctrl_budget),
        "latent_error_mean": float(err_mean),
        "latent_error_std": float(err_std),
        "gamma_lyapunov": float(gamma_lyapunov) if gamma_lyapunov is not None else None,
        "sdp_result": (
            {
                "rho_sq": float(sdp_result[0]),
                "kappa": float(sdp_result[1]),
                "gamma": float(sdp_result[2]),
            }
            if sdp_result is not None
            else None
        ),
        "A": A.numpy().tolist(),
        "B": B_mat.numpy().tolist(),
        "F": lqr.F.numpy().tolist(),
        "P": P.numpy().tolist(),
    }
    out_path = out_dir / "eigen_variables.yaml"
    with out_path.open("w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"\nVariables saved to {out_path}")

    lqr_path = out_dir / "lqr.pt"
    torch.save(
        {"F": lqr.F, "P": P, "Q": Q, "R": R_cost, "B_scale": float(B_scale)},
        lqr_path,
    )
    print(f"LQR weights saved to {lqr_path}")
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit LQR controller on a trained Koopman model.")
    parser.add_argument("--config", required=True, help="Per-stage LQR controller YAML.")
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "lqr_controller_cfg")
    run(cfg)


if __name__ == "__main__":
    main()
