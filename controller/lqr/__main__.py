"""CLI: ``python -m controller.lqr --config <yaml>``.

Loads a trained Koopman model, fits the LQR controller, and runs whichever
stability bounds are enabled in :class:`LQRControllerCfg.stability_analysis`:

* ``use_eigen_bound`` — spectral-radius / transient-constant bound.
* ``use_lyapunov_bound`` — Lyapunov-equation bound (optionally with SDP
  P-optimization via ``optimize_lyapunov_P``).
* ``use_m_free_bound`` — Lipschitz-m-free bound (no encoder lower-Lipschitz
  assumption).
* ``use_alpha_bound`` — alpha-bound state-extraction analysis. Requires
  the Koopman model to have been trained with ``prepend_state=True``.

All four orchestrators live in :mod:`controller.lqr.lqr_analysis`.

Writes ``eigen_variables.yaml`` (everything computed) + ``lqr.pt`` to
``controller/lqr/weights/<output_name>/``.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import yaml

from config.manager import AugmentationCfg, ConfigManager, LQRControllerCfg
from controller.controller_analysis import (
    compute_encoder_lipschitz_bounds,
    compute_latent_errors,
    control_analysis,
    spectral_radius,
)
from controller.lqr.lqr_analysis import (
    alpha_bound_report,
    eigen_bound_report,
    lyapunov_bound_report,
    m_free_bound_report,
    setup_lqr,
)
from data.augmentation import augment_trajectories
from data.dataloader import load_dataset
from train_koopman.checkpointing import load_koopman_experiment, make_device


def _out_dir(name: str) -> Path:
    out = Path("controller") / "lqr" / "weights" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run(cfg: LQRControllerCfg) -> str:
    device = make_device()
    out_dir = _out_dir(cfg.output_name)

    # --- Load Koopman model + reconstruct training augmentation ---
    model, koop_cfg = load_koopman_experiment(cfg.koopman_experiment_name, device)
    aug_cfg = AugmentationCfg(
        prepend_base_action=koop_cfg["prepend_base_action"],
        use_action_delta=koop_cfg["use_action_delta"],
        obs_scale_source=koop_cfg.get("obs_scale_source", "env"),
        act_scale_source=koop_cfg.get("act_scale_source", "env"),
    )
    ds = load_dataset(koop_cfg["dataset_name"])
    aug_trajectories = augment_trajectories(ds, aug_cfg)

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    u_max = float(max(abs(ds.act_space_high).max(), abs(ds.act_space_low).max()))

    # --- Shared diagnostics: open-loop spectrum + controllability + Lipschitz ---
    print("\n" + "=" * 50)
    print("  Open-loop diagnostics")
    print("=" * 50)
    rho_A = spectral_radius(A)
    B_sigma_max = torch.linalg.norm(B_mat, ord=2).item()
    print(f"  Spectral radius of A (open):           {rho_A:.6f}")
    print(f"  B largest singular value:              {B_sigma_max:.6f}")
    ctrl_rank = control_analysis(A, B_mat)

    print("\n" + "=" * 50)
    print("  Encoder Lipschitz Bounds")
    print("=" * 50)
    m_gx, L_gx, m_full, L_full = compute_encoder_lipschitz_bounds(model, aug_trajectories, device)
    m = m_full if m_full is not None else 1.0
    if m_gx is not None:
        print(f"  Encoder lower Lipschitz (m, g(x)):     {m_gx:.6f}")
        print(f"  Encoder upper Lipschitz (L, g(x)):     {L_gx:.6f}")
        print(f"  Encoder lower Lipschitz (m, full):     {m_full:.6f}")
        print(f"  Encoder upper Lipschitz (L, full):     {L_full:.6f}")

    # --- LQR fit (used by every bound below) ---
    flat = {
        **koop_cfg,
        **dataclasses.asdict(cfg),
        **dataclasses.asdict(cfg.stability_analysis),
    }
    print("\n" + "=" * 50)
    print("  LQR Fit")
    print("=" * 50)
    lqr, Q, R_cost, B_scale = setup_lqr(A, B_mat, flat)
    print(f"  Q scale:                               {cfg.q_scale}")
    print(f"  R scale:                               {cfg.r_scale}")
    print(f"  LQR gain norm (||F||):                 {lqr.gain_norm.item():.6f}")

    err_mean, err_std = compute_latent_errors(model, aug_trajectories, device, None)

    sa = cfg.stability_analysis
    variables: dict = {
        "m": float(m),
        "gain_norm": float(lqr.gain_norm.item()),
        "max_tracking_error_x": float(cfg.max_tracking_error_x),
        "max_displacement_x": float(cfg.max_displacement_x),
        "latent_err_mean": float(err_mean),
        "latent_err_std": float(err_std),
        "ctrl_rank": int(ctrl_rank),
        "A": A.numpy().tolist(),
        "B": B_mat.numpy().tolist(),
        "F": lqr.F.numpy().tolist(),
    }
    if m_gx is not None:
        variables.update({"m_gx": float(m_gx), "L_gx": float(L_gx),
                          "m_full": float(m_full), "L_full": float(L_full)})

    # --- Bound reports (each runs only if its flag is on) ---
    if sa.use_eigen_bound:
        variables["eigen_bound"] = eigen_bound_report(
            A=A, B_mat=B_mat, lqr=lqr, m=m,
            max_tracking_error_x=cfg.max_tracking_error_x,
            max_displacement_x=cfg.max_displacement_x,
            err_mean=err_mean, err_std=err_std,
        )

    if sa.use_lyapunov_bound:
        variables["lyapunov_bound"] = lyapunov_bound_report(
            model=model, A=A, B_mat=B_mat, lqr=lqr, Q=Q, R_cost=R_cost, m=m,
            max_tracking_error_x=cfg.max_tracking_error_x,
            max_displacement_x=cfg.max_displacement_x,
            err_mean=err_mean, err_std=err_std,
            scale_B=cfg.scale_B, B_scale=B_scale,
            optimize_lyapunov_P_flag=sa.optimize_lyapunov_P,
            aug_trajectories=aug_trajectories, device=device,
        )

    if sa.use_m_free_bound:
        variables["m_free_bound"] = m_free_bound_report(
            model=model, A=A, B_mat=B_mat, lqr=lqr, Q=Q, R_cost=R_cost,
            u_max=u_max,
            max_tracking_error_x=cfg.max_tracking_error_x,
            m=m,
            err_mean=err_mean, err_std=err_std,
            optimize_lyapunov_P_flag=sa.optimize_lyapunov_P,
            aug_trajectories=aug_trajectories, device=device,
        )

    if sa.use_alpha_bound:
        variables["alpha_bound"] = alpha_bound_report(
            model=model, A=A, B_mat=B_mat, lqr=lqr,
            real_state_dim=ds.state_dim,
            alpha_epsilon_x=sa.alpha_epsilon_x,
            alpha_eta=sa.alpha_eta,
            err_mean=err_mean, err_std=err_std,
            aug_trajectories=aug_trajectories, device=device,
        )

    # --- Persist results ---
    out_path = out_dir / "eigen_variables.yaml"
    with out_path.open("w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"\nVariables saved to {out_path}")

    lqr_path = out_dir / "lqr.pt"
    torch.save(
        {"F": lqr.F, "P": lqr.P, "Q": Q, "R": R_cost, "B_scale": float(B_scale)},
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
