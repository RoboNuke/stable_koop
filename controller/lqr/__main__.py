"""CLI: ``python -m controller.lqr --config <yaml>``.

Loads a trained Koopman model, fits the LQR controller, and runs the
unified stability analysis. The bound formula is picked from the
Koopman model's ``prepend_state`` flag; see
:mod:`controller.lqr.lqr_analysisv2` for details.

Writes ``eigen_variables.yaml`` (everything computed) + ``lqr.pt`` to
``controller/lqr/weights/<output_name>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from config.manager import ConfigManager, LQRControllerCfg
from controller.lqr.lqr_analysis import setup_lqr
from controller.lqr.lqr_analysisv2 import run_stability_report
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
    model, train_cfg, state_dim, action_dim = load_koopman_experiment(
        cfg.koopman_experiment_name, device
    )
    ds = load_dataset(train_cfg.dataset_name)
    aug_trajectories = augment_trajectories(ds, train_cfg.augmentation)

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    u_max = float(max(abs(ds.act_space_high).max(), abs(ds.act_space_low).max()))

    # --- Solve LQR ---
    print("\n" + "=" * 50)
    print("  LQR Fit")
    print("=" * 50)
    lqr, Q, R_cost, B_scale = setup_lqr(
        A, B_mat, cfg, state_dim=state_dim, action_dim=action_dim
    )

    # --- Unified stability analysis ---
    sa = cfg.stability_analysis
    variables = run_stability_report(
        model=model,
        lqr=lqr,
        A=A,
        B_mat=B_mat,
        Q=Q,
        R_cost=R_cost,
        real_state_dim=state_dim,
        u_max=u_max,
        aug_trajectories=aug_trajectories,
        device=device,
        epsilon_x=sa.epsilon_x,
        eta=sa.eta,
        q_scale=cfg.q_scale,
        r_scale=cfg.r_scale,
        use_optimization=sa.use_optimization,
    )
    variables["B_scale"] = float(B_scale)

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
