"""LQR controller fit + stability analysis on a trained Koopman model.

Output layout (nested under the Koopman experiment's results folder)::

    results/<koopman_exp>/lqr/<output_name>/
        ├── ctrl_performance.yaml   # everything printed by run_stability_report
        ├── config.yaml             # copy of the lqr_controller_cfg used
        └── lqr.pt                  # F / P / Q / R / B_scale

The koopman folder is determined by (in priority order):
  1. ``koopman_path`` kwarg passed to :func:`run` (typically from the
     ``--koopman_path`` CLI flag in ``train_koopman``).
  2. ``results/<cfg.koopman_experiment_name>/`` by default.

``output_name_override`` (forwarded from the ``--controller_output_name``
CLI flag) replaces ``cfg.output_name`` for the inner subdirectory; it must
be supplied whenever the caller wants the result written somewhere other
than the default.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import torch
import yaml

from config.manager import ConfigManager, LQRControllerCfg
from controller.lqr.lqr_analysis import setup_lqr
from controller.lqr.lqr_analysisv2 import run_stability_report
from data.augmentation import augment_trajectories
from data.dataloader import load_dataset
from train_koopman.checkpointing import load_koopman_experiment, make_device


CONTROLLER_TYPE = "lqr"


def _resolve_koopman_dir(cfg: LQRControllerCfg, koopman_path: str | None) -> Path:
    """Pick the directory containing ``koopman_ckpt.pt`` for this fit."""
    if koopman_path is not None:
        p = Path(koopman_path)
        if p.is_file():
            return p.parent
        return p
    return Path("results") / cfg.koopman_experiment_name


def run(
    cfg: LQRControllerCfg,
    *,
    koopman_path: str | None = None,
    output_name_override: str | None = None,
    run_traj_eval: bool = True,
) -> str:
    device = make_device()

    koopman_dir = _resolve_koopman_dir(cfg, koopman_path)
    ckpt_path = koopman_dir / "koopman_ckpt.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Koopman checkpoint not found at {ckpt_path}. "
            "Pass --koopman_path or check the koopman_experiment_name field."
        )

    output_name = output_name_override or cfg.output_name
    out_dir = koopman_dir / CONTROLLER_TYPE / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[lqr] koopman dir: {koopman_dir}")
    print(f"[lqr] output dir:  {out_dir}")

    # --- Load Koopman model + reconstruct training augmentation ---
    model, train_cfg, state_dim, action_dim = load_koopman_experiment(
        cfg.koopman_experiment_name, device, ckpt_path=ckpt_path,
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

    # --- Voxel γ-compliance visualization (uses the Koopman's viz cfg) ---
    if train_cfg.visualization.enabled:
        from controller.lqr.visualize import save_compliance_visualization
        viz_summary = save_compliance_visualization(
            out_dir=out_dir,
            model=model,
            ds=ds,
            aug_trajectories=aug_trajectories,
            viz_cfg=train_cfg.visualization,
            device=device,
            gamma_max=float(variables["bound"]["gamma_max"]),
        )
        if viz_summary is not None:
            variables["visualization"] = viz_summary

    # --- Persist results ---
    perf_path = out_dir / "ctrl_performance.yaml"
    with perf_path.open("w") as f:
        yaml.dump(variables, f, default_flow_style=False, sort_keys=False)
    print(f"\nController performance saved to {perf_path}")

    config_path = out_dir / "config.yaml"
    cfg_with_resolved_name = dataclasses.replace(cfg, output_name=output_name)
    with config_path.open("w") as f:
        yaml.dump(
            {"lqr_controller_cfg": dataclasses.asdict(cfg_with_resolved_name)},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Controller config saved to {config_path}")

    lqr_path = out_dir / "lqr.pt"
    torch.save(
        {"F": lqr.F, "P": lqr.P, "Q": Q, "R": R_cost, "B_scale": float(B_scale)},
        lqr_path,
    )
    print(f"LQR weights saved to {lqr_path}")

    # --- Trajectory-level eval (base policy + per-step diagnostics) ---
    if run_traj_eval:
        if not cfg.eval_cfg_path:
            raise ValueError(
                "lqr_controller_cfg.eval_cfg_path is empty; set it in the YAML "
                "or pass --no_traj_eval to skip trajectory evaluation."
            )
        _run_trajectory_eval(
            cfg=cfg,
            out_dir=out_dir,
            model=model,
            train_cfg=train_cfg,
            ds=ds,
            device=device,
            ctrl_variables=variables,
            ctrl_perf_path=perf_path,
        )

    return str(out_dir)


def _run_trajectory_eval(
    *,
    cfg: LQRControllerCfg,
    out_dir: Path,
    model,
    train_cfg,
    ds,
    device,
    ctrl_variables: dict,
    ctrl_perf_path: Path,
) -> None:
    """Roll out the base policy on the env, record per-step Koopman + policy data."""
    from data.env_builder import make_single_env
    from data.dataset_stats import _success_cfg_from_eval
    from eval.trajectory_eval import evaluate_with_trajectories, save_trajectory_npz
    from policy import make_policy

    print("\n" + "=" * 50)
    print("  Trajectory Evaluation (base policy + per-step diagnostics)")
    print("=" * 50)

    eval_cfg = ConfigManager.load_stage(cfg.eval_cfg_path, "eval_cfg")
    success_cfg = _success_cfg_from_eval(cfg.eval_cfg_path)

    # Env + base policy come from the gather snapshot embedded in the dataset.
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = gather_snap["env_name"]
    env_kwargs = gather_snap.get("env_kwargs", {}) or {}
    base_policy_snap = gather_snap["base_policy"]
    base_policy = make_policy(
        base_policy_snap["name"], **(base_policy_snap.get("params", {}) or {})
    )
    env = make_single_env(env_name=env_name, env_kwargs=env_kwargs)

    try:
        metrics, per_trajectory = evaluate_with_trajectories(
            env,
            base_policy,
            model=model,
            device=device,
            env_name=env_name,
            eval_cfg=eval_cfg,
            success_cfg=success_cfg,
            residual_policy=None,  # wired in for later; no residual yet
        )
    finally:
        env.close()

    traj_path = out_dir / "eval_traj.npz"
    save_trajectory_npz(traj_path, per_trajectory)
    print(f"Per-step eval trajectories saved to {traj_path}")

    # Merge eval task metrics into ctrl_performance.yaml.
    ctrl_variables["eval"] = {
        "eval_cfg_path": cfg.eval_cfg_path,
        "success_cfg": success_cfg,
        "metrics": metrics,
        "trajectory_npz": str(traj_path),
    }
    with ctrl_perf_path.open("w") as f:
        yaml.dump(ctrl_variables, f, default_flow_style=False, sort_keys=False)
    print(f"ctrl_performance.yaml updated with eval metrics: {ctrl_perf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit LQR controller on a trained Koopman model.")
    parser.add_argument("--config", required=True, help="LQR controller YAML (or combined yaml).")
    parser.add_argument("--koopman_path", default=None, help="Override the Koopman model directory.")
    parser.add_argument(
        "--controller_output_name", default=None,
        help="Override LQRControllerCfg.output_name (used as the subdir name "
             "under <koopman_dir>/lqr/).",
    )
    parser.add_argument(
        "--no_traj_eval", action="store_true",
        help="Skip the post-fit trajectory eval (env rollouts + eval_traj.npz).",
    )
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "lqr_controller_cfg")
    run(
        cfg,
        koopman_path=args.koopman_path,
        output_name_override=args.controller_output_name,
        run_traj_eval=not args.no_traj_eval,
    )


if __name__ == "__main__":
    main()
