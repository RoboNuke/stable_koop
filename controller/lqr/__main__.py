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
from controller.lqr.lqr_analysis import build_C, setup_lqr
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


def _normalize_eigvals(evs: list) -> tuple[list, bool]:
    """Convert any Python ``complex`` values in ``evs`` to ``[real, imag]`` pairs.

    Older training runs wrote ``open_loop_eigvals`` as a YAML sequence of
    ``!!python/complex`` scalars, which can't be reloaded with
    ``yaml.safe_load``. Returns ``(normalized_list, was_changed)`` so the
    caller can decide whether to rewrite the file.
    """
    out = []
    changed = False
    for z in evs:
        if isinstance(z, complex):
            out.append([float(z.real), float(z.imag)])
            changed = True
        else:
            out.append(z)
    return out, changed


def _read_perf_yaml(perf_path: Path) -> tuple[dict, bool]:
    """Load a model_performance.yaml, normalizing any ``!!python/complex`` eigvals.

    Uses ``yaml.full_load`` so files written by older training runs (which
    embedded Python complex numbers) parse cleanly. Returns
    ``(perf_dict, needs_rewrite)`` where ``needs_rewrite`` is True iff any
    eigvals were normalized — the caller should overwrite the file to
    purge the bad tags.
    """
    perf = yaml.full_load(perf_path.read_text()) or {}
    ctrl = perf.get("controllability_fit")
    needs_rewrite = False
    if isinstance(ctrl, dict):
        evs = ctrl.get("open_loop_eigvals")
        if isinstance(evs, list):
            ctrl["open_loop_eigvals"], needs_rewrite = _normalize_eigvals(evs)
    return perf, needs_rewrite


def _load_cached_model_performance(
    koopman_dir: Path,
) -> tuple[dict | None, dict | None, dict | None]:
    """Pull cached dataset/diagnostic sections from training YAML.

    Returns ``(pred_stats, ctrl_stats, identifiability_stats)`` so
    ``run_stability_report`` can skip the underlying recompute. Any entry
    is ``None`` if the YAML is missing or doesn't carry that section, in
    which case the caller falls back to recomputing.
    """
    perf_path = koopman_dir / "model_performance.yaml"
    if not perf_path.is_file():
        return None, None, None
    perf, _ = _read_perf_yaml(perf_path)
    pred = perf.get("one_step_pred_error")
    ctrl = perf.get("controllability_fit")
    ident = perf.get("dataset_identifiability")
    return pred, ctrl, ident


def _backfill_model_performance(
    koopman_dir: Path,
    *,
    pred_stats: dict,
    ctrl_stats: dict,
    identifiability_stats: dict | None = None,
) -> None:
    """Backfill missing ``one_step_pred_error`` / ``controllability_fit``.

    Writes to the koopman-level ``model_performance.yaml`` so future
    controller fits can reuse them without recomputing. If the file
    doesn't exist (e.g. older checkpoint, or training crashed before
    writing it), creates it with just the two cached sections. The
    training step (:func:`train_koopman.save_performance.save_model_performance`)
    is still the source of truth for the full set of sections
    (including ``training_summary``).

    Also purges any ``!!python/complex`` eigvals from an existing
    ``controllability_fit`` section so the file can be read back by
    ``yaml.safe_load``.
    """
    perf_path = koopman_dir / "model_performance.yaml"
    if perf_path.is_file():
        perf, needs_rewrite_for_complex = _read_perf_yaml(perf_path)
    else:
        perf, needs_rewrite_for_complex = {}, False
    changed = needs_rewrite_for_complex
    if "one_step_pred_error" not in perf:
        perf["one_step_pred_error"] = pred_stats
        changed = True
    if "controllability_fit" not in perf:
        perf["controllability_fit"] = ctrl_stats
        changed = True
    if identifiability_stats is not None and "dataset_identifiability" not in perf:
        perf["dataset_identifiability"] = identifiability_stats
        changed = True
    if not changed:
        return
    with perf_path.open("w") as f:
        yaml.dump(perf, f, default_flow_style=False, sort_keys=False)
    print(f"[lqr] backfilled model_performance.yaml at {perf_path}")


def _backfill_train_config(koopman_dir: Path, train_cfg) -> None:
    """Write ``config.yaml`` next to the koopman checkpoint if it's missing."""
    config_path = koopman_dir / "config.yaml"
    if config_path.is_file():
        return
    with config_path.open("w") as f:
        yaml.dump(
            {"train_koopman_cfg": dataclasses.asdict(train_cfg)},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"[lqr] backfilled training config.yaml at {config_path}")


def run(
    cfg: LQRControllerCfg,
    *,
    koopman_path: str | None = None,
    output_name_override: str | None = None,
    run_traj_eval: bool = True,
    quiet_diagnostics: bool = False,
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
    aug_trajectories = augment_trajectories(
        ds, train_cfg.augmentation, verbose=not quiet_diagnostics
    )

    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    u_max = float(max(abs(ds.act_space_high).max(), abs(ds.act_space_low).max()))

    # --- Solve LQR ---
    print("\n" + "=" * 50)
    print("  LQR Fit")
    print("=" * 50)
    lqr, Q, R_cost, B_scale = setup_lqr(
        A, B_mat, cfg, action_dim=action_dim
    )
    C_np = build_C(cfg.C_mask, latent_dim=A.shape[0])

    # --- Load cached stats from the koopman training run, if available ---
    cached_pred, cached_ctrl, cached_ident = _load_cached_model_performance(koopman_dir)

    # --- Unified stability analysis ---
    sa = cfg.stability_analysis
    variables = run_stability_report(
        model=model,
        lqr=lqr,
        A=A,
        B_mat=B_mat,
        Q=Q,
        R_cost=R_cost,
        C=C_np,
        u_max=u_max,
        aug_trajectories=aug_trajectories,
        device=device,
        epsilon_x=sa.epsilon_x,
        ctrl_percentages=list(sa.ctrl_percentages),
        optimizer=sa.optimizer,
        beta_rho_target=sa.beta_rho_target,
        mode_projection_groups=sa.mode_projection_groups,
        quiet_diagnostics=quiet_diagnostics,
        encoder_lipschitz_batch_size=train_cfg.encoder_lipschitz_batch_size,
        cached_pred_stats=cached_pred,
        cached_ctrl_stats=cached_ctrl,
        cached_identifiability=cached_ident,
    )
    variables["B_scale"] = float(B_scale)

    # Backfill the koopman-level YAMLs if they were missing or incomplete,
    # so future controller fits can reuse the cached stats instead of
    # recomputing them. config.yaml is written from the loaded train_cfg;
    # model_performance.yaml gets any sections that weren't present (and
    # has any legacy ``!!python/complex`` eigvals purged on rewrite).
    _backfill_train_config(koopman_dir, train_cfg)
    _backfill_model_performance(
        koopman_dir,
        pred_stats=variables["pred_error_stats"],
        ctrl_stats=variables["controllability_fit"],
        identifiability_stats=variables.get("dataset_identifiability"),
    )

    # --- Control-budget summary plot (always; doesn't need voxel viz cfg) ---
    from controller.lqr.visualize import save_ctrl_sweep_plot
    viz_summaries: dict = {
        "ctrl_sweep_plot": save_ctrl_sweep_plot(
            out_dir=out_dir,
            sweep=variables["ctrl_pct_sweep"],
            highlight_points=variables["ctrl_pct_results"],
        )
    }

    # --- Voxel γ-compliance heatmaps (only when voxel viz is enabled) ---
    # One compliance heatmap per ctrl_percentage in the config list (each
    # uses η = pct · ||B||·u_max).
    if train_cfg.visualization.enabled:
        from controller.lqr.visualize import save_compliance_visualization
        for entry in variables["ctrl_pct_results"]:
            pct = float(entry["ctrl_percentage"])
            eta_at = float(entry["eta"])
            gamma_at = float(entry["gamma_max"])
            label = f"ctrl_pct_{int(round(pct * 100)):03d}"
            summary = save_compliance_visualization(
                out_dir=out_dir,
                model=model,
                ds=ds,
                aug_trajectories=aug_trajectories,
                viz_cfg=train_cfg.visualization,
                device=device,
                gamma_max=gamma_at,
                label=label,
                title_suffix=f"ctrl_pct={pct:.2f},  η={eta_at:.2e}",
            )
            if summary is not None:
                viz_summaries[label] = summary

    variables["visualization"] = viz_summaries

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

    # --- Post-fit eval (base + LQR via eval.multi_mode) ---
    if run_traj_eval:
        if not cfg.eval_cfg_path:
            raise ValueError(
                "lqr_controller_cfg.eval_cfg_path is empty; set it in the YAML "
                "or pass --no_eval to skip post-fit evaluation."
            )
        _run_post_fit_eval(
            cfg=cfg,
            controller_dir=out_dir,
            model=model,
            lqr=lqr,
            gamma_max=float(variables["bound"]["gamma_max"]),
            ds=ds,
            train_cfg=train_cfg,
            device=device,
            ctrl_variables=variables,
            ctrl_perf_path=perf_path,
        )

    return str(out_dir)


def _run_post_fit_eval(
    *,
    cfg: LQRControllerCfg,
    controller_dir: Path,
    model,
    lqr,
    gamma_max: float,
    ds,
    train_cfg,
    device,
    ctrl_variables: dict,
    ctrl_perf_path: Path,
) -> None:
    """Hand off to ``eval.multi_mode.run_multi_mode`` for base + LQR rollouts."""
    from data.augmentation import compute_act_scale, compute_obs_scale
    from eval.multi_mode import run_multi_mode
    from policy import make_policy

    print("\n" + "=" * 50)
    print("  Post-fit Eval (base + LQR via eval.multi_mode)")
    print("=" * 50)

    eval_cfg = ConfigManager.load_stage(cfg.eval_cfg_path, "eval_cfg")
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = gather_snap["env_name"]
    env_kwargs = gather_snap.get("env_kwargs", {}) or {}
    base_policy_snap = gather_snap["base_policy"]
    base_policy = make_policy(
        base_policy_snap["name"], **(base_policy_snap.get("params", {}) or {})
    )

    # Match the koopman's training-time normalization so the wrapper's
    # gamma_t lands in the same space as the LQR analysis above.
    aug_cfg = train_cfg.augmentation
    obs_scale = compute_obs_scale(aug_cfg, ds)
    act_scale = compute_act_scale(aug_cfg, ds)

    # Base results land in the koopman dir; LQR results live next to the
    # controller artifact.
    koopman_dir = controller_dir.parent.parent  # results/<koopman>/lqr/<out> -> results/<koopman>

    run_multi_mode(
        koopman_dir=koopman_dir,
        lqr_dir=controller_dir,
        eval_cfg=eval_cfg,
        env_name=env_name,
        env_kwargs=env_kwargs,
        base_policy=base_policy,
        koopman_model=model,
        device=device,
        lqr=lqr,
        gamma_max=gamma_max,
        residual_actor=None,  # no residual yet at this stage
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
    )

    ctrl_variables["eval"] = {
        "eval_cfg_path": cfg.eval_cfg_path,
        "koopman_dir": str(koopman_dir),
        "lqr_dir": str(controller_dir),
        "gamma_max": float(gamma_max),
    }
    with ctrl_perf_path.open("w") as f:
        yaml.dump(ctrl_variables, f, default_flow_style=False, sort_keys=False)
    print(f"ctrl_performance.yaml updated with eval results dir: {ctrl_perf_path}")


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
        "--no_eval", action="store_true",
        help="Skip the post-fit eval (base + LQR rollouts via eval.multi_mode).",
    )
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "lqr_controller_cfg")
    run(
        cfg,
        koopman_path=args.koopman_path,
        output_name_override=args.controller_output_name,
        run_traj_eval=not args.no_eval,
    )


if __name__ == "__main__":
    main()
