"""Write ``model_performance.yaml`` + ``config.yaml`` for a trained Koopman.

Centralizes the post-training disk writes so :mod:`train_koopman.joint` and
:mod:`train_koopman.two_phase` produce identical layouts.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from config.manager import TrainKoopmanCfg
from controller.lqr.lqr_analysisv2 import (
    controllability_fit_report,
    dataset_identifiability_report,
    one_step_pred_error_report,
)


def save_model_performance(
    *,
    out_dir: Path,
    model,
    train_cfg: TrainKoopmanCfg,
    aug_trajectories,
    device,
    training_stats: dict,
    b_fitting_summary: dict | None = None,
    ds=None,
) -> Path:
    """Compute post-training diagnostics, print them, and write ``model_performance.yaml``.

    Also writes a YAML copy of ``train_cfg`` to ``config.yaml`` in ``out_dir``.

    Diagnostics included (each section is the dict returned by the corresponding
    report helper):

    * ``training_summary``  — best loss / epoch / per-component best values.
    * ``one_step_pred_error`` — latent + state one-step error stats over the
      training transitions.
    * ``controllability_fit`` — encoder lower-Lipschitz ``m``, ``(A, B)``
      controllability rank, open-loop eigenvalues, spectral norms of A and B.
    * ``b_fitting``  — optional; passed in by the caller when a B-refit ran.

    Returns the path to ``model_performance.yaml``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Diagnostics — these helpers also print to screen, which is fine.
    A = model.A.detach().cpu()
    B_mat = model.B_matrix.detach().cpu()
    print("\n" + "=" * 50)
    print("  Model Performance Summary")
    print("=" * 50)
    identifiability_stats = dataset_identifiability_report(
        model, aug_trajectories, device,
    )
    print()
    pred_stats = one_step_pred_error_report(model, aug_trajectories, device)
    print()
    ctrl_stats = controllability_fit_report(
        model, A, B_mat, aug_trajectories, device,
        encoder_lipschitz_batch_size=train_cfg.encoder_lipschitz_batch_size,
    )

    performance: dict = {
        "training_summary": training_stats,
        "dataset_identifiability": identifiability_stats,
        "one_step_pred_error": pred_stats,
        "controllability_fit": ctrl_stats,
    }
    if b_fitting_summary is not None:
        performance["b_fitting"] = b_fitting_summary

    # Optional voxel-binned one-step state-error visualization.
    if train_cfg.visualization.enabled:
        from train_koopman.visualize import save_voxel_visualization
        if ds is None:
            print(
                "  visualization.enabled=True but the raw dataset was not "
                "passed to save_model_performance; skipping."
            )
        else:
            viz_summary = save_voxel_visualization(
                out_dir=out_dir,
                model=model,
                ds=ds,
                aug_trajectories=aug_trajectories,
                viz_cfg=train_cfg.visualization,
                device=device,
            )
            if viz_summary is not None:
                performance["visualization"] = viz_summary

    perf_path = out_dir / "model_performance.yaml"
    with perf_path.open("w") as f:
        yaml.dump(performance, f, default_flow_style=False, sort_keys=False)
    print(f"\nModel performance saved to {perf_path}")

    config_path = out_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(
            {"train_koopman_cfg": dataclasses.asdict(train_cfg)},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Training config saved to {config_path}")

    return perf_path
