"""CLI entry: ``python -m train_koopman --config <yaml> [flags]``.

The config YAML carries both the ``train_koopman_cfg`` and
``<controller>_controller_cfg`` headers (currently ``lqr_controller_cfg`` is
the only controller type supported). One invocation runs Koopman training
and then fits the controller on the trained model.

Flags
-----

``--skip_train``
    Skip Koopman training entirely; run only the controller-fit step. The
    model is loaded from disk. The script prompts interactively for a
    fresh controller experiment name so the new fit lands in a new subdir
    under ``<koopman_dir>/lqr/`` instead of overwriting the previous one.

``--koopman_path <path>``
    Override the Koopman model checkpoint location. Accepts either a
    directory containing ``koopman_ckpt.pt`` or the ``.pt`` file directly.
    When absent, the controller cfg's ``koopman_experiment_name`` field
    derives the standard ``results/<name>/koopman_ckpt.pt`` path.
"""

from __future__ import annotations

import argparse

from config.manager import ConfigManager


def _run_training(config_path: str) -> str:
    cfg = ConfigManager.load_stage(config_path, "train_koopman_cfg")
    print(f"Training paradigm: {cfg.approach}")

    if cfg.approach == "two_phase":
        from train_koopman.two_phase import run
    elif cfg.approach == "joint":
        from train_koopman.joint import run
    else:
        raise ValueError(
            f"Unknown TrainKoopmanCfg.approach {cfg.approach!r}; "
            f"expected one of: two_phase, joint"
        )

    out = run(cfg)
    print(f"\nTraining complete. Results at: {out}")
    return out


def _prompt_controller_name() -> str:
    return input("Controller Experiment name...\n").strip()


def _run_controller(
    config_path: str,
    koopman_path: str | None,
    output_name: str | None,
    run_traj_eval: bool,
) -> str:
    """Fit the controller named by whichever ``*_controller_cfg`` header
    appears in the combined YAML. Currently only ``lqr_controller_cfg`` is
    supported."""
    from controller.lqr.__main__ import run as run_lqr

    cfg = ConfigManager.load_stage(config_path, "lqr_controller_cfg")
    print("\nFitting LQR controller on trained Koopman model.")
    return run_lqr(
        cfg,
        koopman_path=koopman_path,
        output_name_override=output_name,
        run_traj_eval=run_traj_eval,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Koopman model and fit its controller."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Combined train_koopman + controller YAML "
        "(e.g. config/exp_cfgs/train_koopman/lqr/<exp>.yaml).",
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip Koopman training; run only the controller-fit step. "
        "Prompts interactively for a fresh controller experiment name.",
    )
    parser.add_argument(
        "--koopman_path",
        default=None,
        help="Override the Koopman checkpoint location used by the controller "
        "fitter. Accepts a directory or a direct .pt file. When absent, "
        "results/<koopman_experiment_name>/ is used.",
    )
    parser.add_argument(
        "--no_traj_eval",
        action="store_true",
        help="Skip the trajectory-level evaluation (env rollouts + per-step "
        "Koopman diagnostics saved to eval_traj.npz) that runs at the end "
        "of the controller fit.",
    )
    args = parser.parse_args()

    output_name_override: str | None = None
    if args.skip_train:
        print("--skip_train: skipping Koopman training; loading saved model.")
        output_name_override = _prompt_controller_name()
    else:
        _run_training(args.config)

    _run_controller(
        args.config,
        args.koopman_path,
        output_name_override,
        run_traj_eval=not args.no_traj_eval,
    )


if __name__ == "__main__":
    main()
