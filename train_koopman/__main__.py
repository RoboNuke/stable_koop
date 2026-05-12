"""CLI entry: ``python -m train_koopman --config <yaml>``.

Dispatches on ``TrainKoopmanCfg.approach`` to the appropriate paradigm.
"""

from __future__ import annotations

import argparse

from config.manager import ConfigManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Koopman model.")
    parser.add_argument("--config", required=True, help="Per-stage train_koopman YAML.")
    args = parser.parse_args()

    cfg = ConfigManager.load_stage(args.config, "train_koopman_cfg")
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
    print(f"\nTraining complete. Weights at: {out}")


if __name__ == "__main__":
    main()
