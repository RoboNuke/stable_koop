"""Two-phase Koopman training paradigm.

Phase 1 — pretrain (encoder + A + B) with only the core losses (recon, pred,
latent-consistency); every optional loss is disabled via a
:func:`dataclasses.replace` clone of the ``LossesCfg``. Phase 2 — reinit B,
then run with the full loss suite.

Both phases reuse the canonical :func:`train_koopman.training_loop.train`
function so optimization details stay consistent.
"""

from __future__ import annotations

import dataclasses

import torch.nn as nn

from config.manager import TrainKoopmanCfg
from data.augmentation import augment_trajectories
from data.dataloader import load_dataset
from train_koopman.checkpointing import (
    build_koopman_model,
    experiment_dir,
    make_device,
    save_checkpoint,
)
from train_koopman.save_performance import save_model_performance
from train_koopman.training_loop import train


# Optional losses to disable during the phase-1 pretrain.
_PHASE1_DISABLED_LOSSES = (
    "controllability_loss",
    "bi_lipschitz_loss",
    "spectral_loss",
    "normality_loss",
    "cl_normality_loss",
    "b_eigen_loss",
    "b_scale_loss",
    "eig_spread_loss",
    "unstable_ctrl_loss",
    "upper_lipschitz_loss",
    "unit_circle_gap_loss",
)


def _phase1_cfg(train_cfg: TrainKoopmanCfg) -> TrainKoopmanCfg:
    """Return a copy of ``train_cfg`` with every optional loss disabled."""
    losses_off = dataclasses.replace(
        train_cfg.losses,
        **{name: False for name in _PHASE1_DISABLED_LOSSES},
    )
    return dataclasses.replace(train_cfg, losses=losses_off)


def run(train_cfg: TrainKoopmanCfg) -> str:
    """End-to-end two-phase training; returns the output experiment directory."""
    out_dir = experiment_dir(train_cfg.experiment_name)

    device = make_device()
    ds = load_dataset(train_cfg.dataset_name)
    state_dim = ds.state_dim
    action_dim = ds.action_dim

    model, _ = build_koopman_model(
        train_cfg, state_dim=state_dim, action_dim=action_dim, device=device
    )
    aug_trajectories = augment_trajectories(ds, train_cfg.augmentation)

    # --- Phase 1: core losses only ---
    print("\n=== Phase 1: Pre-Train Koopman Model (Core Losses Only) ===")
    model, _phase1_stats = train(model, aug_trajectories, _phase1_cfg(train_cfg))
    save_checkpoint(
        model,
        train_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        path=out_dir / "koop_a_checkpoint.pt",
    )
    print(f"Phase 1 checkpoint saved to {out_dir / 'koop_a_checkpoint.pt'}")

    # --- Phase 2: reinit B, full loss suite ---
    print("\n=== Phase 2: Train B Matrix (All Losses) ===")
    if train_cfg.init_b_in_a:
        model.initialize_B_in_eigenbasis()
        print("Initialized B matrix in A eigenbasis")
    else:
        nn.init.kaiming_uniform_(model.B.weight)
        print("Reinitialized B matrix with random weights")
    model, training_stats = train(model, aug_trajectories, train_cfg)
    save_checkpoint(
        model,
        train_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        path=out_dir / "koopman_ckpt.pt",
    )
    print(f"Phase 2 checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")

    save_model_performance(
        out_dir=out_dir,
        model=model,
        train_cfg=train_cfg,
        aug_trajectories=aug_trajectories,
        device=device,
        training_stats=training_stats,
        ds=ds,
    )
    return str(out_dir)
