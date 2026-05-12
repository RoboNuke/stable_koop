"""Two-phase Koopman training paradigm.

Phase 1 — pretrain (encoder + A + B) with only the core losses (recon, pred,
latent-consistency) disabled-optional. Phase 2 — reinitialize B, then run the
full loss suite including BiLip / B-eigen / unstable-controllability / etc.

Both phases reuse the canonical :func:`train_koopman.training_loop.train`
function so optimization details stay consistent.
"""

from __future__ import annotations

import dataclasses
import torch.nn as nn

from config.manager import TrainKoopmanCfg
from data.dataloader import load_dataset
from data.gather_data import augment_perturbed_trajectories
from train_koopman.checkpointing import (
    build_koopman_model,
    make_device,
    save_checkpoint,
    weights_dir,
)
from train_koopman.training_loop import train


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


def cfg_to_flat_dict(cfg: TrainKoopmanCfg) -> dict:
    """Flatten a :class:`TrainKoopmanCfg` into the legacy pendulum.yaml dict shape.

    The loss-builder + training loop were written against a flat dict; flattening
    here preserves bit-exact behavior with the legacy code paths.
    """
    flat = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if dataclasses.is_dataclass(v):
            flat.update(dataclasses.asdict(v))
        else:
            flat[f.name] = v
    # Bring B-fitting subfields into top-level names matching the legacy YAML.
    flat["train_for_B"] = cfg.b_fitting.method != "least_squares"
    flat["analytical_B_ctrl_lambda"] = cfg.b_fitting.ctrl_lambda
    flat["analytical_B_train_steps"] = cfg.b_fitting.train_steps
    flat["analytical_B_lr"] = cfg.b_fitting.lr
    return flat


def run(cfg: TrainKoopmanCfg) -> str:
    """End-to-end two-phase training; returns the output weights directory."""
    flat = cfg_to_flat_dict(cfg)
    out_dir = weights_dir(cfg.experiment_name)

    device = make_device()
    ds = load_dataset(cfg.dataset_name)
    flat["state_dim"] = ds.state_dim
    flat["action_dim"] = ds.action_dim

    # Currently the two-phase paradigm always augments the state with the
    # base action. This matches the legacy run.py default behavior.
    augment = True
    model, _ = build_koopman_model(flat, augment=augment, device=device)

    # When augmenting, the Koopman state is [obs; base_action]; its scale
    # vector is the concatenation of obs_scale and act_scale.
    import numpy as np
    koopman_obs_scale = (
        np.concatenate([ds.obs_scale, ds.act_scale]) if augment else ds.obs_scale
    )
    aug_trajectories = augment_perturbed_trajectories(
        ds.perturbed_trajectories,
        augment=augment,
        obs_scale=koopman_obs_scale,
        act_scale=ds.act_scale,
    )

    # --- Phase 1: core losses only ---
    print("\n=== Phase 1: Pre-Train Koopman Model (Core Losses Only) ===")
    phase1_cfg = dict(flat)
    for key in _PHASE1_DISABLED_LOSSES:
        phase1_cfg[key] = False
    model = train(model, aug_trajectories, phase1_cfg)
    save_checkpoint(model, flat, out_dir / "koop_a_checkpoint.pt")
    print(f"Phase 1 checkpoint saved to {out_dir / 'koop_a_checkpoint.pt'}")

    # --- Phase 2: reinit B, full loss suite ---
    print("\n=== Phase 2: Train B Matrix (All Losses) ===")
    if cfg.init_b_in_a:
        model.initialize_B_in_eigenbasis()
        print("Initialized B matrix in A eigenbasis")
    else:
        nn.init.kaiming_uniform_(model.B.weight)
        print("Reinitialized B matrix with random weights")
    model = train(model, aug_trajectories, flat)
    save_checkpoint(model, flat, out_dir / "koopman_ckpt.pt")
    print(f"Phase 2 checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")

    return str(out_dir)
