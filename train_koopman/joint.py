"""Joint training paradigm: encoder + A + B trained together on base trajectories.

Base policy actions are the control input to B (no state augmentation).
Used by ``TrainKoopmanCfg.approach = "joint"``.
"""

from __future__ import annotations

from config.manager import TrainKoopmanCfg
from data.dataloader import load_dataset
from train_koopman.checkpointing import (
    build_koopman_model,
    make_device,
    save_checkpoint,
    weights_dir,
)
from train_koopman.training_loop import train
from train_koopman.two_phase import cfg_to_flat_dict


def run(cfg: TrainKoopmanCfg) -> str:
    flat = cfg_to_flat_dict(cfg)
    out_dir = weights_dir(cfg.experiment_name)

    device = make_device()
    ds = load_dataset(cfg.dataset_name)

    augment = False
    model, _ = build_koopman_model(flat, augment=augment, device=device)

    print(f"Observation scale: {ds.obs_scale}")
    print(f"Action scale: {ds.act_scale}")

    norm_trajectories = []
    for states, actions in ds.base_trajectories:
        norm_states = states / ds.obs_scale
        norm_actions = actions / ds.act_scale
        norm_trajectories.append((norm_states, norm_actions))

    print("\n=== Joint Training: A + B + Encoder Together ===")
    model = train(model, norm_trajectories, flat)

    save_checkpoint(model, flat, out_dir / "koopman_ckpt.pt")
    print(f"Checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")
    return str(out_dir)
