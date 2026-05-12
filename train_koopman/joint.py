"""Joint training paradigm: encoder + A + B trained together.

Augmentation is whatever ``cfg.augmentation`` specifies; the typical joint
config has ``prepend_base_action=False`` and ``use_action_delta=False``.
"""

from __future__ import annotations

from config.manager import TrainKoopmanCfg
from data.augmentation import augment_trajectories
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
    flat["state_dim"] = ds.state_dim
    flat["action_dim"] = ds.action_dim

    augment = cfg.augmentation.prepend_base_action
    model, _ = build_koopman_model(flat, augment=augment, device=device)
    aug_trajectories = augment_trajectories(ds, cfg.augmentation)

    print("\n=== Joint Training: A + B + Encoder Together ===")
    model = train(model, aug_trajectories, flat)

    save_checkpoint(model, flat, out_dir / "koopman_ckpt.pt")
    print(f"Checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")
    return str(out_dir)
