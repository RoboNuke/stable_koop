"""Joint training paradigm: encoder + A + B trained together.

The applied actions drive B directly (no state augmentation). Used by
``TrainKoopmanCfg.approach = "joint"``.
"""

from __future__ import annotations

from config.manager import TrainKoopmanCfg
from data.dataloader import load_dataset
from train_koopman.augmentation import (
    env_act_scale,
    env_obs_scale,
    koopman_augment_joint,
)
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

    model, _ = build_koopman_model(flat, augment=False, device=device)

    obs_scale = env_obs_scale(ds.obs_space_low, ds.obs_space_high)
    act_scale = env_act_scale(ds.act_space_low, ds.act_space_high)
    print(f"Observation scale: {obs_scale}")
    print(f"Action scale: {act_scale}")

    aug_trajectories = koopman_augment_joint(
        ds.trajectories, obs_scale=obs_scale, act_scale=act_scale
    )

    print("\n=== Joint Training: A + B + Encoder Together ===")
    model = train(model, aug_trajectories, flat)

    save_checkpoint(model, flat, out_dir / "koopman_ckpt.pt")
    print(f"Checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")
    return str(out_dir)
