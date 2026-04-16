"""Shared pipeline utilities for Koopman training and evaluation scripts."""
import os
import sys

import numpy as np
import torch
import yaml

from model.autoencoder import KoopmanAutoencoder
from launch.eval_pendulum import evaluate_model


class Tee:
    """Redirect stdout to both a file and the original stream."""
    def __init__(self, filepath, stream):
        self.file = open(filepath, "w")
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def close(self):
        self.file.close()


def make_device():
    """Select CUDA or CPU device with warning."""
    RED = "\033[91m"
    RESET = "\033[0m"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print(f"{RED}WARNING: CUDA not available.{RESET}")
        print(f"{RED}FALLBACK: Using CPU. Training will be significantly slower.{RESET}")
    print(f"Using device: {device}")
    return device


def build_koopman_model(cfg, augment, device):
    """Build a KoopmanAutoencoder from config and move to device.

    Args:
        cfg: config dict
        augment: whether to augment state with base action
        device: torch device

    Returns:
        (model, koopman_state_dim)
    """
    koopman_state_dim = cfg["state_dim"] + cfg["action_dim"] if augment else cfg["state_dim"]
    encoder_type = cfg.get("encoder_type", None)
    RED = "\033[91m"
    RESET = "\033[0m"
    if encoder_type is None:
        encoder_type = "linear"
        print(f"{RED}WARNING: encoder_type not specified in config.{RESET}")
        print(f"{RED}FALLBACK: Using encoder_type='linear'.{RESET}")
    model = KoopmanAutoencoder(
        state_dim=koopman_state_dim,
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        encoder_type=encoder_type,
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
        prepend_state=cfg.get("prepend_state", False),
        prepend_control=cfg.get("prepend_control", False),
        real_state_dim=cfg["state_dim"],
    ).to(device)
    print(f"Koopman model: state_dim={koopman_state_dim}, action_dim={cfg['action_dim']}, "
          f"latent_dim={cfg['latent_dim']}, prepend_state={cfg.get('prepend_state', False)}, "
          f"prepend_control={cfg.get('prepend_control', False)}")
    return model, koopman_state_dim


def save_checkpoint(model, cfg, path):
    """Save model checkpoint with _orig_mod. prefix cleanup."""
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, path)


def load_checkpoint(model, path, device):
    """Load checkpoint with _orig_mod. prefix cleanup."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {path}")
    return checkpoint


def evaluate_and_save(model, aug_trajectories, cfg, run_dir, prefix, title):
    """Evaluate model on trajectories, save heatmap and stats.

    Args:
        model: KoopmanAutoencoder
        aug_trajectories: augmented trajectory data
        cfg: config dict
        run_dir: output directory
        prefix: filename prefix (e.g. "koop_a_" or "koop_b_")
        title: heatmap title

    Returns:
        error_stats dict
    """
    train_horizon = cfg["horizon"]
    fig, error_stats, heatmap_data = evaluate_model(
        model, aug_trajectories, train_horizon,
        eval_horizon=max(25, train_horizon),
        title=title,
        obs_scale=cfg.get("obs_scale"),
        obs_type=cfg.get("obs_type", "cos_sin"))

    plot_path = os.path.join(run_dir, f"{prefix}prediction_error.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap saved to {plot_path}")

    eval_stats = {**error_stats, "heatmap": heatmap_data}
    stats_path = os.path.join(run_dir, f"{prefix}eval_stats.yaml")
    with open(stats_path, "w") as f:
        yaml.dump(eval_stats, f, default_flow_style=False, sort_keys=False)
    print(f"Eval stats saved to {stats_path}")

    return error_stats
