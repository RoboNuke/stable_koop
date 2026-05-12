"""Single canonical save/load for trained Koopman models + a model builder.

Folded from the legacy ``launch/pipeline_utils.py`` so every consumer (LQR
fitting, residual training, eval) reads/writes the same artifact shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.koopman import KoopmanAutoencoder


def make_device() -> torch.device:
    """CUDA if available, else CPU (with a warning)."""
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


def build_koopman_model(cfg: dict, augment: bool, device: torch.device):
    """Build a ``KoopmanAutoencoder`` from a flat ``cfg`` dict.

    ``augment`` is the dataset-augmentation flag (whether the base action is
    appended to the state). Returns ``(model, koopman_state_dim)``.
    """
    koopman_state_dim = cfg["state_dim"] + cfg["action_dim"] if augment else cfg["state_dim"]
    encoder_type = cfg["encoder_type"]
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
        obs_type=cfg.get("obs_type", "theta"),
    ).to(device)
    print(
        f"Koopman model: state_dim={koopman_state_dim}, "
        f"action_dim={cfg['action_dim']}, latent_dim={cfg['latent_dim']}, "
        f"prepend_state={cfg.get('prepend_state', False)}, "
        f"prepend_control={cfg.get('prepend_control', False)}"
    )
    return model, koopman_state_dim


def save_checkpoint(model, cfg: dict, path: str | Path) -> None:
    """Save model checkpoint to ``path``; strips ``torch.compile``'s ``_orig_mod.`` prefix."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, out_path)


def load_checkpoint(model, path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load checkpoint into ``model``; returns the full dict (with ``"config"``)."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {path}")
    return checkpoint


def weights_dir(experiment_name: str) -> Path:
    """Standard output subdir for a Koopman training run."""
    out = Path("train_koopman") / "weights" / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_koopman_experiment(experiment_name: str, device: torch.device):
    """Canonical one-call loader for a trained Koopman experiment.

    Reads ``train_koopman/weights/<experiment_name>/koopman_ckpt.pt``,
    builds a :class:`KoopmanAutoencoder` with the matching augmentation
    (``augment = cfg["prepend_base_action"]`` so joint-trained models
    don't get an extra augmented dim), loads the state dict (stripping
    ``_orig_mod.`` from torch-compiled keys), and returns
    ``(model, koop_cfg_flat)``.
    """
    ckpt_path = Path("train_koopman") / "weights" / experiment_name / "koopman_ckpt.pt"
    raw = torch.load(ckpt_path, map_location=device)
    koop_cfg = raw["config"]
    augment = bool(koop_cfg.get("prepend_base_action", True))
    model, _ = build_koopman_model(koop_cfg, augment=augment, device=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded Koopman weights from {ckpt_path} (augment={augment})")
    return model, koop_cfg
