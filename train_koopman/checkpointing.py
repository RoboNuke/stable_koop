"""Single canonical save/load for trained Koopman models + a model builder.

Folded from the legacy ``launch/pipeline_utils.py`` so every consumer (LQR
fitting, residual training, eval) reads/writes the same artifact shape.

Saved checkpoint layout::

    {
        "model":      {param_name: tensor, ...},          # state_dict
        "config":     dataclasses.asdict(train_cfg),       # nested TrainKoopmanCfg
        "state_dim":  int,                                 # dataset state dim
        "action_dim": int,                                 # dataset action dim
    }

The dataset-derived dims are stored alongside the cfg (rather than inside
it) because they aren't user-configured — they're discovered at gather
time. Pre-refactor flat-dict checkpoints will NOT load with this format.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

from config.manager import TrainKoopmanCfg
from config.manager.manager import ConfigManager
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


def build_koopman_model(
    train_cfg: TrainKoopmanCfg,
    *,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    obs_type: str = "theta",
):
    """Build a :class:`KoopmanAutoencoder` from a :class:`TrainKoopmanCfg`.

    The "augment" flag (whether the base action is appended to the obs) is
    read from ``train_cfg.augmentation.prepend_base_action``. ``obs_type``
    is a per-env knob (e.g. ``"cos_sin"`` for the pendulum) consumed by the
    fixed trig lift; it defaults to ``"theta"`` to match the legacy
    fallback in ``cfg.get("obs_type", "theta")``. Returns
    ``(model, koopman_state_dim)``.
    """
    augment = train_cfg.augmentation.prepend_base_action
    koopman_state_dim = state_dim + action_dim if augment else state_dim
    model = KoopmanAutoencoder(
        state_dim=koopman_state_dim,
        latent_dim=train_cfg.latent_dim,
        action_dim=action_dim,
        k_type=train_cfg.k_type,
        encoder_type=train_cfg.encoder_type,
        rho=train_cfg.rho,
        encoder_spec_norm=train_cfg.encoder_spec_norm,
        encoder_latent=train_cfg.encoder_latent,
        prepend_state=train_cfg.prepend_state,
        prepend_control=train_cfg.prepend_control,
        real_state_dim=state_dim,
        obs_type=obs_type,
    ).to(device)
    print(
        f"Koopman model: state_dim={koopman_state_dim}, "
        f"action_dim={action_dim}, latent_dim={train_cfg.latent_dim}, "
        f"prepend_state={train_cfg.prepend_state}, "
        f"prepend_control={train_cfg.prepend_control}"
    )
    return model, koopman_state_dim


def save_checkpoint(
    model,
    train_cfg: TrainKoopmanCfg,
    *,
    state_dim: int,
    action_dim: int,
    path: str | Path,
) -> None:
    """Persist ``model`` + the dataclass config + dataset dims."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save(
        {
            "model": save_dict,
            "config": dataclasses.asdict(train_cfg),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
        },
        out_path,
    )


def load_checkpoint(model, path: str | Path, device: torch.device) -> dict:
    """Load weights into ``model`` and return ``{config, state_dim, action_dim}``."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded weights from {path}")
    return checkpoint


def experiment_dir(experiment_name: str) -> Path:
    """Output dir for a Koopman experiment: ``results/<experiment_name>/``.

    Holds the model checkpoint (``koopman_ckpt.pt``), a copy of the input
    config (``config.yaml``), the training-summary file
    (``model_performance.yaml``), and any controller-fit subdirs
    (``<controller_type>/<output_name>/``).
    """
    out = Path("results") / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_koopman_experiment(
    experiment_name: str,
    device: torch.device,
    *,
    ckpt_path: str | Path | None = None,
) -> tuple[KoopmanAutoencoder, TrainKoopmanCfg, int, int]:
    """Canonical loader.

    By default reads ``results/<experiment_name>/koopman_ckpt.pt``.
    ``ckpt_path`` overrides the lookup: pass a directory (with
    ``koopman_ckpt.pt`` inside) or a direct ``.pt`` file path.

    Rebuilds the :class:`TrainKoopmanCfg` from the saved nested dict, builds the
    model, loads the state dict (stripping the ``_orig_mod.`` torch-compile
    prefix), and returns ``(model, train_cfg, state_dim, action_dim)``.
    """
    if ckpt_path is None:
        ckpt_path = Path("results") / experiment_name / "koopman_ckpt.pt"
    else:
        ckpt_path = Path(ckpt_path)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "koopman_ckpt.pt"
    raw = torch.load(ckpt_path, map_location=device)
    if not isinstance(raw.get("config"), dict) or "state_dim" not in raw:
        raise RuntimeError(
            f"Checkpoint {ckpt_path} is in the pre-refactor flat-dict format. "
            "Re-train with the current code to get a nested-config checkpoint."
        )
    train_cfg = ConfigManager._build(
        TrainKoopmanCfg, raw["config"], context="checkpoint.config"
    )
    state_dim = int(raw["state_dim"])
    action_dim = int(raw["action_dim"])
    model, _ = build_koopman_model(
        train_cfg, state_dim=state_dim, action_dim=action_dim, device=device
    )
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded Koopman weights from {ckpt_path}")
    return model, train_cfg, state_dim, action_dim
