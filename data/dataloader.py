"""Single canonical reader for raw stable_koop datasets.

The dataset on disk is just trajectories + per-dimension bounds — no
augmentation, no normalization. Consumers (train_koopman, controller/lqr,
eval) apply their own augmentation/normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LoadedDataset:
    """In-memory raw dataset returned by :func:`load_dataset`."""

    trajectories: list
    """List of ``(states, actions, base_actions, rewards)`` per trajectory."""

    state_dim: int
    action_dim: int

    obs_space_low: np.ndarray
    obs_space_high: np.ndarray
    act_space_low: np.ndarray
    act_space_high: np.ndarray
    """Env action / observation space bounds (per dimension)."""

    obs_min: np.ndarray
    obs_max: np.ndarray
    act_min: np.ndarray
    act_max: np.ndarray
    """Per-dimension min/max observed in the collected trajectories."""

    perturbations_enabled: bool
    config_yaml: str
    """YAML snapshot of the GatherDataCfg used to produce this dataset."""

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)


def load_dataset(dataset_name: str, datasets_dir: str | Path = "data/datasets") -> LoadedDataset:
    """Load a dataset written by :func:`data.gather_data.save_dataset`.

    Looks for ``<datasets_dir>/<dataset_name>/<dataset_name>.npz`` first
    (the standalone-gather layout). If ``dataset_name`` ends in ``_pert``,
    also checks the shared ``--both`` folder
    ``<datasets_dir>/<base>/<dataset_name>.npz`` where ``base`` is the
    name with the ``_pert`` suffix stripped.
    """
    root = Path(datasets_dir)
    candidates = [root / dataset_name / f"{dataset_name}.npz"]
    if dataset_name.endswith("_pert"):
        base = dataset_name[: -len("_pert")]
        candidates.append(root / base / f"{dataset_name}.npz")
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} not found; tried: "
            + ", ".join(str(p) for p in candidates)
        )
    arr = np.load(path, allow_pickle=False)

    num = int(arr["num_trajectories"])
    trajectories = []
    for i in range(num):
        trajectories.append(
            (
                arr[f"states_{i}"],
                arr[f"actions_{i}"],
                arr[f"base_actions_{i}"],
                arr[f"rewards_{i}"],
            )
        )

    return LoadedDataset(
        trajectories=trajectories,
        state_dim=int(arr["state_dim"]),
        action_dim=int(arr["action_dim"]),
        obs_space_low=arr["obs_space_low"].astype(np.float32),
        obs_space_high=arr["obs_space_high"].astype(np.float32),
        act_space_low=arr["act_space_low"].astype(np.float32),
        act_space_high=arr["act_space_high"].astype(np.float32),
        obs_min=arr["obs_min"].astype(np.float32),
        obs_max=arr["obs_max"].astype(np.float32),
        act_min=arr["act_min"].astype(np.float32),
        act_max=arr["act_max"].astype(np.float32),
        perturbations_enabled=bool(arr["perturbations_enabled"]),
        config_yaml=str(arr["config_yaml"]),
    )


# Re-export the save function so callers can ``from data.dataloader import save_dataset``
from data.gather_data import save_dataset  # noqa: E402


class TrajectoryDataset(Dataset):
    """Sliding-window torch dataset of ``(states[H+1], actions[H])`` per item.

    Consumes a list of ``(koopman_states, koopman_actions)`` pairs produced
    by ``train_koopman/augmentation.py``.
    """

    def __init__(self, augmented_trajectories, horizon: int):
        self.windows = []
        for states, actions in augmented_trajectories:
            T = len(actions)
            for start in range(T - horizon + 1):
                self.windows.append(
                    (
                        torch.from_numpy(states[start : start + horizon + 1]),
                        torch.from_numpy(actions[start : start + horizon]),
                    )
                )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
