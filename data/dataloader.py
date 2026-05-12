"""PyTorch dataset + canonical .npz reader for stable_koop datasets.

``load_dataset`` is the single point through which training scripts read
stored trajectories — they never touch raw .npz files directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class LoadedDataset:
    """In-memory dataset returned by :func:`load_dataset`."""

    base_trajectories: list  # list of (states, actions)
    perturbed_trajectories: list  # list of (states, base_actions, perturbations)
    obs_scale: np.ndarray
    act_scale: np.ndarray
    config_yaml: str
    """Snapshot of the GatherDataCfg used to produce this dataset, as YAML text."""


def load_dataset(dataset_name: str, datasets_dir: str | Path = "data/datasets") -> LoadedDataset:
    """Load a dataset written by :func:`data.gather_data.save_dataset`."""
    path = Path(datasets_dir) / f"{dataset_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    arr = np.load(path, allow_pickle=False)

    num_base = int(arr["num_base_trajectories"])
    num_pert = int(arr["num_pert_trajectories"])

    base = []
    for i in range(num_base):
        base.append((arr[f"base_states_{i}"], arr[f"base_actions_{i}"]))
    pert = []
    for i in range(num_pert):
        pert.append(
            (
                arr[f"pert_states_{i}"],
                arr[f"pert_base_actions_{i}"],
                arr[f"pert_perturbations_{i}"],
            )
        )
    return LoadedDataset(
        base_trajectories=base,
        perturbed_trajectories=pert,
        obs_scale=arr["obs_scale"].astype(np.float32),
        act_scale=arr["act_scale"].astype(np.float32),
        config_yaml=str(arr["config_yaml"]),
    )


# Re-export the save function so callers can ``from data.dataloader import save_dataset``
from data.gather_data import save_dataset  # noqa: E402


class TrajectoryDataset(Dataset):
    """Sliding-window dataset of ``(states[H+1], actions[H])`` per item.

    Consumes the (koopman_states, actions) pairs produced by
    :func:`data.gather_data.augment_trajectories` /
    :func:`data.gather_data.augment_perturbed_trajectories`.
    """

    def __init__(self, trajectories, horizon: int):
        self.windows = []
        for states, actions in trajectories:
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
