"""Data-collection config: env, policy, optional perturbations, dataset I/O.

One gather call produces one dataset. The dataset is *raw*: just states,
actions, rewards, and per-dimension bounds — no observation normalization,
no state augmentation. Consumers (Koopman training, residual training,
controller analysis) apply their own normalization and augmentation, so the
same dataset can be reused across many configurations.

Examples
--------

Base-policy data, no perturbations (for the Koopman model)::

    gather_data_cfg:
      env_name: "Pendulum-v1"
      base_policy: {name: "bang_energy", params: {...}}
      perturbations: {enabled: false}
      dataset_name: "pendulum_bang_clean"

Same base policy, with random perturbations (e.g., for residual seeding)::

    gather_data_cfg:
      env_name: "Pendulum-v1"
      base_policy: {name: "bang_energy", params: {...}}
      perturbations: {enabled: true, perturb_scale: 1.0, hold_steps: 5}
      dataset_name: "pendulum_bang_perturbed"
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(kw_only=True)
class BasePolicyCfg:
    """Hand-designed policy used to drive the env during data collection."""

    name: str
    """Policy name registered in ``policy/__init__.py``. Required."""

    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Free-form policy-specific kwargs (e.g. ``kp``, ``kd``, ``ke`` for pendulum)."""


@dataclasses.dataclass(kw_only=True)
class PerturbationCfg:
    """Random action perturbations layered on top of the base policy."""

    enabled: bool = False
    """When false, the base policy drives the env directly (no perturbations)."""

    perturb_scale: float = 1.0
    """Multiplier on env action-space bounds for sampled perturbations."""
    fix_perturb_range: bool = False
    """If true, restrict perturbations to a range that won't saturate the
    actuator given the current base action."""
    hold_steps: int = 5
    """Number of env steps a sampled perturbation is held before resampling."""


@dataclasses.dataclass(kw_only=True)
class GatherDataCfg:
    """Top-level data-gathering config — one call produces one dataset."""

    env_name: str
    """Gymnasium env id. Required."""

    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Env-specific wrapper kwargs forwarded to the registered builder."""

    num_trajectories: int = 200
    max_episode_steps: int = 200

    base_policy: BasePolicyCfg = dataclasses.field(
        default_factory=lambda: BasePolicyCfg(name="none")
    )
    perturbations: PerturbationCfg = dataclasses.field(default_factory=PerturbationCfg)

    dataset_name: str = "default"
    seed: int = 42
