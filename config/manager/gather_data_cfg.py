"""Data-collection config: env, policy, perturbation, dataset I/O.

The framework is environment-agnostic: env-specific knobs go in the
free-form ``env_kwargs`` dict and are forwarded verbatim to the wrapper
builder registered under :data:`data.env_builder.ENV_WRAPPERS`. Similarly,
``BasePolicyCfg.params`` carries whatever per-policy kwargs the chosen
policy expects.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(kw_only=True)
class BasePolicyCfg:
    """Hand-designed policy used during data collection."""

    name: str
    """Policy name registered in ``policy/__init__.py`` (e.g. ``none``,
    ``energy``, ``bang_energy``, ``pd``). Required."""

    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Free-form policy-specific kwargs forwarded to the policy constructor.
    For pendulum: ``kp``, ``kd``, ``ke``, ``gamma``, ``switch_angle``. New
    envs add their own keys; unknown keys are swallowed by the policy
    classes' ``**_unused``."""


@dataclasses.dataclass(kw_only=True)
class PerturbationCfg:
    """Action perturbation used to excite the system for B-matrix data."""

    analytical_B_policy: str
    """Underlying policy on which to layer perturbations. Required."""

    params: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Same shape as ``BasePolicyCfg.params``: forwarded to the policy."""

    normalize_analytical_B: bool = False
    perturb_scale: float = 1.0
    fix_perturb_range: bool = False
    hold_steps: int = 5


@dataclasses.dataclass(kw_only=True)
class GatherDataCfg:
    """Top-level data-gathering config."""

    env_name: str
    """Gymnasium env id (e.g. ``Pendulum-v1``). Required."""

    env_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Env-specific wrapper kwargs forwarded to the builder registered in
    :data:`data.env_builder.ENV_WRAPPERS` for ``env_name``. Empty is fine
    for unwrapped envs."""

    num_trajectories: int = 200
    max_episode_steps: int = 200

    base_policy: BasePolicyCfg = dataclasses.field(
        default_factory=lambda: BasePolicyCfg(name="none")
    )
    perturbation: PerturbationCfg = dataclasses.field(
        default_factory=lambda: PerturbationCfg(analytical_B_policy="none")
    )

    dataset_name: str = "pendulum_default"
    seed: int = 42
