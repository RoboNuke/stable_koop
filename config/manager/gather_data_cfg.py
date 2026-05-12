"""Data-collection config: env, policy, perturbation, dataset I/O.

Field defaults mirror ``config/pendulum.yaml`` exactly. Enum-like choices
(``env_name``, ``obs_type``, ``base_policy_name``, ``analytical_B_policy``) are
required (no default) so a YAML must specify them explicitly — the framework
makes no environment assumptions.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class BasePolicyCfg:
    """Hand-designed base policy used during data collection (pendulum)."""

    name: str
    """Policy name registered in ``policy/__init__.py`` (e.g. ``none``,
    ``energy``, ``bang_energy``, ``pd``). Required."""

    kp: float = 10.0
    """PD proportional gain (pendulum)."""
    kd: float = 2.0
    """PD derivative gain (pendulum)."""
    ke: float = 1.0
    """Energy-shaping gain (pendulum)."""
    gamma: float = 0.9
    """Energy-shaping damping multiplier (pendulum)."""
    switch_angle: float = 30.0
    """Energy-shaping → PD switchover angle in degrees (pendulum)."""


@dataclasses.dataclass(kw_only=True)
class PerturbationCfg:
    """Action perturbation used to excite the system for B-matrix data."""

    analytical_B_policy: str
    """Underlying policy on which to layer perturbations (``none``, ``energy``,
    ``bang_energy``). Required."""

    normalize_analytical_B: bool = False
    """Normalize fitted B to unit spectral norm."""
    perturb_scale: float = 1.0
    """Multiplier on max action for sampled perturbations."""
    fix_perturb_range: bool = False
    """If true, restrict perturbations to a fixed range rather than scaling
    with the underlying policy."""
    hold_steps: int = 5
    """Number of env steps to hold each sampled perturbation before resampling."""


@dataclasses.dataclass(kw_only=True)
class GatherDataCfg:
    """Top-level data-gathering config."""

    env_name: str
    """Gymnasium env id (e.g. ``Pendulum-v1``). Required."""
    obs_type: str
    """Observation convention; ``cos_sin`` = [cos_th, sin_th, thdot] (3D),
    ``theta`` = [theta, thdot] (2D). Required."""

    num_trajectories: int = 200
    """Number of trajectories collected per call."""
    max_episode_steps: int = 200
    """Max steps per trajectory."""
    limited_spawn: bool = False
    """If true, restrict reset distribution via :class:`LimitedSpawnWrapper`."""
    spawn_angle_range: float = 180.0
    """Half-range (degrees) of restricted reset distribution when
    ``limited_spawn`` is true."""

    state_dim: int = 3
    """State / observation dimensionality (post-wrappers)."""
    action_dim: int = 1
    """Action dimensionality."""

    base_policy: BasePolicyCfg = dataclasses.field(
        default_factory=lambda: BasePolicyCfg(name="none")
    )
    """Base policy used to roll out the trajectories."""

    perturbation: PerturbationCfg = dataclasses.field(
        default_factory=lambda: PerturbationCfg(analytical_B_policy="none")
    )
    """Perturbation settings for B-fitting data collection."""

    dataset_name: str = "pendulum_default"
    """Filename stem under ``data/datasets/`` (``<dataset_name>.npz``)."""
    seed: int = 42
    """Seed forwarded to env reset and numpy/torch RNGs."""
