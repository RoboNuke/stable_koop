"""Hand-designed base policies + a name → instance factory.

Every policy implements the same callable interface:

    policy(obs: np.ndarray) -> np.ndarray  # shape (action_dim,)

Consumers (data gathering, residual training, eval) build a policy by name and
then call it interchangeably regardless of which environment it was designed
for. The registry is the single source of truth for policy dispatch.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

# Lazy imports of per-env policies so unused imports never run.
from policy.inverted_pendulum.pd import PDPolicy as InvertedPendulumPDPolicy
from policy.pendulum.energy_based import (
    BangEnergyShapingPolicy,
    EnergyShapingPolicy,
    ZeroPolicy,
)
from policy.pendulum.pd import PDPolicy


class BasePolicy(Protocol):
    """Structural type for any hand-designed policy."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


_REGISTRY: dict[str, Callable[..., BasePolicy]] = {
    "none": ZeroPolicy,
    "pd": PDPolicy,
    "energy": EnergyShapingPolicy,
    "bang_energy": BangEnergyShapingPolicy,
    "PD_policy": InvertedPendulumPDPolicy,
}


def available_policies() -> list[str]:
    return sorted(_REGISTRY)


def make_policy(name: str, **params) -> BasePolicy:
    """Construct a policy by registered name.

    ``params`` are the policy-specific kwargs (e.g. ``kp``, ``kd``, ``ke``).
    Unknown ``name`` raises; unknown params are passed through and surface
    as the policy class's own TypeError.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown policy {name!r}; registered policies: {available_policies()}"
        )
    return _REGISTRY[name](**params)


__all__ = ["BasePolicy", "available_policies", "make_policy"]
