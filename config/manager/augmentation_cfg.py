"""Shared data-augmentation config.

Augmentation projects the raw ``(states, actions, base_actions, rewards)`` in
a stored dataset into the ``(koopman_state, koopman_action)`` shape a model
consumes. The exact projection is consumer-dependent (e.g. two-phase Koopman
prepends the base action; joint uses raw obs), so each consumer's top-level
config carries its own :class:`AugmentationCfg` instance.

The augmentation functions in :mod:`data.augmentation` read this config; the
config itself is just data.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(kw_only=True)
class AugmentationCfg:
    """How to project a raw dataset into ``(koopman_state, koopman_action)``."""

    prepend_base_action: bool
    """Prepend the base-policy action to each observation
    (``koopman_state[t] = [obs[t]; base_action[t]]``). Required."""

    use_action_delta: bool
    """Koopman control input is ``action - base_action`` (the realized
    perturbation) when ``True``; otherwise it's the full applied action.
    Required. Two-phase typically pairs ``prepend_base_action=True`` with
    ``use_action_delta=True``; joint pairs both ``False``."""

    obs_scale_source: str = "env"
    """How to scale observations. One of:
    ``"env"``  — use ``max(|obs_space_low|, |obs_space_high|)`` per dim.
    ``"observed"`` — use ``max(|obs_min|, |obs_max|)`` from the dataset.
    ``"none"`` — no normalization."""

    act_scale_source: str = "env"
    """Same as ``obs_scale_source`` but for actions."""
