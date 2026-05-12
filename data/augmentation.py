"""Shared data augmentation + normalization for stable_koop datasets.

Every consumer that turns a raw stored dataset into
``(koopman_state, koopman_action)`` pairs calls into this module. The
exact projection is parameterized by an :class:`AugmentationCfg`:

* ``prepend_base_action`` and ``use_action_delta`` select the trainer's
  view of the data (e.g. two-phase prepends + uses delta; joint does neither).
* ``obs_scale_source`` and ``act_scale_source`` pick the normalization scale
  from the dataset's stored env-space bounds or observed min/max.

This is the single source of truth for that projection — train_koopman,
controller/lqr, and eval all import :func:`augment_trajectories` from here.
"""

from __future__ import annotations

import numpy as np

from config.manager import AugmentationCfg


def _scale_from(source: str, space_low, space_high, observed_min, observed_max):
    """Return the per-dim scale vector for the requested source."""
    if source == "env":
        return np.maximum(np.abs(space_high), np.abs(space_low)).astype(np.float32)
    if source == "observed":
        return np.maximum(np.abs(observed_max), np.abs(observed_min)).astype(np.float32)
    if source == "none":
        return None
    raise ValueError(
        f"Unknown scale source {source!r}; expected 'env', 'observed', or 'none'."
    )


def compute_obs_scale(cfg: AugmentationCfg, ds) -> np.ndarray | None:
    return _scale_from(cfg.obs_scale_source, ds.obs_space_low, ds.obs_space_high, ds.obs_min, ds.obs_max)


def compute_act_scale(cfg: AugmentationCfg, ds) -> np.ndarray | None:
    return _scale_from(cfg.act_scale_source, ds.act_space_low, ds.act_space_high, ds.act_min, ds.act_max)


def augment_trajectories(ds, cfg: AugmentationCfg) -> list:
    """Project ``ds.trajectories`` into ``(koopman_state, koopman_action)`` pairs.

    ``ds`` is a :class:`data.dataloader.LoadedDataset`. The output is the
    list the sliding-window :class:`data.dataloader.TrajectoryDataset` expects:
    each pair has ``len(states) == len(actions) + 1``.

    Behavior matrix:

    +---------------------+--------------------+-----------------------------+
    | prepend_base_action | use_action_delta   | result                      |
    +=====================+====================+=============================+
    | True                | True               | two-phase Koopman (legacy)  |
    +---------------------+--------------------+-----------------------------+
    | False               | False              | joint Koopman (legacy)      |
    +---------------------+--------------------+-----------------------------+
    | True                | False              | [obs; ba] state, full u in  |
    +---------------------+--------------------+-----------------------------+
    | False               | True               | obs state, delta u in       |
    +---------------------+--------------------+-----------------------------+

    The two-phase preserves the legacy quirk of dropping the last realized
    control input (so ``len(actions) == T-1`` for a trajectory with ``T``
    transitions). All other combinations keep every transition.
    """
    obs_scale = compute_obs_scale(cfg, ds)
    act_scale = compute_act_scale(cfg, ds)
    if cfg.prepend_base_action:
        koop_obs_scale = (
            np.concatenate([obs_scale, act_scale])
            if obs_scale is not None and act_scale is not None
            else None
        )
    else:
        koop_obs_scale = obs_scale

    out = []
    for states, actions, base_actions, _rewards in ds.trajectories:
        T = len(actions)

        # State channel
        if cfg.prepend_base_action:
            koopman_states = np.concatenate([states[:T], base_actions], axis=-1)
        else:
            koopman_states = states[: T + 1]

        # Action channel
        if cfg.use_action_delta:
            koopman_actions = actions - base_actions
        else:
            koopman_actions = actions

        # Normalize
        if koop_obs_scale is not None:
            koopman_states = koopman_states / koop_obs_scale
        if act_scale is not None:
            koopman_actions = koopman_actions / act_scale

        # Two-phase legacy quirk: drop the last realized action to match the
        # bit-equivalent layout used historically.
        if cfg.prepend_base_action and cfg.use_action_delta:
            koopman_actions = koopman_actions[:-1]

        out.append((koopman_states.astype(np.float32), koopman_actions.astype(np.float32)))
    return out
