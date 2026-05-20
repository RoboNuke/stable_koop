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


def _fmt_vec(v) -> str:
    """One-line per-dim formatter for small float vectors."""
    if v is None:
        return "n/a"
    return "[" + ", ".join(f"{x:.3g}" for x in np.asarray(v).ravel()) + "]"


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


def augment_trajectories(ds, cfg: AugmentationCfg, *, verbose: bool = True) -> list:
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

    # --- pre-augmentation diagnostics ---
    if verbose:
        print("\n=== Augmentation ===")
        print(f"  prepend_base_action: {cfg.prepend_base_action}")
        print(f"  use_action_delta:    {cfg.use_action_delta}")
        print(f"  obs_scale_source:    {cfg.obs_scale_source}")
        print(f"  act_scale_source:    {cfg.act_scale_source}")
        print("  --- raw env / observed bounds ---")
        print(f"  obs env  low / high: {_fmt_vec(ds.obs_space_low)} / {_fmt_vec(ds.obs_space_high)}")
        print(f"  obs obsv min / max:  {_fmt_vec(ds.obs_min)} / {_fmt_vec(ds.obs_max)}")
        print(f"  act env  low / high: {_fmt_vec(ds.act_space_low)} / {_fmt_vec(ds.act_space_high)}")
        print(f"  act obsv min / max:  {_fmt_vec(ds.act_min)} / {_fmt_vec(ds.act_max)}")
        print("  --- chosen scales ---")
        print(f"  obs scale (per dim): {_fmt_vec(obs_scale)}")
        print(f"  act scale (per dim): {_fmt_vec(act_scale)}")
        if cfg.prepend_base_action:
            print(f"  koop_state scale ([obs; base_action]): {_fmt_vec(koop_obs_scale)}")

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

        # When prepending the base action, the augmented state channel only
        # has T entries (one per available base_action) rather than T+1, so
        # the matching action channel must drop its last entry to preserve
        # the TrajectoryDataset invariant ``len(states) == len(actions) + 1``.
        if cfg.prepend_base_action:
            koopman_actions = koopman_actions[:-1]

        out.append((koopman_states.astype(np.float32), koopman_actions.astype(np.float32)))

    # --- post-augmentation diagnostics: per-dim min/max across all output ---
    if verbose:
        all_states = np.concatenate([s for s, _ in out], axis=0)
        all_actions = np.concatenate([a for _, a in out], axis=0) if any(len(a) for _, a in out) else None
        print("  --- post-augmentation (normalized) ---")
        print(f"  koop_state min / max: {_fmt_vec(all_states.min(axis=0))} / {_fmt_vec(all_states.max(axis=0))}")
        if all_actions is not None and all_actions.size > 0:
            print(f"  koop_action min / max: {_fmt_vec(all_actions.min(axis=0))} / {_fmt_vec(all_actions.max(axis=0))}")
        print(f"  trajectories: {len(out)}  transitions: {sum(len(a) for _, a in out)}")

    return out
