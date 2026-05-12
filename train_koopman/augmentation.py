"""Per-trainer state/action augmentation + normalization helpers.

The raw dataset (see :mod:`data.gather_data`) carries
``(states, actions, base_actions, rewards)``. Different Koopman training
paradigms project that into different ``(koopman_state, koopman_action)`` pairs:

* **two_phase**: absorb the base policy into the autonomous dynamics by
  prepending ``base_action`` to the state. The Koopman control input is then
  ``action − base_action`` — the realized perturbation. If the dataset has
  no perturbations, that channel is zero everywhere and B never sees any
  excitation (intentional).
* **joint**: no augmentation; the model sees raw observations and ``action``
  as the control input.

Normalization (``obs_scale`` / ``act_scale``) is applied here so the dataset
on disk stays raw and is shareable across configs.
"""

from __future__ import annotations

import numpy as np


def env_obs_scale(obs_space_low: np.ndarray, obs_space_high: np.ndarray) -> np.ndarray:
    """Per-dimension max-absolute scale from env observation bounds."""
    return np.maximum(np.abs(obs_space_high), np.abs(obs_space_low)).astype(np.float32)


def env_act_scale(act_space_low: np.ndarray, act_space_high: np.ndarray) -> np.ndarray:
    """Per-dimension max-absolute scale from env action bounds."""
    return np.maximum(np.abs(act_space_high), np.abs(act_space_low)).astype(np.float32)


def koopman_augment_two_phase(
    trajectories,
    *,
    obs_scale: np.ndarray,
    act_scale: np.ndarray,
) -> list:
    """Two-phase augmentation.

    For each trajectory with ``T`` transitions, emits ``(koopman_states,
    koopman_actions)``:

    * ``koopman_states[t] = [obs_t; base_action_t] / koop_obs_scale``,
      ``t = 0..T-1`` (length ``T``).
    * ``koopman_actions[t] = (action_t − base_action_t) / act_scale``,
      ``t = 0..T-2`` (length ``T−1``; matches legacy bit-equivalent layout
      where the last realized delta is dropped).
    """
    koop_obs_scale = np.concatenate([obs_scale, act_scale])
    out = []
    for states, actions, base_actions, _rewards in trajectories:
        T = len(actions)
        koopman_states = np.concatenate([states[:T], base_actions], axis=-1) / koop_obs_scale
        delta = (actions - base_actions) / act_scale
        out.append((koopman_states.astype(np.float32), delta[:-1].astype(np.float32)))
    return out


def koopman_augment_joint(
    trajectories,
    *,
    obs_scale: np.ndarray,
    act_scale: np.ndarray,
) -> list:
    """Joint augmentation.

    For each trajectory with ``T`` transitions, emits the normalized raw
    states (length ``T+1``) and full applied actions (length ``T``).
    """
    out = []
    for states, actions, _base_actions, _rewards in trajectories:
        koopman_states = (states / obs_scale).astype(np.float32)
        koopman_actions = (actions / act_scale).astype(np.float32)
        out.append((koopman_states, koopman_actions))
    return out
