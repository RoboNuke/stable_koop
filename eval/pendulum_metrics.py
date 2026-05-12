"""Pendulum-specific eval logic — success criteria, trajectory metrics, heatmaps.

Imported and registered by :mod:`eval` at import time. The generic framework
in ``eval/policy_rollout.py`` and ``eval/koopman_accuracy.py`` does not know
about pendulum — it dispatches into here through the eval registries.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Observation parsing (pendulum convention).
# ---------------------------------------------------------------------------


def _parse_states(states, obs_type=None):
    """Extract ``(theta, cos_th, thdot)`` from a state array."""
    if obs_type is not None:
        use_theta = obs_type == "theta"
    else:
        use_theta = states.shape[-1] != 3
    if not use_theta:
        cos_th, sin_th, thdot = states[..., 0], states[..., 1], states[..., 2]
        theta = np.arctan2(sin_th, cos_th)
    else:
        theta, thdot = states[..., 0], states[..., 1]
        cos_th = np.cos(theta)
    return theta, cos_th, thdot


def obs_to_angle(obs, obs_type="cos_sin"):
    """Pendulum-specific angle extraction (used by the koopman accuracy heatmap)."""
    if obs_type == "theta":
        return obs[..., 0]
    return np.arctan2(obs[..., 1], obs[..., 0])


# ---------------------------------------------------------------------------
# Per-rollout scorers used by eval/policy_rollout.py via the registry.
# ---------------------------------------------------------------------------


def pendulum_check_success(states, cfg):
    """True iff the final ``success_hold_steps`` are within the success region."""
    hold = cfg["success_hold_steps"]
    if len(states) < hold:
        return False
    tail = np.array(states[-hold:])
    theta, _, thdot = _parse_states(tail)
    angle_ok = np.all(np.abs(theta) < np.radians(cfg["success_angle_deg"]))
    vel_ok = np.all(np.abs(thdot) < cfg["success_max_thdot"])
    return bool(angle_ok and vel_ok)


def pendulum_compute_metrics(states, actions):
    """Per-trajectory mean-absolute metrics (energy, torque, angular velocity)."""
    theta, cos_th, thdot = _parse_states(states)
    energy = np.abs(0.5 * thdot ** 2 + 10.0 * (1.0 - cos_th))
    return {
        "length": len(actions),
        "energy": float(np.mean(energy)),
        "control_torque": float(np.mean(np.abs(actions))),
        "angular_velocity": float(np.mean(np.abs(thdot))),
        "reward": 0.0,
    }


# ---------------------------------------------------------------------------
# Koopman accuracy heatmap (pendulum angle vs prediction step).
# ---------------------------------------------------------------------------


def make_koopman_angle_heatmap(
    model, trajectories, train_horizon, eval_horizon=25, title=None, obs_scale=None,
    obs_type="cos_sin",
):
    """Multi-step prediction error heatmap binned by true pendulum angle."""
    device = next(model.parameters()).device
    model.eval()
    _obs_scale = np.array(obs_scale, dtype=np.float32) if obs_scale is not None else None

    true_angles_all = []
    errors_all = []
    with torch.no_grad():
        for states, actions in trajectories:
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            z = model.encode(states_t[0:1])
            T = min(eval_horizon, len(actions))
            for t in range(T):
                z = model.predict(z, actions_t[t:t + 1])
                x_pred = model.decode(z).cpu().numpy()[0]
                true_state = states[t + 1] * _obs_scale if _obs_scale is not None else states[t + 1]
                pred_state = x_pred * _obs_scale if _obs_scale is not None else x_pred
                true_angle = obs_to_angle(true_state, obs_type)
                pred_angle = obs_to_angle(pred_state, obs_type)
                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    steps = np.arange(1, eval_horizon + 1)
    heatmap = np.full((len(angle_centers), eval_horizon), np.nan)
    true_angles_all = np.array(true_angles_all)
    errors_arr = np.array(errors_all)
    for t in range(eval_horizon):
        mask = errors_arr[:, 0] == (t + 1)
        bin_idx = np.digitize(true_angles_all[mask], angle_bins) - 1
        errs_t = errors_arr[mask, 1]
        for b in range(len(angle_centers)):
            in_bin = errs_t[bin_idx == b]
            if len(in_bin) > 0:
                heatmap[b, t] = np.mean(in_bin)

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_deg = np.degrees(heatmap)
    im = ax.pcolormesh(steps, np.degrees(angle_centers), heatmap_deg, cmap=cmap, shading="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Angle Error (degrees)")
    no_data = np.argwhere(np.isnan(heatmap))
    for b, t in no_data:
        ax.text(steps[t], np.degrees(angle_centers[b]), "x", ha="center", va="center", color="white", fontsize=6, alpha=0.7)
    ax.axvline(x=train_horizon, color="red", linestyle="--", linewidth=1.5, label=f"Train horizon ({train_horizon})")
    ax.legend(loc="upper left")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("True Pendulum Angle (degrees)")
    ax.set_title(title or f"Koopman Prediction Error vs Angle & Horizon ({len(trajectories)} trajectories)")
    heatmap_data = {
        "angle_centers_deg": np.degrees(angle_centers).tolist(),
        "steps": steps.tolist(),
        "heatmap_deg": np.where(np.isnan(heatmap_deg), None, heatmap_deg).tolist(),
    }
    return fig, heatmap_data
