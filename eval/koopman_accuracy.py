"""Koopman model accuracy evaluation: multi-step prediction error vs angle/horizon.

Ported from ``launch/eval_pendulum.py``. ``evaluate_model`` is the headline
entry point — it produces a (figure, stats, heatmap) triple consumed by the
LQR-controller fitting stage and by standalone analysis scripts.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch


_obs_type = "cos_sin"  # module-level default — pendulum-specific


def set_obs_type(obs_type: str) -> None:
    """Set the obs-type used by :func:`obs_to_angle`."""
    global _obs_type
    _obs_type = obs_type


def obs_to_angle(obs):
    """Pendulum-specific angle extraction (uses module-level ``_obs_type``)."""
    if _obs_type == "theta":
        return obs[..., 0]
    return np.arctan2(obs[..., 1], obs[..., 0])


def evaluate_model(model, trajectories, train_horizon, eval_horizon=25, title=None,
                   obs_scale=None, obs_type=None):
    """Multi-step Koopman prediction evaluation on a list of trajectories."""
    if obs_type is not None:
        set_obs_type(obs_type)

    device = next(model.parameters()).device
    model.eval()
    num_trajectories = len(trajectories)
    print(f"Evaluating on {num_trajectories} trajectories, {eval_horizon} steps forward")

    true_angles_all = []
    errors_all = []
    max_pred_error_latent = 0.0
    max_pred_error_state = 0.0
    all_latent_errs = []
    all_state_errs = []
    _obs_scale = np.array(obs_scale, dtype=np.float32) if obs_scale is not None else None

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
                true_angle = obs_to_angle(true_state)
                pred_angle = obs_to_angle(pred_state)
                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_next_all = model.encode(states_t[1:T_act + 1])
            z_pred_all = model.predict(z_all, actions_t[:T_act])
            latent_errs = torch.linalg.norm(z_next_all - z_pred_all, dim=-1)
            max_pred_error_latent = max(max_pred_error_latent, latent_errs.max().item())
            all_latent_errs.append(latent_errs.cpu())
            x_pred_all = model.decode(z_pred_all)
            state_errs = torch.linalg.norm(x_pred_all - states_t[1:T_act + 1], dim=-1)
            max_pred_error_state = max(max_pred_error_state, state_errs.max().item())
            all_state_errs.append(state_errs.cpu())

    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    steps = np.arange(1, eval_horizon + 1)
    heatmap = np.full((len(angle_centers), eval_horizon), np.nan)
    true_angles_all = np.array(true_angles_all)
    errors_arr = np.array(errors_all)
    for t in range(eval_horizon):
        mask = errors_arr[:, 0] == (t + 1)
        angles_t = true_angles_all[mask]
        errs_t = errors_arr[mask, 1]
        bin_idx = np.digitize(angles_t, angle_bins) - 1
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
    if title is None:
        title = f"Koopman Prediction Error vs Angle & Horizon ({num_trajectories} trajectories)"
    ax.set_title(title)

    heatmap_data = {
        "angle_centers_deg": np.degrees(angle_centers).tolist(),
        "steps": steps.tolist(),
        "heatmap_deg": np.where(np.isnan(heatmap_deg), None, heatmap_deg).tolist(),
    }

    all_latent_errs = torch.cat(all_latent_errs)
    all_state_errs = torch.cat(all_state_errs)
    mean_pred_error_latent = all_latent_errs.mean().item()
    std_pred_error_latent = all_latent_errs.std().item()
    mean_pred_error_state = all_state_errs.mean().item()
    std_pred_error_state = all_state_errs.std().item()

    print(
        f"One-step prediction error (latent): max={max_pred_error_latent:.6f}  "
        f"mean={mean_pred_error_latent:.6f}  std={std_pred_error_latent:.6f}"
    )
    print(
        f"One-step prediction error (state):  max={max_pred_error_state:.6f}  "
        f"mean={mean_pred_error_state:.6f}  std={std_pred_error_state:.6f}"
    )

    error_stats = {
        "max_pred_error_latent": max_pred_error_latent,
        "mean_pred_error_latent": mean_pred_error_latent,
        "std_pred_error_latent": std_pred_error_latent,
        "max_pred_error_state": max_pred_error_state,
        "mean_pred_error_state": mean_pred_error_state,
        "std_pred_error_state": std_pred_error_state,
    }
    return fig, error_stats, heatmap_data
