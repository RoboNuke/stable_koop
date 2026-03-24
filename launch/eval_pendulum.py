import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model.autoencoder import KoopmanAutoencoder


def collect_eval_trajectories(env_name, num_trajectories, min_steps, seed):
    """Collect trajectories with at least min_steps steps."""
    env = gym.make(env_name)
    np.random.seed(seed)

    trajectories = []
    for _ in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        actions = []
        for t in range(min_steps):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            states.append(obs)
            actions.append(action)
            if done:
                break
        trajectories.append((
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32).reshape(-1, 1),
        ))

    env.close()
    return trajectories


def obs_to_angle(obs):
    """Convert pendulum observation [cos(θ), sin(θ), θ_dot] to angle θ."""
    return np.arctan2(obs[..., 1], obs[..., 0])


def evaluate_model(model, trajectories, train_horizon, eval_horizon=25):
    """Evaluate Koopman model on trajectories.

    Args:
        model: KoopmanAutoencoder (already on device, eval mode set internally)
        trajectories: list of (states, actions) numpy arrays
        train_horizon: training horizon (shown as red dashed line on plot)
        eval_horizon: number of prediction steps for heatmap (default 25)

    Returns:
        (fig, max_pred_error_latent, max_pred_error_state, heatmap_data) where
        heatmap_data is a dict with keys: angle_centers_deg, steps, heatmap_deg
    """
    device = next(model.parameters()).device
    model.eval()

    num_trajectories = len(trajectories)
    print(f"Evaluating on {num_trajectories} trajectories, {eval_horizon} steps forward")

    true_angles_all = []
    errors_all = []
    max_pred_error_latent = 0.0
    max_pred_error_state = 0.0

    with torch.no_grad():
        for states, actions in trajectories:
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)

            # Multi-step prediction for heatmap
            z = model.encode(states_t[0:1])
            T = min(eval_horizon, len(actions))
            for t in range(T):
                z = model.predict(z, actions_t[t:t + 1])
                x_pred = model.decode(z).cpu().numpy()[0]

                true_angle = obs_to_angle(states[t + 1])
                pred_angle = obs_to_angle(x_pred)

                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

            # Batched one-step prediction errors for max error
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])       # (T, latent)
            z_next_all = model.encode(states_t[1:T_act + 1])  # (T, latent)
            z_pred_all = model.predict(z_all, actions_t[:T_act])  # (T, latent)

            latent_errs = torch.linalg.norm(z_next_all - z_pred_all, dim=-1)  # (T,)
            max_pred_error_latent = max(max_pred_error_latent, latent_errs.max().item())

            x_pred_all = model.decode(z_pred_all)  # (T, state_dim)
            state_errs = torch.linalg.norm(x_pred_all - states_t[1:T_act + 1], dim=-1)  # (T,)
            max_pred_error_state = max(max_pred_error_state, state_errs.max().item())

    # Bin into a 2D histogram: x=step, y=true angle, value=mean error
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

    # Plot — use grey for bins with no data
    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_deg = np.degrees(heatmap)  # NaN cells stay NaN → rendered as "bad"
    im = ax.pcolormesh(
        steps, np.degrees(angle_centers), heatmap_deg,
        cmap=cmap, shading="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Angle Error (degrees)")

    # Mark no-data cells with an 'x'
    no_data = np.argwhere(np.isnan(heatmap))
    for b, t in no_data:
        ax.text(steps[t], np.degrees(angle_centers[b]), "x",
                ha="center", va="center", color="white", fontsize=6, alpha=0.7)
    ax.axvline(x=train_horizon, color="red", linestyle="--", linewidth=1.5,
               label=f"Train horizon ({train_horizon})")
    ax.legend(loc="upper left")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("True Pendulum Angle (degrees)")
    ax.set_title(f"Koopman Prediction Error vs Angle & Horizon ({num_trajectories} trajectories)")

    heatmap_data = {
        "angle_centers_deg": np.degrees(angle_centers).tolist(),
        "steps": steps.tolist(),
        "heatmap_deg": np.where(np.isnan(heatmap_deg), None, heatmap_deg).tolist(),
    }

    print(f"Max one-step prediction error (latent): {max_pred_error_latent:.6f}")
    print(f"Max one-step prediction error (state):  {max_pred_error_state:.6f}")
    return fig, max_pred_error_latent, max_pred_error_state, heatmap_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KoopmanAutoencoder on Pendulum")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pendulum/final.pt")
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]

    model = KoopmanAutoencoder(
        state_dim=cfg["state_dim"],
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    trajectories = collect_eval_trajectories(
        cfg["env_name"], args.num_trajectories, args.horizon, args.seed,
    )

    train_horizon = cfg.get("horizon", args.horizon)
    fig, max_err_latent, max_err_state, heatmap_data = evaluate_model(model, trajectories, train_horizon, eval_horizon=args.horizon)
    fig.savefig("eval_prediction_error.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to eval_prediction_error.png")
    plt.show()
