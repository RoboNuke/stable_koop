import argparse

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from model.autoencoder import KoopmanAutoencoder


def collect_eval_trajectories(env_name, num_trajectories, min_steps, seed):
    """Collect trajectories with at least min_steps steps."""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)

    trajectories = []
    for _ in range(num_trajectories):
        obs = env.reset()
        states = [obs]
        actions = []
        for t in range(min_steps):
            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
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


def evaluate(checkpoint_path, num_trajectories=200, horizon=25, seed=123):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint and config
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]

    # Build model and load weights
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
    model.eval()

    # Collect evaluation trajectories
    trajectories = collect_eval_trajectories(
        cfg["env_name"], num_trajectories, horizon, seed,
    )
    print(f"Evaluating on {len(trajectories)} trajectories, {horizon} steps forward")

    # Collect per-step: true angle and absolute angle error
    true_angles_all = []  # list of (true_angle, step, error)
    errors_all = []

    with torch.no_grad():
        for i, (states, actions) in enumerate(trajectories):
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)

            z = model.encode(states_t[0:1])
            for t in range(horizon):
                z = model.predict(z, actions_t[t:t + 1])
                x_pred = model.decode(z).cpu().numpy()[0]

                true_angle = obs_to_angle(states[t + 1])
                pred_angle = obs_to_angle(x_pred)

                err = pred_angle - true_angle
                err = (err + np.pi) % (2 * np.pi) - np.pi
                true_angles_all.append(true_angle)
                errors_all.append((t + 1, np.abs(err)))

    # Bin into a 2D histogram: x=step, y=true angle, value=mean error
    angle_bins = np.linspace(-np.pi, np.pi, 37)  # 36 bins
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    steps = np.arange(1, horizon + 1)

    heatmap = np.full((len(angle_centers), horizon), np.nan)
    true_angles_all = np.array(true_angles_all)
    errors_arr = np.array(errors_all)

    for t in range(horizon):
        mask = errors_arr[:, 0] == (t + 1)
        angles_t = true_angles_all[mask]
        errs_t = errors_arr[mask, 1]
        bin_idx = np.digitize(angles_t, angle_bins) - 1
        for b in range(len(angle_centers)):
            in_bin = errs_t[bin_idx == b]
            if len(in_bin) > 0:
                heatmap[b, t] = np.mean(in_bin)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        steps, np.degrees(angle_centers), np.degrees(heatmap),
        cmap="inferno", shading="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Angle Error (degrees)")
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("True Pendulum Angle (degrees)")
    ax.set_title(f"Koopman Prediction Error vs Angle & Horizon ({num_trajectories} trajectories)")

    save_path = "eval_prediction_error.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate KoopmanAutoencoder on Pendulum")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pendulum/final.pt")
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.num_trajectories, args.horizon, args.seed)
