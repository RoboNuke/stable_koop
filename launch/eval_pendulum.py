import argparse
import os

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


_obs_type = "cos_sin"  # module-level default, set by set_obs_type()

def set_obs_type(obs_type):
    """Set the observation type for angle extraction."""
    global _obs_type
    _obs_type = obs_type

def obs_to_angle(obs):
    """Convert pendulum observation to angle θ. Uses module-level _obs_type."""
    if _obs_type == "theta":
        return obs[..., 0]  # theta is already the first element
    else:
        return np.arctan2(obs[..., 1], obs[..., 0])


def evaluate_model(model, trajectories, train_horizon, eval_horizon=25, title=None,
                   obs_scale=None, obs_type=None):
    if obs_type is not None:
        set_obs_type(obs_type)
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
    all_latent_errs = []
    all_state_errs = []

    # Precompute unscaling for angle extraction
    _obs_scale = np.array(obs_scale, dtype=np.float32) if obs_scale is not None else None

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

                # Unscale before extracting angles
                true_state = states[t + 1] * _obs_scale if _obs_scale is not None else states[t + 1]
                pred_state = x_pred * _obs_scale if _obs_scale is not None else x_pred
                true_angle = obs_to_angle(true_state)
                pred_angle = obs_to_angle(pred_state)

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
            all_latent_errs.append(latent_errs.cpu())

            x_pred_all = model.decode(z_pred_all)  # (T, state_dim)
            state_errs = torch.linalg.norm(x_pred_all - states_t[1:T_act + 1], dim=-1)  # (T,)
            max_pred_error_state = max(max_pred_error_state, state_errs.max().item())
            all_state_errs.append(state_errs.cpu())

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

    print(f"One-step prediction error (latent): max={max_pred_error_latent:.6f}  "
          f"mean={mean_pred_error_latent:.6f}  std={std_pred_error_latent:.6f}")
    print(f"One-step prediction error (state):  max={max_pred_error_state:.6f}  "
          f"mean={mean_pred_error_state:.6f}  std={std_pred_error_state:.6f}")

    error_stats = {
        "max_pred_error_latent": max_pred_error_latent,
        "mean_pred_error_latent": mean_pred_error_latent,
        "std_pred_error_latent": std_pred_error_latent,
        "max_pred_error_state": max_pred_error_state,
        "mean_pred_error_state": mean_pred_error_state,
        "std_pred_error_state": std_pred_error_state,
    }
    return fig, error_stats, heatmap_data


def _extract_theta_thdot(raw_trajectories):
    """Extract theta and thdot arrays from raw perturbed trajectories.

    Each raw trajectory is (states, base_actions, perturbations).
    Returns list of (thetas, thdots) arrays, one per trajectory, with length T-1
    (matching one-step prediction pairs).
    """
    all_thetas = []
    all_thdots = []
    for traj in raw_trajectories:
        states_raw = traj[0]  # (T+1, obs_dim)
        T = len(states_raw) - 2  # match z_t/z_next pairs (exclude last state)
        thetas = np.zeros(T)
        thdots = np.zeros(T)
        for t in range(T):
            obs = states_raw[t]
            if len(obs) == 3:  # cos_sin
                thetas[t] = np.arctan2(obs[1], obs[0])
                thdots[t] = obs[2]
            else:  # theta
                thetas[t] = obs[0]
                thdots[t] = obs[1]
        all_thetas.append(thetas)
        all_thdots.append(thdots)
    return all_thetas, all_thdots


def _make_theta_thdot_heatmap(thetas, thdots, errors, title, filepath):
    """Create and save a theta vs theta_dot heatmap with error as color."""
    angle_bins = np.linspace(-np.pi, np.pi, 37)
    angle_centers = 0.5 * (angle_bins[:-1] + angle_bins[1:])
    thdot_bins = np.linspace(-8, 8, 33)
    thdot_centers = 0.5 * (thdot_bins[:-1] + thdot_bins[1:])

    heatmap = np.full((len(angle_centers), len(thdot_centers)), np.nan)
    a_idx = np.digitize(thetas, angle_bins) - 1
    t_idx = np.digitize(thdots, thdot_bins) - 1

    for ai in range(len(angle_centers)):
        for ti in range(len(thdot_centers)):
            mask = (a_idx == ai) & (t_idx == ti)
            if mask.sum() > 0:
                heatmap[ai, ti] = np.mean(errors[mask])

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color="lightgrey")

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(thdot_centers, np.degrees(angle_centers), heatmap,
                        cmap=cmap, shading="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean Error")

    no_data = np.argwhere(np.isnan(heatmap))
    for ai, ti in no_data:
        ax.text(thdot_centers[ti], np.degrees(angle_centers[ai]), "x",
                ha="center", va="center", color="white", fontsize=5, alpha=0.5)

    ax.set_xlabel("θ̇ (rad/s)")
    ax.set_ylabel("θ (degrees)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to {filepath}")


def make_recon_and_lc_heatmaps(model, aug_trajectories, raw_trajectories,
                                run_dir, prefix=""):
    """Generate theta vs theta_dot heatmaps for reconstruction and LC errors.

    Creates two heatmaps:
      1. Reconstruction error: ||decode(encode(x)) - x|| at each (theta, thdot)
      2. Latent consistency error: ||z_{t+1} - predict(z_t, u_t)|| at each (theta, thdot)

    Args:
        model: KoopmanAutoencoder (on device)
        aug_trajectories: list of (koopman_states, actions) — normalized
        raw_trajectories: list of (states, base_actions, perturbations) — raw
        run_dir: output directory
        prefix: filename prefix (e.g. "phase1_" or "phase2_")
    """
    device = next(model.parameters()).device
    model.eval()

    all_thetas_list, all_thdots_list = _extract_theta_thdot(raw_trajectories)

    all_recon_errs = []
    all_lc_errs = []
    all_thetas = []
    all_thdots = []

    with torch.no_grad():
        for i, (states_norm, actions_norm) in enumerate(aug_trajectories):
            states_t = torch.tensor(states_norm, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions_norm, dtype=torch.float32, device=device)

            # Reconstruction error: ||decode(encode(x)) - x||
            z_all = model.encode(states_t)
            x_recon = model.decode(z_all)
            recon_errs = torch.linalg.norm(x_recon - states_t, dim=-1).cpu().numpy()

            # LC error: ||z_{t+1} - predict(z_t, u_t)||
            z_t = z_all[:-1]
            z_next = z_all[1:]
            u_t = actions_t[:len(z_t)]
            z_pred = model.predict(z_t, u_t)
            lc_errs = torch.linalg.norm(z_next - z_pred, dim=-1).cpu().numpy()

            # Trim to match (use T-1 pairs like LC)
            thetas = all_thetas_list[i]
            thdots = all_thdots_list[i]
            n = min(len(thetas), len(lc_errs), len(recon_errs) - 1)
            all_recon_errs.append(recon_errs[:n])
            all_lc_errs.append(lc_errs[:n])
            all_thetas.append(thetas[:n])
            all_thdots.append(thdots[:n])

    all_recon_errs = np.concatenate(all_recon_errs)
    all_lc_errs = np.concatenate(all_lc_errs)
    all_thetas = np.concatenate(all_thetas)
    all_thdots = np.concatenate(all_thdots)

    _make_theta_thdot_heatmap(
        all_thetas, all_thdots, all_recon_errs,
        f"Reconstruction Error vs θ, θ̇",
        os.path.join(run_dir, f"{prefix}recon_theta_thdot.png"))

    _make_theta_thdot_heatmap(
        all_thetas, all_thdots, all_lc_errs,
        f"Latent Consistency Error (1-step) vs θ, θ̇",
        os.path.join(run_dir, f"{prefix}lc_theta_thdot.png"))


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
        encoder_type=cfg.get("encoder_type", "linear"),
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
    ).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)

    trajectories = collect_eval_trajectories(
        cfg["env_name"], args.num_trajectories, args.horizon, args.seed,
    )

    train_horizon = cfg.get("horizon", args.horizon)
    fig, error_stats, heatmap_data = evaluate_model(model, trajectories, train_horizon,
                                                     eval_horizon=args.horizon,
                                                     obs_scale=cfg.get("obs_scale"),
                                                     obs_type=cfg.get("obs_type", "cos_sin"))
    fig.savefig("eval_prediction_error.png", dpi=150, bbox_inches="tight")
    print(f"Plot saved to eval_prediction_error.png")
    plt.show()
