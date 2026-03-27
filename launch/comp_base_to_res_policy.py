"""Generate side-by-side comparison videos: base policy vs combined (base + residual) policy.

Both policies start from the same initial conditions. The video shows:
- Left: base policy rollout
- Right: combined policy rollout
- Green border when a side succeeds (frozen image, stats keep updating)
- Text overlays: energy, reward, reconstruction error (combined side only)

Usage:
    python -m launch.comp_base_to_res_policy \
        --config path/to/config.yaml \
        --koopman-ckpt path/to/koopman_ckpt.pt \
        --residual-weights path/to/best.pt \
        --stability-yaml path/to/eigen_variables.yaml \
        --output-dir videos/ \
        --num-videos 5
"""
import argparse
import os
import sys

sys.path.insert(0, ".")

import cv2
import gymnasium as gym
import numpy as np
import torch
import yaml

from controllers.lqr import LQR
from launch.eval_policy import check_success, make_policy
from model.autoencoder import KoopmanAutoencoder
from model.residual import StochasticActor
from wrappers.residual import ResidualPolicyEnv


def _step_energy(obs):
    """Instantaneous energy: 0.5 * thdot^2 + 10 * (1 - cos_th)."""
    cos_th, sin_th, thdot = obs[0], obs[1], obs[2]
    return abs(0.5 * thdot ** 2 + 10.0 * (1.0 - cos_th))


def _put_text(frame, text, pos, color, scale=0.5, thickness=1):
    """Draw text with a thin white outline for readability."""
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def _draw_border(frame, color, width=4):
    """Draw a colored border around a frame (in-place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, width)


def make_composite_policy_with_u_res(base_policy, residual_model, lqr_F_np,
                                     z_ref_limit, device, action_bounds):
    """Build composite policy that also returns u_res for reconstruction error."""
    act_low, act_high = action_bounds

    def policy(obs):
        base_action = base_policy(obs)
        obs_aug = np.concatenate([obs, base_action]).astype(np.float32)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_aug).unsqueeze(0).to(device)
            raw_action = residual_model.act({"states": obs_t})[0]
            raw_action_np = raw_action.cpu().numpy().flatten()
        z_ref = z_ref_limit * raw_action_np
        u_res = lqr_F_np @ z_ref
        total_action = np.clip(base_action + u_res, act_low, act_high)
        return total_action, u_res

    return policy


def generate_video(base_policy, combined_policy, koopman_model, cfg,
                   obs_scale, max_runtime_error, seed, output_path, device):
    """Generate a single comparison video."""
    env_name = cfg["env_name"]
    max_steps = cfg["eval_max_steps"]

    # Two envs with same seed for identical initial conditions
    env_base = gym.make(env_name, render_mode="rgb_array")
    env_comb = gym.make(env_name, render_mode="rgb_array")

    obs_base, _ = env_base.reset(seed=seed)
    obs_comb, _ = env_comb.reset(seed=seed)

    # Render initial frames
    frame_base = env_base.render()
    frame_comb = env_comb.render()

    h, w = frame_base.shape[:2]
    fps = int(1.0 / env_base.unwrapped.dt)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w * 2, h))

    # Tracking state
    states_base = [obs_base]
    states_comb = [obs_comb]
    energy_base = 0.0
    energy_comb = 0.0
    reward_base = 0.0
    reward_comb = 0.0
    success_base = False
    success_comb = False
    frozen_frame_base = None
    frozen_frame_comb = None

    obs_scale_np = np.array(obs_scale, dtype=np.float32)

    # Colors (BGR for OpenCV)
    BLACK = (0, 0, 0)
    GREEN = (0, 180, 0)
    RED = (0, 0, 200)

    for step in range(max_steps):
        # --- Base policy side ---
        action_base = base_policy(obs_base)
        obs_base, rew_base, term_b, trunc_b, _ = env_base.step(action_base)
        states_base.append(obs_base)
        energy_base += _step_energy(obs_base)
        reward_base += rew_base

        if not success_base:
            frame_base = env_base.render()
            if check_success(states_base, cfg):
                success_base = True
                frozen_frame_base = frame_base.copy()

        # --- Combined policy side ---
        total_action, u_res = combined_policy(obs_comb)
        obs_comb, rew_comb, term_c, trunc_c, _ = env_comb.step(total_action)
        states_comb.append(obs_comb)
        energy_comb += _step_energy(obs_comb)
        reward_comb += rew_comb

        # Reconstruction error: encode prev and current obs, predict, compare
        with torch.no_grad():
            prev_obs_scaled = torch.FloatTensor(
                np.array(states_comb[-2], dtype=np.float32) / obs_scale_np
            ).unsqueeze(0).to(device)
            curr_obs_scaled = torch.FloatTensor(
                np.array(obs_comb, dtype=np.float32) / obs_scale_np
            ).unsqueeze(0).to(device)
            u_res_t = torch.FloatTensor(
                np.array(u_res, dtype=np.float32)
            ).unsqueeze(0).to(device)

            z_t = koopman_model.encode(prev_obs_scaled)
            z_next = koopman_model.encode(curr_obs_scaled)
            recon_error = koopman_model.prediction_error(z_t, u_res_t, z_next)

        if not success_comb:
            frame_comb = env_comb.render()
            if check_success(states_comb, cfg):
                success_comb = True
                frozen_frame_comb = frame_comb.copy()

        # --- Compose frame ---
        left = frozen_frame_base.copy() if success_base else frame_base.copy()
        right = frozen_frame_comb.copy() if success_comb else frame_comb.copy()

        # Green border on success
        if success_base:
            _draw_border(left, GREEN)
        if success_comb:
            _draw_border(right, GREEN)

        # Avg energy and reward so far
        n = step + 1
        avg_energy_base = energy_base / n
        avg_energy_comb = energy_comb / n

        # --- Text overlays: base side (black) ---
        _put_text(left, f"Energy: {avg_energy_base:.2f}", (10, 25), BLACK)
        _put_text(left, f"Reward: {reward_base:.1f}", (10, 50), BLACK)

        # --- Text overlays: combined side (colored vs base) ---
        energy_color = GREEN if avg_energy_comb <= avg_energy_base else RED
        reward_color = GREEN if reward_comb >= reward_base else RED

        _put_text(right, f"Energy: {avg_energy_comb:.2f}", (10, 25), energy_color)
        _put_text(right, f"Reward: {reward_comb:.1f}", (10, 50), reward_color)

        # Reconstruction error with percentage of max
        if max_runtime_error > 0:
            pct = (recon_error / max_runtime_error) * 100
        else:
            pct = 999.0
        recon_color = GREEN if recon_error < max_runtime_error else RED
        _put_text(right, f"Recon: {recon_error:.3f} ({pct:.0f}%)", (10, 75), recon_color)

        # Combine side-by-side (OpenCV uses BGR)
        combined = np.concatenate([left, right], axis=1)
        # gymnasium renders RGB, OpenCV expects BGR
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        writer.write(combined_bgr)

        # Stop if both done (not just success — env termination)
        if (term_b or trunc_b) and (term_c or trunc_c):
            break

    writer.release()
    env_base.close()
    env_comb.close()

    result_str = (f"base={'SUCCESS' if success_base else 'FAIL'}, "
                  f"combined={'SUCCESS' if success_comb else 'FAIL'}")
    print(f"  Video saved: {output_path} ({step + 1} steps, {result_str})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side base vs combined policy comparison videos")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--koopman-ckpt", type=str, required=True,
                        help="Path to Koopman model checkpoint (.pt)")
    parser.add_argument("--residual-weights", type=str, required=True,
                        help="Path to residual policy weights (.pt)")
    parser.add_argument("--stability-yaml", type=str, required=True,
                        help="Path to stability variables YAML (eigen or lyapunov)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save videos")
    parser.add_argument("--num-videos", type=int, default=5,
                        help="Number of videos to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Starting seed (incremented per video)")
    args = parser.parse_args()

    # 1. Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Koopman model
    koopman_model = KoopmanAutoencoder(
        state_dim=cfg["state_dim"],
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        encoder_type=cfg.get("encoder_type", "linear"),
        rho=cfg["rho"],
        encoder_spec_norm=cfg.get("encoder_spec_norm", False),
        encoder_latent=cfg.get("encoder_latent", 64),
    ).to(device)
    ckpt = torch.load(args.koopman_ckpt, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    koopman_model.load_state_dict(state_dict)
    koopman_model.eval()
    print(f"Loaded Koopman model from {args.koopman_ckpt}")

    # 3. Build LQR
    A = koopman_model.A.detach().cpu()
    B_mat = koopman_model.B_matrix.detach().cpu()
    latent_dim = cfg["latent_dim"]
    action_dim = cfg["action_dim"]
    q_scale = cfg.get("q_scale", 1.0)
    Q = torch.eye(latent_dim) * q_scale
    R = torch.eye(action_dim) * cfg.get("r_scale", 1.0)
    lqr = LQR(A, B_mat, Q, R, q_scale=q_scale,
              controllable_subspace=cfg.get("controllable_subspace", False),
              ctrl_threshold=cfg.get("ctrl_threshold", None))
    lqr_F_np = lqr.F.numpy().astype(np.float32)
    print(f"LQR: F shape={lqr.F.shape}, gain_norm={lqr.gain_norm:.4f}")

    # 4. Load stability variables
    with open(args.stability_yaml) as f:
        stab_vars = yaml.safe_load(f)
    max_runtime_error = stab_vars.get("max_runtime_error_latent",
                                      stab_vars.get("delta_max_lyapunov", 1.0))
    z_ref_limit = stab_vars.get("eta", 1.0)
    print(f"max_runtime_error: {max_runtime_error:.6f}")
    print(f"z_ref_limit (eta): {z_ref_limit:.6f}")

    # 5. Load residual policy
    # Build dummy env to get obs/act spaces for the model
    dummy_env = gym.make(cfg["env_name"])
    base_policy = make_policy(cfg)
    dummy_wrapped = ResidualPolicyEnv(dummy_env, base_policy, lqr, latent_dim, z_ref_limit)
    obs_space = dummy_wrapped.observation_space
    act_space = dummy_wrapped.action_space
    dummy_wrapped.close()

    residual_model = StochasticActor(
        obs_space, act_space, device,
        hidden_size=cfg.get("residual_actor_hidden_size", 64),
        hidden_layers=cfg.get("residual_actor_hidden_layers", 2),
    )
    residual_model.load_state_dict(torch.load(args.residual_weights, map_location=device))
    residual_model.to(device)
    residual_model.eval()
    print(f"Loaded residual policy from {args.residual_weights}")

    # 6. Build policies
    action_bounds = (dummy_env.action_space.low, dummy_env.action_space.high)
    combined_policy = make_composite_policy_with_u_res(
        base_policy, residual_model, lqr_F_np, z_ref_limit, device, action_bounds)

    # 7. Obs scale from config
    obs_scale = cfg.get("obs_scale", [1.0] * cfg["state_dim"])
    print(f"Obs scale: {obs_scale}")

    # 8. Generate videos
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nGenerating {args.num_videos} comparison videos...")

    for i in range(args.num_videos):
        seed = args.seed + i
        output_path = os.path.join(args.output_dir, f"comparison_{seed}.mp4")
        print(f"\nVideo {i + 1}/{args.num_videos} (seed={seed}):")
        generate_video(
            base_policy, combined_policy, koopman_model, cfg,
            obs_scale, max_runtime_error, seed, output_path, device,
        )

    print(f"\nDone. {args.num_videos} videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
