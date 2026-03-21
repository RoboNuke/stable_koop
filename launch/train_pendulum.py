import os
import argparse

import gym
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader

from model.autoencoder import KoopmanAutoencoder


def pd_policy(obs, kp, kd):
    """PD controller targeting the upright position (theta=0)."""
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)
    u = -kp * theta - kd * thdot
    return np.array([np.clip(u, -2.0, 2.0)])


def collect_data(env_name, num_trajectories, max_steps, seed, policy=None):
    """Collect trajectories using the given policy (random if None)."""
    env = gym.make(env_name)
    env.seed(seed)
    np.random.seed(seed)

    trajectories = []
    for i in range(num_trajectories):
        obs = env.reset()
        states = [obs]
        actions = []
        for t in range(max_steps):
            action = policy(obs) if policy else env.action_space.sample()
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
    print(f"Collected {len(trajectories)} trajectories "
          f"({sum(len(a) for _, a in trajectories)} transitions)")
    return trajectories


class TrajectoryDataset(Dataset):
    """Provides sliding windows of (H+1 states, H actions) from trajectories."""

    def __init__(self, trajectories, horizon):
        self.windows = []
        for states, actions in trajectories:
            T = len(actions)
            for start in range(T - horizon + 1):
                self.windows.append((
                    torch.from_numpy(states[start:start + horizon + 1]),
                    torch.from_numpy(actions[start:start + horizon]),
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]


def compute_loss(model, states_seq, actions_seq, recon_weight, pred_weight):
    """
    Args:
        model: KoopmanAutoencoder
        states_seq: (B, H+1, state_dim)
        actions_seq: (B, H, action_dim)
    Returns:
        total_loss, recon_loss, pred_loss
    """
    B, Hp1, S = states_seq.shape
    H = Hp1 - 1

    # Reconstruction loss: encode + decode all states in the window
    all_states = states_seq.reshape(B * Hp1, S)
    all_z = model.encode(all_states)
    all_recon = model.decode(all_z)
    recon_loss = F.mse_loss(all_recon, all_states)

    # Multi-step prediction loss: roll forward in latent space, decode, compare
    z = model.encode(states_seq[:, 0])
    pred_loss = torch.tensor(0.0, device=z.device)
    for t in range(H):
        z = model.predict(z, actions_seq[:, t])
        x_pred = model.decode(z)
        pred_loss = pred_loss + F.mse_loss(x_pred, states_seq[:, t + 1])
    pred_loss = pred_loss / H

    total_loss = recon_weight * recon_loss + pred_weight * pred_loss
    return total_loss, recon_loss, pred_loss


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    print(f"Using device: {device}")

    # 1. Collect data
    if cfg.get("random_policy", False):
        policy = None
        print("Using random policy")
    else:
        kp, kd = cfg["kp"], cfg["kd"]
        policy = lambda obs: pd_policy(obs, kp, kd)
        print(f"Using PD policy (kp={kp}, kd={kd})")

    trajectories = collect_data(
        cfg["env_name"], cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"], policy=policy,
    )

    # 2. Build dataset and dataloader
    dataset = TrajectoryDataset(trajectories, cfg["horizon"])
    print(f"Dataset size: {len(dataset)} windows")
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True, pin_memory=True,
    )

    # 3. Build model
    model = KoopmanAutoencoder(
        state_dim=cfg["state_dim"],
        latent_dim=cfg["latent_dim"],
        action_dim=cfg["action_dim"],
        k_type=cfg["k_type"],
        rho=cfg["rho"],
        encoder_spec_norm=cfg["encoder_spec_norm"],
        encoder_latent=cfg["encoder_latent"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"],
    )

    # 5. Training loop
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_total, epoch_recon, epoch_pred, n_batches = 0.0, 0.0, 0.0, 0

        for states_seq, actions_seq in loader:
            states_seq = states_seq.to(device)
            actions_seq = actions_seq.to(device)

            total_loss, recon_loss, pred_loss = compute_loss(
                model, states_seq, actions_seq,
                cfg["recon_weight"], cfg["pred_weight"],
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if hasattr(model.K_module, "project"):
                model.K_module.project()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_pred += pred_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % cfg["log_interval"] == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Total: {epoch_total / n_batches:.6f} | "
                f"Recon: {epoch_recon / n_batches:.6f} | "
                f"Pred:  {epoch_pred / n_batches:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    # 6. Save final checkpoint
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KoopmanAutoencoder on Pendulum")
    parser.add_argument("--config", type=str, default="config/pendulum.yaml")
    parser.add_argument("--random-policy", action="store_true", default=False,
                        help="Use random actions instead of PD policy")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["random_policy"] = args.random_policy
    train(cfg)
