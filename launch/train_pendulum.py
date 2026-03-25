import os
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap
import yaml
from torch.utils.data import Dataset, DataLoader

from model.autoencoder import KoopmanAutoencoder

#def pd_policy(obs, kp=6.0, kd=1.0):
#    cos_th, sin_th, thdot = obs
#    theta = np.arctan2(sin_th, cos_th)      # in [-π, π], 0=down, ±π=up
#    error = np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))  # deviation from upright
#    u = -kp * error - kd * thdot
#    return np.array([np.clip(u, -2.0, 2.0)])

def pd_policy(obs, kp, kd):
    """PD controller targeting the upright position (theta=0)."""
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)
    u = -kp * theta - kd * thdot
    return np.array([np.clip(u, -2.0, 2.0)])


def collect_data(env, num_trajectories, max_steps, seed, policy=None):
    """Collect trajectories using the given policy (random if None)."""
    np.random.seed(seed)

    trajectories = []
    for i in range(num_trajectories):
        obs, _ = env.reset()
        states = [obs]
        actions = []
        for t in range(max_steps):
            action = policy(obs) if policy else env.action_space.sample()
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


def bi_lipschitz_loss(encoder, x_batch, m_target=0.5):
    """Penalize encoder Jacobian singular values below m_target."""
    J = vmap(jacrev(encoder))(x_batch)
    sigma_mins = torch.linalg.svdvals(J)[:, -1]
    return torch.relu(m_target - sigma_mins).mean()


def controllability_loss(A, B, horizon=4):
    """Negative minimum singular value of the controllability matrix [B, AB, ..., A^{h-1}B]."""
    cols = [B]
    Ak = A
    for _ in range(horizon - 1):
        cols.append(Ak @ B)
        Ak = Ak @ A
    C_mat = torch.cat(cols, dim=1)
    sigma_min = torch.linalg.svdvals(C_mat)[-1]
    return -sigma_min


def spectral_loss(A):
    """Penalize sum of eigenvalue magnitudes of A to encourage stability."""
    eigenvalues = torch.linalg.eigvals(A)
    return eigenvalues.abs().sum()


def normality_loss(A):
    """Penalize non-normality of A: ||A^T A - A A^T||_F."""
    return torch.norm(A.T @ A - A @ A.T, p='fro')


def _pred_loss_sequential(model, states_seq, actions_seq):
    """Original sequential multi-step prediction loss."""
    z = model.encode(states_seq[:, 0])
    H = actions_seq.shape[1]
    pred_loss = torch.tensor(0.0, device=z.device)
    for t in range(H):
        z = model.predict(z, actions_seq[:, t])
        x_pred = model.decode(z)
        pred_loss = pred_loss + F.mse_loss(x_pred, states_seq[:, t + 1])
    return pred_loss / H


def _pred_loss_vectorized(model, states_seq, actions_seq):
    """Vectorized multi-step prediction loss using closed-form linear recurrence.

    Since predict is z_{t+1} = K z_t + B u_t, we have:
        z_t = K^t z_0 + sum_{k=0}^{t-1} K^{t-1-k} B u_k

    This builds K powers and uses a causal matmul to compute all z_t at once.
    """
    B_batch, H, _ = actions_seq.shape
    device = actions_seq.device

    # Get K and B matrices
    K = model.A  # (L, L)
    B_mat = model.B_matrix  # (L, A)
    L = K.shape[0]

    # Precompute K powers: K^0, K^1, ..., K^H
    K_powers = [torch.eye(L, device=device)]
    for _ in range(H):
        K_powers.append(K_powers[-1] @ K)
    K_powers = torch.stack(K_powers)  # (H+1, L, L)

    # z_0
    z0 = model.encode(states_seq[:, 0])  # (B_batch, L)

    # Free evolution: K^t z_0 for t=1..H
    # K_powers[1:H+1] is (H, L, L), z0 is (B_batch, L)
    z_free = torch.einsum('hij,bj->bhi', K_powers[1:H+1], z0)  # (B_batch, H, L)

    # Forced response: Bu for all timesteps
    Bu = actions_seq @ B_mat.T  # (B_batch, H, L)

    # Build causal convolution matrix from K powers
    # conv[t, k] = K^{t-k} for k <= t, else 0  (t=0..H-1, k=0..H-1)
    # For target step t+1, we need sum_{k=0}^{t} K^{t-k} Bu_k
    conv = torch.zeros(H, H, L, L, device=device)
    for t in range(H):
        for k in range(t + 1):
            conv[t, k] = K_powers[t - k]

    # Apply convolution: z_forced[t] = sum_k conv[t,k] @ Bu[k]
    z_forced = torch.einsum('tklm,bkm->btl', conv, Bu)  # (B_batch, H, L)

    # All predicted latent states for t=1..H
    z_all = z_free + z_forced  # (B_batch, H, L)

    # Decode all at once
    x_pred = model.decode(z_all.reshape(B_batch * H, L)).reshape(B_batch, H, -1)
    x_target = states_seq[:, 1:]  # (B_batch, H, S)

    return F.mse_loss(x_pred, x_target)


def compute_loss(model, states_seq, actions_seq, recon_weight, pred_weight,
                 ctrl_weight=0.0, ctrl_horizon=4,
                 bilip_weight=0.0, bilip_m_target=0.5,
                 spectral_weight=0.0, normality_weight=0.0,
                 vectorize_rollout=False):
    """
    Args:
        model: KoopmanAutoencoder
        states_seq: (B, H+1, state_dim)
        actions_seq: (B, H, action_dim)
        vectorize_rollout: if True, use closed-form vectorized prediction
    Returns:
        total_loss, recon_loss, pred_loss, ctrl_loss, bilip_loss, spec_loss, norm_loss
    """
    B, Hp1, S = states_seq.shape
    H = Hp1 - 1

    # Reconstruction loss: encode + decode all states in the window
    all_states = states_seq.reshape(B * Hp1, S)
    all_z = model.encode(all_states)
    all_recon = model.decode(all_z)
    recon_loss = F.mse_loss(all_recon, all_states)

    # Multi-step prediction loss
    if vectorize_rollout:
        pred_loss = _pred_loss_vectorized(model, states_seq, actions_seq)
    else:
        pred_loss = _pred_loss_sequential(model, states_seq, actions_seq)

    total_loss = recon_weight * recon_loss + pred_weight * pred_loss

    if ctrl_weight > 0:
        ctrl_loss = controllability_loss(model.A, model.B_matrix, horizon=ctrl_horizon)
        total_loss = total_loss + ctrl_weight * ctrl_loss
    else:
        ctrl_loss = torch.tensor(0.0, device=all_states.device)

    if bilip_weight > 0:
        bilip_loss_val = bi_lipschitz_loss(model.encode, all_states.detach(), m_target=bilip_m_target)
        total_loss = total_loss + bilip_weight * bilip_loss_val
    else:
        bilip_loss_val = torch.tensor(0.0, device=all_states.device)

    if spectral_weight > 0:
        spec_loss_val = spectral_loss(model.A)
        total_loss = total_loss + spectral_weight * spec_loss_val
    else:
        spec_loss_val = torch.tensor(0.0, device=all_states.device)

    if normality_weight > 0:
        norm_loss_val = normality_loss(model.A)
        total_loss = total_loss + normality_weight * norm_loss_val
    else:
        norm_loss_val = torch.tensor(0.0, device=all_states.device)

    return total_loss, recon_loss, pred_loss, ctrl_loss, bilip_loss_val, spec_loss_val, norm_loss_val


def train(model, trajectories, cfg):
    """Train a KoopmanAutoencoder on pre-collected trajectories.

    Args:
        model: KoopmanAutoencoder (already on device)
        trajectories: list of (states, actions) numpy arrays
        cfg: config dict with training hyperparameters

    Returns:
        The trained model.
    """
    device = next(model.parameters()).device
    torch.manual_seed(cfg["seed"])

    # 1. Build dataset and dataloader
    dataset = TrajectoryDataset(trajectories, cfg["horizon"])
    print(f"Dataset size: {len(dataset)} windows")
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True, pin_memory=True,
    )

    # 2. Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"],
    )

    # 2b. Optional torch.compile
    if cfg.get("torch_compile", False):
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # 3. Training loop
    loss_threshold = cfg.get("loss_threshold", 0.001)
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_total, epoch_recon, epoch_pred, epoch_ctrl, epoch_bilip, epoch_spec, epoch_norm, n_batches = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        ctrl_weight = cfg.get("ctrl_weight", 0.0) if cfg.get("controllability_loss", False) else 0.0
        bilip_weight = cfg.get("bilip_weight", 0.0) if cfg.get("bi_lipschitz_loss", False) else 0.0
        bilip_m_target = cfg.get("bilip_m_target", 0.5)
        spectral_weight = cfg.get("spectral_weight", 0.0) if cfg.get("spectral_loss", False) else 0.0
        normality_weight = cfg.get("normality_weight", 0.0) if cfg.get("normality_loss", False) else 0.0

        for states_seq, actions_seq in loader:
            states_seq = states_seq.to(device)
            actions_seq = actions_seq.to(device)

            total_loss, recon_loss, pred_loss, ctrl_loss, bilip_loss, spec_loss, norm_loss = compute_loss(
                model, states_seq, actions_seq,
                cfg["recon_weight"], cfg["pred_weight"],
                ctrl_weight=ctrl_weight, ctrl_horizon=cfg["horizon"],
                bilip_weight=bilip_weight, bilip_m_target=bilip_m_target,
                spectral_weight=spectral_weight, normality_weight=normality_weight,
                vectorize_rollout=cfg.get("vectorize_rollout", False),
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if hasattr(model.K_module, "project"):
                model.K_module.project()

            epoch_total += total_loss.item()
            epoch_recon += recon_loss.item()
            epoch_pred += pred_loss.item()
            epoch_ctrl += ctrl_loss.item()
            epoch_bilip += bilip_loss.item()
            epoch_spec += spec_loss.item()
            epoch_norm += norm_loss.item()
            n_batches += 1

        scheduler.step()

        avg_total = epoch_total / n_batches
        if epoch % cfg["log_interval"] == 0 or epoch == 1:
            msg = (
                f"Epoch {epoch:4d} | "
                f"Total: {avg_total:.6f} | "
                f"Recon: {epoch_recon / n_batches:.6f} | "
                f"Pred:  {epoch_pred / n_batches:.6f}"
            )
            if ctrl_weight > 0:
                msg += f" | Ctrl: {epoch_ctrl / n_batches:.6f}"
            if bilip_weight > 0:
                msg += f" | BiLip: {epoch_bilip / n_batches:.6f}"
            if spectral_weight > 0:
                msg += f" | Spec: {epoch_spec / n_batches:.6f}"
            if normality_weight > 0:
                msg += f" | Norm: {epoch_norm / n_batches:.6f}"
            msg += f" | LR: {scheduler.get_last_lr()[0]:.2e}"
            print(msg)

        if avg_total < loss_threshold:
            print(f"Early stop: total loss {avg_total:.6f} < threshold {loss_threshold}")
            break

    # Final loss summary
    print("\n--- Training Complete ---")
    print(f"  Total: {epoch_total / n_batches:.6f}")
    print(f"  Recon: {epoch_recon / n_batches:.6f}")
    print(f"  Pred:  {epoch_pred / n_batches:.6f}")
    if ctrl_weight > 0:
        print(f"  Ctrl:  {epoch_ctrl / n_batches:.6f}")
    if bilip_weight > 0:
        print(f"  BiLip: {epoch_bilip / n_batches:.6f}")
    if spectral_weight > 0:
        print(f"  Spec:  {epoch_spec / n_batches:.6f}")
    if normality_weight > 0:
        print(f"  Norm:  {epoch_norm / n_batches:.6f}")
    print(f"  Epochs: {epoch}/{cfg['num_epochs']}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KoopmanAutoencoder on Pendulum")
    parser.add_argument("--config", type=str, default="config/pendulum.yaml")
    parser.add_argument("--random-policy", action="store_true", default=False,
                        help="Use random actions instead of PD policy")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Collect data
    env = gym.make(cfg["env_name"])
    if args.random_policy:
        policy = None
        print("Using random policy")
    else:
        kp, kd = cfg["kp"], cfg["kd"]
        policy = lambda obs: pd_policy(obs, kp, kd)
        print(f"Using PD policy (kp={kp}, kd={kd})")

    trajectories = collect_data(
        env, cfg["num_trajectories"],
        cfg["max_episode_steps"], cfg["seed"], policy=policy,
    )
    env.close()

    # Build model
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

    # Train
    model = train(model, trajectories, cfg)

    # Save checkpoint
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], "final.pt")
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
