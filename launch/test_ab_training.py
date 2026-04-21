"""
Test A and B matrix learning with known ground truth.

Generates trajectories from a known linear system, trains a Koopman model
with identity encoder, and compares learned A/B to ground truth.

Usage:
    python -m launch.test_ab_training
"""
import numpy as np
import torch
import yaml

from model.autoencoder import KoopmanAutoencoder
from launch.train_pendulum import train

# =====================================================================
#  Ground Truth System
# =====================================================================
# Stable, well-conditioned, controllable
A_true = np.array([
    [0.9, 0.1, 0.0, 0.0],
    [-0.1, 0.8, 0.05, 0.0],
    [0.0, 0.0, 0.7, 0.2],
    [0.0, 0.0, -0.2, 0.7]
], dtype=np.float32)

B_true = np.array([
    [1.0],
    [0.5],
    [0.0],
    [0.3]
], dtype=np.float32)

# Properties:
# eigenvalues: ~0.85±0.1j, 0.7±0.2j (all stable)
# ||A|| ≈ 0.95
# ||B|| ≈ 1.17
# rank(controllability) = 4


def generate_trajectories(A, B, num_traj=500, steps=25, x_range=2.0, u_range=1.0):
    """Generate trajectories from x_{t+1} = A x_t + B u_t."""
    n, m = A.shape[0], B.shape[1]
    trajectories = []
    for _ in range(num_traj):
        x = np.random.uniform(-x_range, x_range, size=(n,)).astype(np.float32)
        states = [x.copy()]
        actions = []
        for t in range(steps):
            u = np.random.uniform(-u_range, u_range, size=(m,)).astype(np.float32)
            x = A @ x + B @ u
            states.append(x.copy())
            actions.append(u.copy())
        states = np.array(states, dtype=np.float32)   # (steps+1, n)
        actions = np.array(actions, dtype=np.float32)  # (steps, m)
        trajectories.append((states, actions))
    return trajectories


def ctrl_rank(A, B):
    n = A.shape[0]
    C = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    return np.linalg.matrix_rank(C)


def main():
    # Load config for training hyperparameters
    with open("config/caylay_a.yaml") as f:
        cfg = yaml.safe_load(f)

    state_dim = A_true.shape[0]
    action_dim = B_true.shape[1]

    print("=" * 60)
    print("  Ground Truth System")
    print("=" * 60)
    print(f"A_true ({state_dim}x{state_dim}):")
    print(A_true)
    A_eigvals = np.linalg.eigvals(A_true)
    print(f"  ||A_true|| = {np.linalg.norm(A_true, ord=2):.6f}")
    print(f"  eigenvalues: {[f'{e:.4f}' for e in A_eigvals]}")
    print(f"  spectral radius: {np.max(np.abs(A_eigvals)):.6f}")
    print(f"\nB_true ({state_dim}x{action_dim}):")
    print(B_true)
    print(f"  ||B_true|| = {np.linalg.norm(B_true, ord=2):.6f}")
    print(f"  controllability rank: {ctrl_rank(A_true, B_true)}/{state_dim}")

    # Generate data
    print(f"\n{'='*60}")
    print("  Generating Trajectories")
    print("=" * 60)
    trajectories = generate_trajectories(A_true, B_true,
                                          num_traj=cfg.get("num_trajectories", 500),
                                          steps=25)
    print(f"Generated {len(trajectories)} trajectories, "
          f"{sum(len(a) for _, a in trajectories)} transitions")

    # Build model with identity encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Override config for this test
    cfg["state_dim"] = state_dim
    cfg["action_dim"] = action_dim
    cfg["latent_dim"] = state_dim  # identity encoder: latent = state
    cfg["encoder_type"] = "identity"
    cfg["prepend_state"] = False
    cfg["recon_pretrain_epochs"] = 0

    model = KoopmanAutoencoder(
        state_dim=state_dim,
        latent_dim=state_dim,
        action_dim=action_dim,
        k_type=cfg["k_type"],
        encoder_type="identity",
        rho=cfg["rho"],
        encoder_spec_norm=False,
        encoder_latent=64,
        prepend_state=False,
        real_state_dim=state_dim,
    ).to(device)

    print(f"Model: k_type={cfg['k_type']}, rho={cfg['rho']}")
    print(f"  A shape: {model.A.shape}")
    print(f"  B shape: {model.B_matrix.shape}")
    print(f"  Learnable parameters: {sum(p.numel() for p in model.parameters())}")

    # Train
    print(f"\n{'='*60}")
    print("  Training")
    print("=" * 60)
    model = train(model, trajectories, cfg)

    # Extract learned matrices
    with torch.no_grad():
        A_learned = model.A.detach().cpu().numpy()
        B_learned = model.B_matrix.detach().cpu().numpy()

    # =====================================================================
    #  Comparison
    # =====================================================================
    print(f"\n{'='*60}")
    print("  Results")
    print("=" * 60)

    # A matrix comparison
    A_diff = A_learned - A_true
    print(f"\nA_learned:")
    print(A_learned)
    print(f"\nA_true - A_learned (element-wise):")
    print(A_diff)
    print(f"  ||A_diff||_F = {np.linalg.norm(A_diff, 'fro'):.6f}")
    print(f"  max |diff| = {np.max(np.abs(A_diff)):.6f}")

    A_learned_eigvals = np.linalg.eigvals(A_learned)
    print(f"\n  ||A_true||  = {np.linalg.norm(A_true, ord=2):.6f}")
    print(f"  ||A_learned|| = {np.linalg.norm(A_learned, ord=2):.6f}")
    print(f"\n  Eigenvalues (true):    {sorted([f'{e:.4f}' for e in A_eigvals])}")
    print(f"  Eigenvalues (learned): {sorted([f'{e:.4f}' for e in A_learned_eigvals])}")
    print(f"  Spectral radius (true):    {np.max(np.abs(A_eigvals)):.6f}")
    print(f"  Spectral radius (learned): {np.max(np.abs(A_learned_eigvals)):.6f}")

    # B matrix comparison
    B_diff = B_learned - B_true
    print(f"\nB_learned:")
    print(B_learned)
    print(f"\nB_true - B_learned (element-wise):")
    print(B_diff)
    print(f"  ||B_diff||_F = {np.linalg.norm(B_diff, 'fro'):.6f}")
    print(f"  max |diff| = {np.max(np.abs(B_diff)):.6f}")

    print(f"\n  ||B_true||  = {np.linalg.norm(B_true, ord=2):.6f}")
    print(f"  ||B_learned|| = {np.linalg.norm(B_learned, ord=2):.6f}")

    # Controllability
    print(f"\n  Controllability rank (true):    {ctrl_rank(A_true, B_true)}/{state_dim}")
    print(f"  Controllability rank (learned): {ctrl_rank(A_learned, B_learned)}/{state_dim}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
