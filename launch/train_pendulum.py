import os
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap
from scipy.linalg import solve_discrete_are
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

def energy_shaping_policy(obs, kp=10.0, kd=3.0, k_e=2.0, switch_angle=0.7854): #0.9599):#1.0472):
    """Energy-shaping swing-up + PD balance for Pendulum-v1.

    Pendulum-v1 physics: m=1, l=1, g=10, I=ml^2/3=1/3, max_torque=2.
    Energy: E = thdot^2/6 + 5*cos_th.  E_upright = 5 (theta=0, thdot=0).

    Args:
        switch_angle: radians, switch to PD balance when |theta| < this value.
    """
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)

    E = thdot**2 / 6.0 + 5.0 * cos_th
    E_target = 5.0  # energy at upright equilibrium

    u_swing = k_e * thdot * (E_target - E)

    # PD balance near upright
    u_balance = -kp * theta - kd * thdot

    near_top = abs(theta) < switch_angle
    u = u_balance if near_top else u_swing
    return np.array([np.clip(u, -2.0, 2.0)])

def bang_energy_policy(obs, kp=10.0, kd=3.0, k_e=2.0, switch_angle=1.0472):
    """Bang-bang energy shaping swing-up + PD balance for Pendulum-v1.

    Uses sign-based energy pumping instead of continuous proportional control.
    Pendulum-v1 physics: m=1, l=1, g=10, I=ml^2/3=1/3, max_torque=2.
    Energy: E = thdot^2/6 + 5*cos_th.  E_upright = 5 (theta=0, thdot=0).

    When E < E_target: pump energy via bang-bang in direction of thdot*cos_th.
    When E > E_target: brake at half gain to shed excess energy.

    Args:
        switch_angle: radians, switch to PD balance when |theta| < this value.
    """
    cos_th, sin_th, thdot = obs
    theta = np.arctan2(sin_th, cos_th)

    E = thdot**2 / 6.0 - 5.0 * cos_th
    E_target = -5.0

    if E < E_target:
        u_swing = k_e * np.sign(thdot * cos_th)
    else:
        u_swing = -k_e * np.sign(thdot * cos_th) * 0.5

    u_balance = -kp * theta - kd * thdot

    near_top = abs(theta) < switch_angle
    u = u_balance if near_top else u_swing
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


"""
def bi_lipschitz_loss(encoder, x_batch, m_target=0.5):
    #Penalize encoder Jacobian singular values below m_target.
    J = vmap(jacrev(encoder))(x_batch)
    sigma_mins = torch.linalg.svdvals(J)[:, -1]
    return torch.relu(m_target - sigma_mins).mean()
"""

def bi_lipschitz_loss(encoder, x_batch, m_target=0.5):
    """Penalize encoder Jacobian singular values below m_target."""
    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)
    J = vmap(jacrev(encode_single))(x_batch)  # (N, latent_dim, state_dim)
    sigma_mins = torch.linalg.svdvals(J)[:, -1]
    return torch.relu(m_target - sigma_mins).mean()

def upper_lipschitz_loss(encoder, x_batch, m_max=1.0):
    """Penalize encoder Jacobian singular values above m_max."""
    J = vmap(jacrev(encoder))(x_batch)
    sigma_maxs = torch.linalg.svdvals(J)[:, 0]
    return torch.relu(sigma_maxs - m_max).mean()


def b_eigen_loss(b_eigen, min_scale=0.1):
    """Penalize eigenbasis B components below min_scale."""
    return torch.relu(min_scale - b_eigen.abs()).mean()


def b_scale_loss(B_matrix, target_scale=1.0):
    """Penalize B spectral norm exceeding target_scale."""
    current_scale = torch.norm(B_matrix, p=2)
    return torch.relu(current_scale - target_scale)


def eigenvalue_spread_loss(log_d, rho, min_gap=0.1):
    d = torch.sort(torch.tanh(log_d) * rho).values
    gaps = d[1:] - d[:-1]
    return torch.relu(min_gap - gaps).sum()


def unit_circle_gap_loss(A, gap=0.05):
    """Penalize eigenvalues within gap of the unit circle.

    Pushes eigenvalues to be either clearly stable (|λ| < 1-gap)
    or clearly unstable (|λ| > 1+gap). Near-unit eigenvalues
    make the Lyapunov decrease trivially small regardless of Q.
    """
    eigenvalues = torch.linalg.eigvals(A)
    mags = eigenvalues.abs()
    dist_from_unit = gap - torch.abs(mags - 1.0)
    return torch.relu(dist_from_unit).sum()


def unstable_controllability_loss(A, B_matrix, threshold=1.0):
    eigenvalues, V = torch.linalg.eig(A)
    unstable_mask = eigenvalues.abs() > threshold
    if not unstable_mask.any():
        return torch.tensor(0.0, device=A.device)
    V_unstable = V[:, unstable_mask]
    B_complex = B_matrix.to(torch.cfloat)
    projections = (V_unstable.conj().T @ B_complex).abs()
    # Normalize: sigmoid drives projections toward 1, saturates naturally
    return -torch.sigmoid(projections - 1.0).sum()


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


def closed_loop_normality_loss(A, B, Q, R):
    """Normality loss on the closed-loop matrix A - B @ F.

    F is computed via LQR with no_grad so gradients flow only through
    A and B in the closed-loop construction.
    """
    with torch.no_grad():
        A_np = A.detach().cpu().numpy()
        B_np = B.detach().cpu().numpy()
        P = solve_discrete_are(A_np, B_np, Q.numpy(), R.numpy())
        P = torch.from_numpy(P).to(A.dtype).to(A.device)
        F_current = torch.linalg.solve(
            R.to(A.device) + B.detach().T @ P @ B.detach(),
            B.detach().T @ P @ A.detach(),
        )

    cl = A - B @ F_current
    return normality_loss(cl)


def latent_consistency_loss(model, states_seq, actions_seq):
    """Enforce z_{t+1} ≈ A*z_t + B*u_t directly in latent space (vectorized)."""
    B_batch, Hp1, S = states_seq.shape
    H = Hp1 - 1

    all_states = states_seq.reshape(B_batch * Hp1, S)
    all_z = model.encode(all_states).reshape(B_batch, Hp1, -1)

    z_t = all_z[:, :H].reshape(B_batch * H, -1)       # (B*H, L)
    u_t = actions_seq.reshape(B_batch * H, -1)          # (B*H, A)
    z_pred = model.predict(z_t, u_t)                    # (B*H, L)

    z_target = all_z[:, 1:].detach().reshape(B_batch * H, -1)  # (B*H, L)
    return F.mse_loss(z_pred, z_target)


def build_loss_fns(cfg, model):
    """Build a dict of {name: (loss_fn, weight)} from config.

    Each loss_fn has signature: fn(model, states_seq, actions_seq, all_states) -> scalar tensor.
    The all_states arg is the pre-flattened (B*(H+1), S) tensor for losses that need it.

    Returns:
        dict of {name: (fn, weight)}
    """
    losses = {}

    # --- Always-on core losses ---
    recon_w = cfg["recon_weight"]
    pred_w = cfg["pred_weight"]
    lc_w = cfg.get("latent_consistency_weight", 1.0)
    vectorize = cfg.get("vectorize_rollout", False)

    def _recon(model, states_seq, actions_seq, all_states):
        all_z = model.encode(all_states)
        return F.mse_loss(model.decode(all_z), all_states)
    losses["Recon"] = (_recon, recon_w)

    def _pred(model, states_seq, actions_seq, all_states):
        if vectorize:
            return _pred_loss_vectorized(model, states_seq, actions_seq)
        return _pred_loss_sequential(model, states_seq, actions_seq)
    losses["Pred"] = (_pred, pred_w)

    def _lc(model, states_seq, actions_seq, all_states):
        return latent_consistency_loss(model, states_seq, actions_seq)
    losses["LC"] = (_lc, lc_w)

    # --- Optional losses (enabled by config bool + weight) ---

    if cfg.get("controllability_loss", False):
        horizon = cfg["horizon"]
        def _ctrl(model, states_seq, actions_seq, all_states):
            return controllability_loss(model.A, model.B_matrix, horizon=horizon)
        losses["Ctrl"] = (_ctrl, cfg["ctrl_weight"])

    if cfg.get("unstable_ctrl_loss", False):
        uc_threshold = cfg.get("unstable_ctrl_threshold", 1.0)
        def _uctrl(model, states_seq, actions_seq, all_states):
            return unstable_controllability_loss(model.A, model.B_matrix, threshold=uc_threshold)
        losses["UCtrl"] = (_uctrl, cfg["unstable_ctrl_weight"])

    if cfg.get("bi_lipschitz_loss", False):
        m_target = cfg.get("bilip_m_target", 0.5)
        def _bilip(model, states_seq, actions_seq, all_states):
            return bi_lipschitz_loss(model.encode, all_states.detach(), m_target=m_target)
        losses["BiLip"] = (_bilip, cfg["bilip_weight"])

    if cfg.get("spectral_loss", False):
        def _spec(model, states_seq, actions_seq, all_states):
            return spectral_loss(model.A)
        losses["Spec"] = (_spec, cfg["spectral_weight"])

    if cfg.get("normality_loss", False):
        def _norm(model, states_seq, actions_seq, all_states):
            return normality_loss(model.A)
        losses["Norm"] = (_norm, cfg["normality_weight"])

    if cfg.get("cl_normality_loss", False):
        latent_dim = cfg["latent_dim"]
        action_dim = cfg["action_dim"]
        cl_Q = torch.eye(latent_dim) * cfg.get("q_scale", 1.0)
        cl_R = torch.eye(action_dim) * cfg.get("r_scale", 1.0)
        def _cl_norm(model, states_seq, actions_seq, all_states):
            return closed_loop_normality_loss(model.A, model.B_matrix, cl_Q, cl_R)
        losses["CLNorm"] = (_cl_norm, cfg["cl_normality_weight"])

    if cfg.get("b_eigen_loss", False) and hasattr(model.K_module, 'b_eigen'):
        min_scale = cfg.get("b_eigen_min_scale", 0.1)
        def _beig(model, states_seq, actions_seq, all_states):
            return b_eigen_loss(model.K_module.b_eigen, min_scale=min_scale)
        losses["BEig"] = (_beig, cfg["b_eigen_weight"])

    if cfg.get("b_scale_loss", False):
        target_scale = cfg.get("b_scale_target", 1.0)
        def _bscale(model, states_seq, actions_seq, all_states):
            return b_scale_loss(model.B_matrix, target_scale=target_scale)
        losses["BScl"] = (_bscale, cfg["b_scale_weight"])

    if cfg.get("eig_spread_loss", False) and hasattr(model.K_module, 'log_d'):
        min_gap = cfg.get("eig_spread_min_gap", 0.1)
        rho = cfg.get("rho", 1.0)
        def _espread(model, states_seq, actions_seq, all_states):
            return eigenvalue_spread_loss(model.K_module.log_d, rho, min_gap=min_gap)
        losses["ESprd"] = (_espread, cfg["eig_spread_weight"])

    if cfg.get("unit_circle_gap_loss", False):
        gap = cfg.get("unit_circle_gap", 0.05)
        def _ucgap(model, states_seq, actions_seq, all_states):
            return unit_circle_gap_loss(model.A, gap=gap)
        losses["UCGap"] = (_ucgap, cfg["unit_circle_gap_weight"])

    if cfg.get("upper_lipschitz_loss", False):
        m_max = cfg.get("upper_lip_m_max", 1.0)
        def _ulip(model, states_seq, actions_seq, all_states):
            return upper_lipschitz_loss(model.encode, all_states.detach(), m_max=m_max)
        losses["ULip"] = (_ulip, cfg["upper_lip_weight"])

    return losses


def compute_loss(model, states_seq, actions_seq, loss_fns):
    """Compute all losses and return total + individual values.

    Args:
        model: KoopmanAutoencoder
        states_seq: (B, H+1, state_dim)
        actions_seq: (B, H, action_dim)
        loss_fns: dict from build_loss_fns

    Returns:
        total_loss: scalar tensor
        losses: dict of {name: scalar tensor}
    """
    B, Hp1, S = states_seq.shape
    all_states = states_seq.reshape(B * Hp1, S)

    total_loss = torch.tensor(0.0, device=states_seq.device)
    losses = {}
    for name, (fn, weight) in loss_fns.items():
        val = fn(model, states_seq, actions_seq, all_states)
        losses[name] = val
        total_loss = total_loss + weight * val

    return total_loss, losses


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
    num_workers = cfg.get("num_workers", 0)
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        print(f"DataLoader: {num_workers} workers (persistent)")

    # 2. Optimizer and scheduler
    if cfg.get("riemannian_optimizer", False):
        import geoopt
        optimizer = geoopt.optim.RiemannianAdam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
        print("Using RiemannianAdam optimizer")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
    if cfg.get("cosine_scheduler", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["cosine_T_max"], eta_min=cfg["cosine_eta_min"],
        )
        print(f"Using CosineAnnealingLR scheduler (T_max={cfg['cosine_T_max']}, eta_min={cfg['cosine_eta_min']})")
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg["scheduler_step"], gamma=cfg["scheduler_gamma"],
        )
        print(f"Using StepLR scheduler (step_size={cfg['scheduler_step']}, gamma={cfg['scheduler_gamma']})")

    # 2b. Optional torch.compile
    if cfg.get("torch_compile", False):
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # 3. Build loss functions from config
    loss_fns = build_loss_fns(cfg, model)
    active_names = list(loss_fns.keys())
    print(f"Active losses: {', '.join(f'{n} (w={loss_fns[n][1]})' for n in active_names)}")

    # 4. Training loop
    import copy
    loss_threshold = cfg.get("loss_threshold", 0.001)
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        epoch_accum = {name: 0.0 for name in active_names}
        epoch_total = 0.0
        n_batches = 0

        for states_seq, actions_seq in loader:
            states_seq = states_seq.to(device)
            actions_seq = actions_seq.to(device)

            total_loss, losses = compute_loss(model, states_seq, actions_seq, loss_fns)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if hasattr(model.K_module, "project"):
                model.K_module.project()

            epoch_total += total_loss.item()
            for name in active_names:
                epoch_accum[name] += losses[name].item()
            n_batches += 1

        scheduler.step()

        avg_total = epoch_total / n_batches

        # Track best model
        if avg_total < best_loss:
            best_loss = avg_total
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        if epoch % cfg["log_interval"] == 0 or epoch == 1:
            parts = [f"Epoch {epoch:4d} | Total: {avg_total:.6f}"]
            for name in active_names:
                parts.append(f"{name}: {epoch_accum[name] / n_batches:.6f}")
            parts.append(f"LR: {scheduler.get_last_lr()[0]:.2e}")
            print(" | ".join(parts))

        if avg_total < loss_threshold:
            print(f"Early stop: total loss {avg_total:.6f} < threshold {loss_threshold}")
            break

    # Restore best model
    model.load_state_dict(best_state_dict)

    # Final loss summary
    print("\n--- Training Complete ---")
    print(f"  Best total: {best_loss:.6f} (epoch {best_epoch})")
    print(f"  Final total: {epoch_total / n_batches:.6f}")
    for name in active_names:
        print(f"  {name:>6s}: {epoch_accum[name] / n_batches:.6f}")
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
        encoder_type=cfg.get("encoder_type", "linear"),
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
    save_dict = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    torch.save({"model": save_dict, "config": cfg}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")
