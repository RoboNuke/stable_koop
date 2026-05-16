"""All Koopman training loss functions + the loss-builder + the loss aggregator.

Single source of truth: every loss the codebase uses is defined exactly once
here, in semantically-grouped sections. ``build_loss_fns`` and ``compute_loss``
own the dispatch and aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import solve_discrete_are
from torch.func import jacrev, vmap

if TYPE_CHECKING:
    from config.manager import TrainKoopmanCfg


# ---------------------------------------------------------------------------
# Encoder Lipschitz losses (single source — replaces the duplicate that lived
# in launch/train_pendulum.py).
# ---------------------------------------------------------------------------


def bi_lipschitz_loss(encoder, x_batch, m_target=0.5):
    """Penalize worst-case encoder Jacobian min singular value below ``m_target``."""

    def encode_single(x):
        return encoder(x.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(encode_single))(x_batch)  # (N, latent_dim, state_dim)
    sigma_mins = torch.linalg.svdvals(J)[:, -1]
    return torch.relu(m_target - sigma_mins.min())


def upper_lipschitz_loss(encoder, x_batch, m_max=1.0):
    """Penalize encoder Jacobian singular values above ``m_max``."""
    J = vmap(jacrev(encoder))(x_batch)
    sigma_maxs = torch.linalg.svdvals(J)[:, 0]
    return torch.relu(sigma_maxs - m_max).mean()


# ---------------------------------------------------------------------------
# B-matrix structural losses.
# ---------------------------------------------------------------------------


def b_eigen_loss(b_eigen, min_scale=0.1):
    """Penalize eigenbasis B components below ``min_scale``."""
    return torch.relu(min_scale - b_eigen.abs()).mean()


def b_scale_loss(B_matrix, target_scale=1.0):
    """Penalize B spectral norm exceeding ``target_scale``."""
    current_scale = torch.linalg.norm(B_matrix, ord=2)
    return torch.relu(current_scale - target_scale)


def b_min_sv_loss(B_matrix, min_sv=0.1):
    """Penalize B minimum singular value below ``min_sv``."""
    sigma_min = torch.linalg.svdvals(B_matrix)[-1]
    return torch.relu(min_sv - sigma_min)


# ---------------------------------------------------------------------------
# A-matrix structural losses.
# ---------------------------------------------------------------------------


def eigenvalue_spread_loss(log_d, rho, min_gap=0.1):
    """Enforce a minimum gap between adjacent (sorted) NormalK eigenvalue magnitudes."""
    d = torch.sort(torch.tanh(log_d) * rho).values
    gaps = d[1:] - d[:-1]
    return torch.relu(min_gap - gaps).sum()


def unit_circle_gap_loss(A, gap=0.05):
    """Push eigenvalues away from the unit circle (either stable or clearly unstable)."""
    eigenvalues = torch.linalg.eigvals(A)
    mags = eigenvalues.abs()
    dist_from_unit = gap - torch.abs(mags - 1.0)
    return torch.relu(dist_from_unit).sum()


def spectral_loss(A):
    """Sum of eigenvalue magnitudes of A — encourages stability."""
    eigenvalues = torch.linalg.eigvals(A)
    return eigenvalues.abs().sum()


def normality_loss(A):
    """Frobenius norm of ``A^T A - A A^T`` — pushes A toward normal matrices."""
    return torch.norm(A.T @ A - A @ A.T, p="fro")


# ---------------------------------------------------------------------------
# Controllability-related losses.
# ---------------------------------------------------------------------------


def unstable_controllability_loss(A, B_matrix, threshold=1.0):
    """Encourage controllability of A's unstable subspace via projection onto B."""
    eigenvalues, V = torch.linalg.eig(A)
    unstable_mask = eigenvalues.abs() > threshold
    if not unstable_mask.any():
        return torch.tensor(0.0, device=A.device)
    V_unstable = V[:, unstable_mask]
    B_complex = B_matrix.to(torch.cfloat)
    projections = (V_unstable.conj().T @ B_complex).abs()
    return -torch.sigmoid(projections - 1.0).sum()


def controllability_loss(A, B, horizon=4):
    """Negative minimum singular value of the controllability matrix."""
    cols = [B]
    Ak = A
    for _ in range(horizon - 1):
        cols.append(Ak @ B)
        Ak = Ak @ A
    C_mat = torch.cat(cols, dim=1)
    sigma_min = torch.linalg.svdvals(C_mat)[-1]
    return -sigma_min


# ---------------------------------------------------------------------------
# Prediction / consistency losses (state space and latent space).
# ---------------------------------------------------------------------------


def l_inf_pred_loss(model, states_seq, actions_seq):
    """Max-over-batch reconstruction + one-step prediction error."""
    B_batch, Hp1, S = states_seq.shape
    H = Hp1 - 1

    all_states_flat = states_seq.reshape(B_batch * Hp1, S)
    all_z = model.encode(all_states_flat).reshape(B_batch, Hp1, -1)

    z_t = all_z[:, :H, :].reshape(B_batch * H, -1)
    x_t = states_seq[:, :H, :].reshape(B_batch * H, S)
    recon_err = torch.linalg.norm(model.decode(z_t) - x_t, dim=-1)

    u_t = actions_seq.reshape(B_batch * H, -1)
    z_pred = model.predict(z_t, u_t)
    x_target = states_seq[:, 1:, :].reshape(B_batch * H, S)
    pred_err = torch.linalg.norm(model.decode(z_pred) - x_target, dim=-1)

    return (recon_err + pred_err).max()


def _pred_loss_sequential(model, states_seq, actions_seq):
    """Original sequential multi-step prediction MSE loss."""
    z = model.encode(states_seq[:, 0])
    H = actions_seq.shape[1]
    pred_loss = torch.tensor(0.0, device=z.device)
    for t in range(H):
        z = model.predict(z, actions_seq[:, t])
        x_pred = model.decode(z)
        pred_loss = pred_loss + F.mse_loss(x_pred, states_seq[:, t + 1])
    return pred_loss / H


def _pred_loss_vectorized(model, states_seq, actions_seq):
    """Multi-step prediction loss starting from every timestep in each window."""
    B_batch, Hp1, S = states_seq.shape
    H = Hp1 - 1
    device = states_seq.device

    all_states_flat = states_seq.reshape(B_batch * Hp1, S)
    all_z = model.encode(all_states_flat).reshape(B_batch, Hp1, -1)

    total_loss = torch.tensor(0.0, device=device)
    n_predictions = 0
    for t_start in range(H):
        z = all_z[:, t_start, :]
        for t in range(t_start, H):
            z = model.predict(z, actions_seq[:, t, :])
            x_pred = model.decode(z)
            x_target = states_seq[:, t + 1, :]
            total_loss = total_loss + F.mse_loss(x_pred, x_target)
            n_predictions += 1
    return total_loss / n_predictions


def closed_loop_normality_loss(A, B, Q, R):
    """Normality loss on the closed-loop matrix ``A - B @ F`` (F via LQR with no_grad)."""
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


def latent_rollout_consistency_loss(model, states_seq, actions_seq):
    """Multi-step latent consistency loss starting from every timestep."""
    B_batch, Hp1, S = states_seq.shape
    H = Hp1 - 1
    device = states_seq.device

    all_states_flat = states_seq.reshape(B_batch * Hp1, S)
    all_z = model.encode(all_states_flat).reshape(B_batch, Hp1, -1)

    total_loss = torch.tensor(0.0, device=device)
    n_predictions = 0
    for t_start in range(H):
        z = all_z[:, t_start, :]
        for t in range(t_start, H):
            z = model.predict(z, actions_seq[:, t, :])
            z_target = all_z[:, t + 1, :]
            total_loss = total_loss + F.mse_loss(z, z_target)
            n_predictions += 1
    return total_loss / n_predictions


def latent_consistency_loss(model, states_seq, actions_seq):
    """Enforce ``z_{t+1} ≈ A z_t + B u_t`` directly in latent space (vectorized)."""
    B_batch, Hp1, S = states_seq.shape
    H = Hp1 - 1

    all_states = states_seq.reshape(B_batch * Hp1, S)
    all_z = model.encode(all_states).reshape(B_batch, Hp1, -1)
    u_t = actions_seq.reshape(B_batch * H, -1)

    z_t = all_z[:, :H].reshape(B_batch * H, -1)
    z_pred = model.predict(z_t, u_t)
    z_target = all_z[:, 1:].reshape(B_batch * H, -1)
    return F.mse_loss(z_pred, z_target)


# ---------------------------------------------------------------------------
# Loss-builder + aggregator.
#
# ``build_loss_fns`` reads from a :class:`TrainKoopmanCfg` dataclass. Per-loss
# weights / targets / enable flags live on ``train_cfg.losses`` (LossesCfg);
# the few cross-section reads (``latent_dim``, ``action_dim``, ``horizon``,
# ``rho``, ``vectorize_rollout``) live at the top level of ``train_cfg``.
# ---------------------------------------------------------------------------


def build_loss_fns(model, train_cfg: "TrainKoopmanCfg"):
    """Return ``{name: (fn, weight)}`` for every active loss given ``train_cfg``."""
    L = train_cfg.losses
    losses = {}

    if model.decoder is not None:
        def _recon(model, states_seq, actions_seq, all_states):
            all_z = model.encode(all_states)
            return F.mse_loss(model.decode(all_z), all_states)
        losses["Recon"] = (_recon, L.recon_weight)

    if getattr(model, "prepend_state", False):
        xpred_w = L.x_pred_weight
        p_dim = model.prepend_dim

        def _xpred(model, states_seq, actions_seq, all_states):
            B_batch, Hp1, S = states_seq.shape
            H = Hp1 - 1
            device = states_seq.device
            all_states_flat = states_seq.reshape(B_batch * Hp1, S)
            all_z = model.encode(all_states_flat).reshape(B_batch, Hp1, -1)
            total_loss = torch.tensor(0.0, device=device)
            n_predictions = 0
            for t_start in range(H):
                z = all_z[:, t_start, :]
                for t in range(t_start, H):
                    z = model.predict(z, actions_seq[:, t, :])
                    z_target = all_z[:, t + 1, :]
                    total_loss = total_loss + F.mse_loss(
                        z[:, :p_dim], z_target[:, :p_dim]
                    )
                    n_predictions += 1
            return total_loss / n_predictions

        losses["XPred"] = (_xpred, xpred_w)

    vectorize = train_cfg.vectorize_rollout

    def _pred(model, states_seq, actions_seq, all_states):
        if vectorize:
            return _pred_loss_vectorized(model, states_seq, actions_seq)
        return _pred_loss_sequential(model, states_seq, actions_seq)

    losses["Pred"] = (_pred, L.pred_weight)

    if L.l_inf_pred_loss:
        def _linf(model, states_seq, actions_seq, all_states):
            return l_inf_pred_loss(model, states_seq, actions_seq)
        losses["L_inf"] = (_linf, L.l_inf_pred_weight)

    def _lc(model, states_seq, actions_seq, all_states):
        return latent_consistency_loss(model, states_seq, actions_seq)

    losses["LC"] = (_lc, L.latent_consistency_weight)

    if L.latent_rollout_loss:
        def _lrc(model, states_seq, actions_seq, all_states):
            return latent_rollout_consistency_loss(model, states_seq, actions_seq)

        losses["LRC"] = (_lrc, L.latent_rollout_weight)

    if L.controllability_loss:
        horizon = train_cfg.horizon

        def _ctrl(model, states_seq, actions_seq, all_states):
            return controllability_loss(model.A, model.B_matrix, horizon=horizon)

        losses["Ctrl"] = (_ctrl, L.ctrl_weight)

    if L.unstable_ctrl_loss:
        uc_threshold = L.unstable_ctrl_threshold

        def _uctrl(model, states_seq, actions_seq, all_states):
            return unstable_controllability_loss(
                model.A, model.B_matrix, threshold=uc_threshold
            )

        losses["UCtrl"] = (_uctrl, L.unstable_ctrl_weight)

    if L.bi_lipschitz_loss:
        m_target = L.bilip_m_target

        def _bilip(model, states_seq, actions_seq, all_states):
            return bi_lipschitz_loss(model.encode, all_states.detach(), m_target=m_target)

        losses["BiLip"] = (_bilip, L.bilip_weight)

    if L.spectral_loss:
        def _spec(model, states_seq, actions_seq, all_states):
            return spectral_loss(model.A)

        losses["Spec"] = (_spec, L.spectral_weight)

    if L.normality_loss:
        def _norm(model, states_seq, actions_seq, all_states):
            return normality_loss(model.A)

        losses["Norm"] = (_norm, L.normality_weight)

    if L.cl_normality_loss:
        # cl_normality_loss reads Q/R from the LQR config, not the training
        # config. The training pipeline doesn't have access to those; legacy
        # code defaulted to identity scaling (q_scale=1.0, r_scale=1.0).
        cl_Q = torch.eye(train_cfg.latent_dim)
        cl_R = torch.eye(train_cfg.action_dim)

        def _cl_norm(model, states_seq, actions_seq, all_states):
            return closed_loop_normality_loss(model.A, model.B_matrix, cl_Q, cl_R)

        losses["CLNorm"] = (_cl_norm, L.cl_normality_weight)

    if L.b_eigen_loss and hasattr(model.K_module, "b_eigen"):
        min_scale = L.b_eigen_min_scale

        def _beig(model, states_seq, actions_seq, all_states):
            return b_eigen_loss(model.K_module.b_eigen, min_scale=min_scale)

        losses["BEig"] = (_beig, L.b_eigen_weight)

    if L.b_scale_loss:
        target_scale = L.b_scale_target

        def _bscale(model, states_seq, actions_seq, all_states):
            return b_scale_loss(model.B_matrix, target_scale=target_scale)

        losses["BScl"] = (_bscale, L.b_scale_weight)

    if L.b_min_sv_loss:
        min_sv = L.b_min_sv_target

        def _bminsv(model, states_seq, actions_seq, all_states):
            return b_min_sv_loss(model.B_matrix, min_sv=min_sv)

        losses["BMinSV"] = (_bminsv, L.b_min_sv_weight)

    if L.eig_spread_loss and hasattr(model.K_module, "log_d"):
        min_gap = L.eig_spread_min_gap
        rho = train_cfg.rho

        def _espread(model, states_seq, actions_seq, all_states):
            return eigenvalue_spread_loss(model.K_module.log_d, rho, min_gap=min_gap)

        losses["ESprd"] = (_espread, L.eig_spread_weight)

    if L.unit_circle_gap_loss:
        gap = L.unit_circle_gap

        def _ucgap(model, states_seq, actions_seq, all_states):
            return unit_circle_gap_loss(model.A, gap=gap)

        losses["UCGap"] = (_ucgap, L.unit_circle_gap_weight)

    if L.upper_lipschitz_loss:
        m_max = L.upper_lip_m_max

        def _ulip(model, states_seq, actions_seq, all_states):
            return upper_lipschitz_loss(model.encode, all_states.detach(), m_max=m_max)

        losses["ULip"] = (_ulip, L.upper_lip_weight)

    return losses


def compute_loss(model, states_seq, actions_seq, loss_fns):
    """Sum all active losses with their weights; return ``(total, dict)``."""
    B, Hp1, S = states_seq.shape
    all_states = states_seq.reshape(B * Hp1, S)

    total_loss = torch.tensor(0.0, device=states_seq.device)
    losses = {}
    for name, (fn, weight) in loss_fns.items():
        val = fn(model, states_seq, actions_seq, all_states)
        losses[name] = val
        total_loss = total_loss + weight * val
    return total_loss, losses
