"""Joint training paradigm: encoder + A + B trained together.

When ``train_cfg.b_fitting.enabled`` is set, after the gradient-descent
training a second (perturbed) dataset is loaded, B is reinitialized so the
joint training has no residual influence on it, and B is re-optimized via
either :func:`train_koopman.b_fitting.gradient_b` (gradient descent with a
controllability regularizer) or
:func:`train_koopman.b_fitting.least_squares_projected_b` (closed-form
least-squares followed by a PBH/SVD controllability projection). The fitted
B replaces ``model.B.weight`` before the final checkpoint is saved.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.manager import TrainKoopmanCfg
from data.augmentation import augment_trajectories
from data.dataloader import load_dataset
from train_koopman.b_fitting import gradient_b, least_squares_projected_b
from train_koopman.checkpointing import (
    build_koopman_model,
    experiment_dir,
    make_device,
    save_checkpoint,
)
from train_koopman.save_performance import save_model_performance
from train_koopman.training_loop import train


def _reinit_b(model) -> None:
    """Zero out any joint-training influence on B before the analytical refit."""
    with torch.no_grad():
        nn.init.kaiming_uniform_(model.B.weight)
    print("  Reinitialized B (kaiming-uniform) before B fit")


def _encode_transitions(model, aug_trajectories, device):
    """Encode every transition once and return ``(z_t, z_next, u_t)`` stacked.

    The encoder is called once per trajectory over all of its states; ``z_t``
    and ``z_next`` are slices of the same encoded sequence (matches legacy).
    """
    model.eval()
    z_t_list, z_next_list, u_list = [], [], []
    with torch.no_grad():
        for states, actions in aug_trajectories:
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
            T_act = len(actions)
            z_all = model.encode(states_t)
            z_t_list.append(z_all[:T_act])
            z_next_list.append(z_all[1 : T_act + 1])
            u_list.append(actions_t[:T_act])
    return torch.cat(z_t_list, dim=0), torch.cat(z_next_list, dim=0), torch.cat(u_list, dim=0)


def _refit_b_analytical(model, train_cfg: TrainKoopmanCfg, device) -> dict:
    """Reinit B, encode perturbed trajectories, fit B with the chosen method.

    Returns a small summary dict describing the refit for inclusion in
    ``model_performance.yaml``.
    """
    bcfg = train_cfg.b_fitting
    if not bcfg.perturbed_dataset_name:
        raise ValueError(
            "b_fitting.enabled=True requires b_fitting.perturbed_dataset_name."
        )

    print("\n=== Analytical B refit ===")
    print(f"  method={bcfg.method}  dataset={bcfg.perturbed_dataset_name}")

    _reinit_b(model)

    ds_perturbed = load_dataset(bcfg.perturbed_dataset_name)
    aug = augment_trajectories(ds_perturbed, train_cfg.augmentation)
    z_t, z_next, u_t = _encode_transitions(model, aug, device)
    A = model.A.detach().to(z_t.dtype).to(device)
    B_before = model.B_matrix.detach().clone()

    if bcfg.method == "gradient":
        print(
            f"  ctrl_lambda={bcfg.ctrl_lambda}  steps={bcfg.train_steps}  "
            f"lr={bcfg.lr}  init=least-squares"
        )
        B_new = gradient_b(
            A,
            z_t,
            z_next,
            u_t,
            ctrl_lambda=bcfg.ctrl_lambda,
            train_steps=bcfg.train_steps,
            lr=bcfg.lr,
            init_B=None,
            log_interval=bcfg.log_interval,
        )
    elif bcfg.method == "least_squares_projected":
        B_new = least_squares_projected_b(A, z_t, z_next, u_t)
    else:
        raise ValueError(
            f"Unknown b_fitting.method={bcfg.method!r}; "
            "expected 'gradient' or 'least_squares_projected'."
        )

    with torch.no_grad():
        model.B.weight.copy_(B_new.to(model.B.weight.dtype).to(model.B.weight.device))
    b_norm_before = float(torch.linalg.norm(B_before, ord=2).item())
    b_norm_after = float(torch.linalg.norm(B_new, ord=2).item())
    print(f"  ||B||_2 before refit: {b_norm_before:.4e}  after: {b_norm_after:.4e}")
    return {
        "method": bcfg.method,
        "perturbed_dataset_name": bcfg.perturbed_dataset_name,
        "B_norm_before": b_norm_before,
        "B_norm_after": b_norm_after,
    }


def run(train_cfg: TrainKoopmanCfg) -> str:
    out_dir = experiment_dir(train_cfg.experiment_name)

    device = make_device()
    ds = load_dataset(train_cfg.dataset_name)
    state_dim = ds.state_dim
    action_dim = ds.action_dim

    model, _ = build_koopman_model(
        train_cfg, state_dim=state_dim, action_dim=action_dim, device=device
    )
    aug_trajectories = augment_trajectories(ds, train_cfg.augmentation)

    print("\n=== Joint Training: A + B + Encoder Together ===")
    model, training_stats = train(model, aug_trajectories, train_cfg)

    b_summary = None
    if train_cfg.b_fitting.enabled:
        b_summary = _refit_b_analytical(model, train_cfg, device)

    save_checkpoint(
        model,
        train_cfg,
        state_dim=state_dim,
        action_dim=action_dim,
        path=out_dir / "koopman_ckpt.pt",
    )
    print(f"Checkpoint saved to {out_dir / 'koopman_ckpt.pt'}")

    save_model_performance(
        out_dir=out_dir,
        model=model,
        train_cfg=train_cfg,
        aug_trajectories=aug_trajectories,
        device=device,
        training_stats=training_stats,
        b_fitting_summary=b_summary,
        ds=ds,
    )
    return str(out_dir)
