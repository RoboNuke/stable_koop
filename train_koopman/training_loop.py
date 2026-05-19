"""Core Koopman training loop — paradigm-agnostic optimizer/scheduler driver.

Called by both ``two_phase`` and ``joint``. Reads its config directly from
the :class:`config.manager.TrainKoopmanCfg` dataclass; nothing flat-dict.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.manager import TrainKoopmanCfg
from data.dataloader import TrajectoryDataset
from train_koopman.losses import bi_lipschitz_loss, build_loss_fns, compute_loss


def train(model, trajectories, train_cfg: TrainKoopmanCfg):
    """Train a KoopmanAutoencoder on pre-collected ``trajectories``.

    Returns the trained model (mutated in place; best epoch's state_dict reloaded).
    """
    device = next(model.parameters()).device
    torch.manual_seed(train_cfg.seed)

    dataset = TrajectoryDataset(trajectories, train_cfg.horizon)
    print(f"Dataset size: {len(dataset)} windows")
    num_workers = train_cfg.num_workers
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        print(f"DataLoader: {num_workers} workers (persistent)")

    if train_cfg.riemannian_optimizer:
        import geoopt

        optimizer = geoopt.optim.RiemannianAdam(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )
        print("Using RiemannianAdam optimizer")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
        )

    if train_cfg.cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg.cosine_T_max, eta_min=train_cfg.cosine_eta_min
        )
        print(
            f"Using CosineAnnealingLR scheduler "
            f"(T_max={train_cfg.cosine_T_max}, eta_min={train_cfg.cosine_eta_min})"
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=train_cfg.scheduler_step, gamma=train_cfg.scheduler_gamma
        )
        print(
            f"Using StepLR scheduler "
            f"(step_size={train_cfg.scheduler_step}, gamma={train_cfg.scheduler_gamma})"
        )

    if train_cfg.torch_compile:
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Reconstruction pretraining (encoder/decoder only).
    recon_pretrain_epochs = train_cfg.recon_pretrain_epochs
    has_learned_encoder = hasattr(model, "encoder") and model.encoder is not None
    if recon_pretrain_epochs > 0 and not has_learned_encoder:
        print("Skipping reconstruction pretraining (encoder is fixed, no learnable params)")
        recon_pretrain_epochs = 0
    if recon_pretrain_epochs > 0:
        pretrain_lr = train_cfg.recon_pretrain_lr
        pretrain_bilip = train_cfg.recon_pretrain_bilip
        pretrain_bilip_w = train_cfg.recon_pretrain_bilip_weight
        pretrain_bilip_m = train_cfg.losses.bilip_m_target
        print(
            f"\n--- Reconstruction Pretraining "
            f"({recon_pretrain_epochs} epochs, lr={pretrain_lr}) ---"
        )
        if pretrain_bilip:
            print(f"  BiLip loss enabled (weight={pretrain_bilip_w}, m_target={pretrain_bilip_m})")
        pretrain_optimizer = torch.optim.Adam(
            model.parameters(), lr=pretrain_lr, weight_decay=train_cfg.weight_decay
        )
        for epoch in range(1, recon_pretrain_epochs + 1):
            model.train()
            epoch_recon = 0.0
            epoch_bilip = 0.0
            n_batches = 0
            for states_seq, actions_seq in loader:
                states_seq = states_seq.to(device)
                B, Hp1, S = states_seq.shape
                all_states = states_seq.reshape(B * Hp1, S)
                all_z = model.encode(all_states)
                all_recon = model.decode(all_z)
                loss = F.mse_loss(all_recon, all_states)
                epoch_recon += loss.item()

                if pretrain_bilip:
                    bilip = bi_lipschitz_loss(
                        model.encode, all_states.detach(), m_target=pretrain_bilip_m
                    )
                    loss = loss + pretrain_bilip_w * bilip
                    epoch_bilip += bilip.item()

                pretrain_optimizer.zero_grad()
                loss.backward()
                pretrain_optimizer.step()
                n_batches += 1

            if epoch % train_cfg.log_interval == 0 or epoch == 1:
                parts = [f"  Pretrain Epoch {epoch:4d} | Recon: {epoch_recon / n_batches:.6f}"]
                if pretrain_bilip:
                    parts.append(f"BiLip: {epoch_bilip / n_batches:.6f}")
                print(" | ".join(parts))
        del pretrain_optimizer
        print("--- Pretraining complete ---\n")

    loss_fns = build_loss_fns(model, train_cfg)
    active_names = list(loss_fns.keys())
    print(
        f"Active losses: "
        f"{', '.join(f'{n} (w={loss_fns[n][1]})' for n in active_names)}"
    )

    grad_clip_enabled = train_cfg.grad_clip
    grad_clip_type = train_cfg.grad_clip_type
    grad_clip_value = train_cfg.grad_clip_value
    if grad_clip_enabled:
        print(f"Gradient clipping: {grad_clip_type}, max={grad_clip_value}")

    loss_threshold = train_cfg.loss_threshold
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_components = {name: float("nan") for name in active_names}
    print("(Press Ctrl+C to end training early and continue pipeline)")

    epoch = 0
    n_batches = 0
    epoch_total = 0.0
    epoch_accum = {name: 0.0 for name in active_names}
    try:
        for epoch in range(1, train_cfg.num_epochs + 1):
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
                if grad_clip_enabled:
                    if grad_clip_type == "norm":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                    elif grad_clip_type == "value":
                        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                optimizer.step()

                if hasattr(model.K_module, "project"):
                    model.K_module.project()

                epoch_total += total_loss.item()
                for name in active_names:
                    epoch_accum[name] += losses[name].item()
                n_batches += 1

            scheduler.step()

            avg_total = epoch_total / n_batches
            if avg_total < best_loss:
                best_loss = avg_total
                best_state_dict = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_components = {
                    name: epoch_accum[name] / n_batches for name in active_names
                }

            if epoch % train_cfg.log_interval == 0 or epoch == 1:
                parts = [f"Epoch {epoch:4d} | Total: {avg_total:.6f}"]
                for name in active_names:
                    parts.append(f"{name}: {epoch_accum[name] / n_batches:.6f}")
                parts.append(f"LR: {scheduler.get_last_lr()[0]:.2e}")
                print(" | ".join(parts))

            if avg_total < loss_threshold:
                print(f"Early stop: total loss {avg_total:.6f} < threshold {loss_threshold}")
                break
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}. Continuing with best model...")

    model.load_state_dict(best_state_dict)

    print("\n--- Training Complete ---")
    print(f"  Best total: {best_loss:.6f} (epoch {best_epoch})")
    for name in active_names:
        print(f"  {name:>6s}: {best_components[name]:.6f}")
    print(f"  Epochs: {epoch}/{train_cfg.num_epochs}")

    training_stats = {
        "best_loss": float(best_loss),
        "best_epoch": int(best_epoch),
        "num_epochs_completed": int(epoch),
        "num_epochs_planned": int(train_cfg.num_epochs),
        "active_losses": list(active_names),
        "best_components": {name: float(v) for name, v in best_components.items()},
    }
    return model, training_stats
