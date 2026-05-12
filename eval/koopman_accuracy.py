"""Koopman model accuracy evaluation: env-agnostic prediction error statistics.

``evaluate_model_stats`` returns per-step latent and state-space prediction
errors (max / mean / std) over a list of trajectories — no environment
knowledge required. ``evaluate_model`` keeps the legacy signature by
producing a stats triple plus an optional figure delegated to the env's
registered :func:`EnvScorer.make_koopman_heatmap` hook.
"""

from __future__ import annotations

from typing import Optional

import torch


def evaluate_model_stats(model, trajectories, eval_horizon: int) -> dict:
    """Per-step latent/state prediction error stats over ``trajectories``.

    Returns ``max/mean/std`` of one-step prediction error in both spaces.
    """
    device = next(model.parameters()).device
    model.eval()

    max_latent = 0.0
    max_state = 0.0
    all_latent = []
    all_state = []
    with torch.no_grad():
        for states, actions in trajectories:
            states_t = torch.from_numpy(states).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            T_act = len(actions)
            z_all = model.encode(states_t[:T_act])
            z_next_all = model.encode(states_t[1 : T_act + 1])
            z_pred_all = model.predict(z_all, actions_t[:T_act])
            latent_errs = torch.linalg.norm(z_next_all - z_pred_all, dim=-1)
            max_latent = max(max_latent, latent_errs.max().item())
            all_latent.append(latent_errs.cpu())
            x_pred_all = model.decode(z_pred_all)
            state_errs = torch.linalg.norm(x_pred_all - states_t[1 : T_act + 1], dim=-1)
            max_state = max(max_state, state_errs.max().item())
            all_state.append(state_errs.cpu())

    all_latent = torch.cat(all_latent)
    all_state = torch.cat(all_state)
    stats = {
        "max_pred_error_latent": max_latent,
        "mean_pred_error_latent": all_latent.mean().item(),
        "std_pred_error_latent": all_latent.std().item(),
        "max_pred_error_state": max_state,
        "mean_pred_error_state": all_state.mean().item(),
        "std_pred_error_state": all_state.std().item(),
    }
    print(
        f"One-step prediction error (latent): max={stats['max_pred_error_latent']:.6f}  "
        f"mean={stats['mean_pred_error_latent']:.6f}  std={stats['std_pred_error_latent']:.6f}"
    )
    print(
        f"One-step prediction error (state):  max={stats['max_pred_error_state']:.6f}  "
        f"mean={stats['mean_pred_error_state']:.6f}  std={stats['std_pred_error_state']:.6f}"
    )
    return stats


def evaluate_model(
    model,
    trajectories,
    train_horizon: int,
    eval_horizon: int = 25,
    *,
    env_name: Optional[str] = None,
    title: Optional[str] = None,
    obs_scale=None,
    obs_type=None,
):
    """Generic stats + optional env-specific heatmap via the registered scorer.

    Returns ``(fig_or_None, stats_dict, heatmap_data_or_None)``. The figure
    and heatmap are only produced when the env registered a
    ``make_koopman_heatmap`` hook.
    """
    stats = evaluate_model_stats(model, trajectories, eval_horizon=eval_horizon)

    fig, heatmap_data = None, None
    if env_name is not None:
        from eval import ENV_SCORERS

        scorer = ENV_SCORERS.get(env_name)
        if scorer is not None and scorer.make_koopman_heatmap is not None:
            fig, heatmap_data = scorer.make_koopman_heatmap(
                model,
                trajectories,
                train_horizon=train_horizon,
                eval_horizon=eval_horizon,
                title=title,
                obs_scale=obs_scale,
                obs_type=obs_type or "cos_sin",
            )
    return fig, stats, heatmap_data
