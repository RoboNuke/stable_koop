"""Orchestrator: run base / LQR / residual eval modes and write results.

``run_multi_mode`` is the single entry point shared by the CLI
(``python -m eval``), ``controller.lqr`` (post-fit), and
``train_residual`` (post-train). It decides which modes apply based on
which artifacts the caller passes in:

* ``lqr is None``                            → base only (gather/koopman ctx)
* ``lqr`` given, ``residual_policy is None`` → base + LQR
* both given                                 → base + LQR + residual

For every mode it builds the appropriate residual-wrapper env stack via
:func:`eval.rollout.build_residual_eval_env`, vectorizes the policy, and
calls :func:`eval.rollout.do_rollout`. Results land in:

    out_dir/
      {mode}_eval_stats.yaml      # task metrics + gamma summary
      {mode}_eval_traj.npz        # per-trajectory states/actions/gamma
      gamma_metrics_summary.yaml  # base/lqr/residual side-by-side
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
import yaml

from eval.gamma_metrics import compare_modes, summarize_gammas
from eval.rollout import (
    build_residual_eval_env,
    concat_valid_per_step,
    do_rollout,
    save_trajectories_npz,
    success_cfg_from_eval_cfg,
)


# ----------------------------------------------------------------------
# policy adapters
# ----------------------------------------------------------------------

def _vectorize_single_step(policy, num_envs: int) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a 1-at-a-time hand policy into a batched callable."""
    def batch(obs_batch: np.ndarray) -> np.ndarray:
        out = np.stack(
            [np.asarray(policy(obs_batch[i]), dtype=np.float32) for i in range(num_envs)]
        )
        return out.reshape(num_envs, -1)
    return batch


def _zero_z_ref_policy(latent_dim: int) -> Callable[[np.ndarray], np.ndarray]:
    """LQR-mode policy: residual is fixed at zero so the wrapper applies ``u = F @ z_t``."""
    def batch(obs_batch: np.ndarray) -> np.ndarray:
        return np.zeros((obs_batch.shape[0], latent_dim), dtype=np.float32)
    return batch


def _residual_actor_policy(actor, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a ``StochasticActor`` for batched numpy inference."""
    actor.eval()

    def batch(obs_batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
            z_refs = actor.act({"states": obs_t})[0]
        return z_refs.detach().cpu().numpy().astype(np.float32)
    return batch


# ----------------------------------------------------------------------
# per-mode runner
# ----------------------------------------------------------------------

def _run_mode(
    mode: str,
    *,
    out_dir: Path,
    eval_cfg,
    make_single_env_fn,
    env_name: str,
    koopman_model,
    lqr,
    gamma_max: float,
    device: torch.device,
    policy_batch: Callable[[np.ndarray], np.ndarray],
    pred_error_space: str,
    z_ref_max_mode: str,
    obs_augmentation_override: Optional[str],
    gamma_worst_case: float,
) -> dict:
    print(f"\n=== Eval mode: {mode} ===")
    env = build_residual_eval_env(
        mode=mode,
        make_single_env_fn=make_single_env_fn,
        num_envs=eval_cfg.num_parallel_evals,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        gamma_worst_case=gamma_worst_case,
        obs_augmentation_override=obs_augmentation_override,
    )
    try:
        results, per_trajectory = do_rollout(
            env,
            policy_batch,
            num_trajectories=eval_cfg.eval_num_trajectories,
            max_steps=eval_cfg.eval_max_steps,
            env_name=env_name,
            success_cfg=success_cfg_from_eval_cfg(eval_cfg),
            capture_per_step=bool(getattr(eval_cfg, "capture_per_step", True)),
        )
        wrapper_state = _wrapper_state_snapshot(env, mode=mode)
    finally:
        env.close()

    flat = concat_valid_per_step(per_trajectory, ("gamma_t", "eta_t", "stability_term"))
    gamma_summary = summarize_gammas(
        gamma=flat["gamma_t"],
        eta=flat["eta_t"],
        stability_term=flat["stability_term"],
        gamma_max=gamma_max,
    )
    applied_action_summary = _applied_action_summary(per_trajectory)

    stats_path = out_dir / f"{mode}_eval_stats.yaml"
    with stats_path.open("w") as f:
        yaml.dump(
            {
                "task_metrics": results,
                "gamma_metrics": gamma_summary,
                "applied_action": applied_action_summary,
                "wrapper_state": wrapper_state,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    save_trajectories_npz(out_dir / f"{mode}_eval_traj.npz", per_trajectory)
    _print_gamma_summary(mode, gamma_summary)
    print(f"  {mode} stats saved to {stats_path}")
    # Fold wrapper_state + applied_action into the gamma summary so the
    # top-level summary collates everything per mode.
    return {**gamma_summary, "applied_action": applied_action_summary,
            "wrapper_state": wrapper_state}


def _print_gamma_summary(mode: str, g: dict) -> None:
    print(
        f"  [gamma] mode={mode}  "
        f"violation_rate={g['violation_rate']:.3f}  "
        f"mean_normalized_gamma={g['mean_normalized_gamma']:.4f}  "
        f"mean_gamma_reward={g['mean_gamma_reward']:.4f}  "
        f"max_gamma={g['max_gamma']:.4f}  "
        f"p95_gamma={g['p95_gamma']:.4f}  "
        f"mean_eta={g['mean_eta']:.4f}  "
        f"(gamma_max={g['gamma_max']:.4f}, valid_steps={g['num_valid_steps']})"
    )


def _wrapper_state_snapshot(env, *, mode: str) -> dict:
    """Pull constants from the ``ResidualPolicyEnv`` after the rollout.

    ``z_ref_max`` only carries semantic weight when the wrapper actually
    applies the LQR transform (``lqr`` / ``residual`` modes). For ``base``
    mode the wrapper is in passthrough, so we omit it.
    """
    tensor_env = getattr(env, "tensor_env", None)
    if tensor_env is None:
        return {}
    snap = {
        "F_norm": float(tensor_env.F_norm),
        "B_norm": float(tensor_env.B_norm),
        "pred_error_space": str(tensor_env.pred_error_space),
        "obs_augmentation": str(tensor_env.obs_augmentation),
        "disable_action_augmentation": bool(tensor_env.disable_action_augmentation),
    }
    if mode != "base":
        snap["z_ref_max"] = float(tensor_env.z_ref_max)
    return snap


def _applied_action_summary(per_trajectory: list[dict]) -> dict:
    """Per-step ``applied_action`` summary stats (across all valid steps)."""
    arrs = [t["applied_action"] for t in per_trajectory
            if "applied_action" in t and t["applied_action"].size > 0]
    if not arrs:
        return {"num_steps": 0}
    stacked = np.concatenate(arrs, axis=0)  # (total_steps, action_dim)
    abs_stacked = np.abs(stacked)
    norms = np.linalg.norm(stacked, axis=-1)
    return {
        "num_steps": int(stacked.shape[0]),
        "action_dim": int(stacked.shape[1]),
        "mean_abs": [float(x) for x in abs_stacked.mean(axis=0)],
        "max_abs": [float(x) for x in abs_stacked.max(axis=0)],
        "mean_norm": float(norms.mean()),
        "max_norm": float(norms.max()),
    }


# ----------------------------------------------------------------------
# entry
# ----------------------------------------------------------------------

def run_multi_mode(
    *,
    out_dir: str | Path,
    eval_cfg,
    env_name: str,
    env_kwargs: dict,
    base_policy,
    koopman_model,
    device: torch.device,
    lqr=None,
    gamma_max: Optional[float] = None,
    residual_actor=None,
    residual_obs_augmentation: str = "both",
    residual_pred_error_space: str = "latent",
    residual_z_ref_max_mode: str = "action_bound",
    residual_gamma_worst_case: float = 0.0,
) -> dict:
    """Run the three eval modes that apply and write all output files.

    ``base_policy`` is a single-step callable ``(obs) -> action``.
    ``residual_actor`` is the loaded ``StochasticActor`` (with state_dict
    already applied) or ``None``.

    Returns the gamma metrics summary dict (also written to
    ``gamma_metrics_summary.yaml`` on disk).
    """
    from data.env_builder import make_single_env

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_single():
        return make_single_env(env_name=env_name, env_kwargs=env_kwargs)

    base_policy_name = getattr(base_policy, "__class__", type(base_policy)).__name__
    print(
        "\n[multi_mode] env={env}  base_policy={pol}  "
        "gamma_max={gm}  residual={res}  out_dir={out}".format(
            env=env_name,
            pol=base_policy_name,
            gm=("none" if gamma_max is None else f"{gamma_max:.6g}"),
            res=("loaded" if residual_actor is not None else "none"),
            out=str(out_dir),
        )
    )

    per_mode_gamma: dict[str, dict] = {}

    # ---- base mode ----
    # Requires LQR to instantiate the wrapper (gamma_max is a non-optional
    # field). In the pre-LQR gather context we just run the base policy on
    # a plain vec env without gamma capture.
    if lqr is None or gamma_max is None:
        _run_base_only(
            out_dir=out_dir,
            eval_cfg=eval_cfg,
            env_name=env_name,
            env_kwargs=env_kwargs,
            base_policy=base_policy,
        )
        return {}

    base_batch = _vectorize_single_step(base_policy, eval_cfg.num_parallel_evals)
    per_mode_gamma["base"] = _run_mode(
        "base",
        out_dir=out_dir,
        eval_cfg=eval_cfg,
        make_single_env_fn=make_single,
        env_name=env_name,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        policy_batch=base_batch,
        pred_error_space=residual_pred_error_space,
        z_ref_max_mode=residual_z_ref_max_mode,
        obs_augmentation_override=None,
        gamma_worst_case=residual_gamma_worst_case,
    )

    # ---- lqr mode ----
    latent_dim = int(koopman_model.encoder_latent_dim) + int(koopman_model.prepend_dim)
    per_mode_gamma["lqr"] = _run_mode(
        "lqr",
        out_dir=out_dir,
        eval_cfg=eval_cfg,
        make_single_env_fn=make_single,
        env_name=env_name,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        policy_batch=_zero_z_ref_policy(latent_dim),
        pred_error_space=residual_pred_error_space,
        z_ref_max_mode=residual_z_ref_max_mode,
        obs_augmentation_override=None,
        gamma_worst_case=residual_gamma_worst_case,
    )

    # ---- residual mode ----
    if residual_actor is not None:
        per_mode_gamma["residual"] = _run_mode(
            "residual",
            out_dir=out_dir,
            eval_cfg=eval_cfg,
            make_single_env_fn=make_single,
            env_name=env_name,
            koopman_model=koopman_model,
            lqr=lqr,
            gamma_max=gamma_max,
            device=device,
            policy_batch=_residual_actor_policy(residual_actor, device),
            pred_error_space=residual_pred_error_space,
            z_ref_max_mode=residual_z_ref_max_mode,
            obs_augmentation_override=residual_obs_augmentation,
            gamma_worst_case=residual_gamma_worst_case,
        )

    # ---- summary ----
    summary = compare_modes(per_mode_gamma)
    summary_path = out_dir / "gamma_metrics_summary.yaml"
    with summary_path.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"\nGamma metrics summary saved to {summary_path}")
    return summary


def _run_base_only(
    *,
    out_dir: Path,
    eval_cfg,
    env_name: str,
    env_kwargs: dict,
    base_policy,
) -> None:
    """Pre-LQR gather/koopman context: roll out the base policy on a plain
    ``SyncVectorEnv``. No gamma capture; writes only ``base_eval_stats.yaml``
    and ``base_eval_traj.npz``."""
    from data.env_builder import make_single_env

    def make_single():
        return make_single_env(env_name=env_name, env_kwargs=env_kwargs)

    env = gym.vector.SyncVectorEnv(
        [make_single for _ in range(eval_cfg.num_parallel_evals)]
    )
    try:
        batch = _vectorize_single_step(base_policy, eval_cfg.num_parallel_evals)
        results, per_trajectory = do_rollout(
            env,
            batch,
            num_trajectories=eval_cfg.eval_num_trajectories,
            max_steps=eval_cfg.eval_max_steps,
            env_name=env_name,
            success_cfg=success_cfg_from_eval_cfg(eval_cfg),
            capture_per_step=False,
        )
    finally:
        env.close()

    with (out_dir / "base_eval_stats.yaml").open("w") as f:
        yaml.dump({"task_metrics": results}, f, default_flow_style=False, sort_keys=False)
    save_trajectories_npz(out_dir / "base_eval_traj.npz", per_trajectory)
    print(f"\nBase-policy eval (no LQR / no gamma) saved to {out_dir}/base_eval_stats.yaml")
