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
calls :func:`eval.rollout.do_rollout`. Because every eval mode is
controller-dependent (gamma_t is meaningful only relative to that LQR's
``gamma_max``), results live under the LQR controller directory:

    lqr_dir/
      eval_base_only/{stats.yaml,traj.npz}
      eval_lqr/{stats.yaml,traj.npz}
      eval_residual/{stats.yaml,traj.npz}     # only when residual_actor given
      eval_gamma_metrics_summary.yaml              # base/lqr/residual side-by-side

The pre-LQR fallback (no controller available) writes only base-mode
results to ``koopman_dir/eval_base_only/``.
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

_MODE_SUBDIR = {
    "base": "eval_base_only",
    "lqr": "eval_lqr",
    "residual": "eval_residual",
}


def _run_mode(
    mode: str,
    *,
    parent_dir: Path,
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
    obs_scale=None,
    act_scale=None,
    aug_cfg=None,
) -> dict:
    out_dir = parent_dir / _MODE_SUBDIR[mode]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Eval mode: {mode} ===")
    env = build_residual_eval_env(
        mode=mode,
        make_single_env_fn=make_single_env_fn,
        num_envs=eval_cfg.num_parallel_evals,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        env_name=env_name,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        gamma_worst_case=gamma_worst_case,
        obs_augmentation_override=obs_augmentation_override,
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
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

    gamma_summary = summarize_gammas(
        per_trajectory=per_trajectory,
        gamma_max=gamma_max,
    )
    applied_action_summary = _applied_action_summary(per_trajectory)

    stats_path = out_dir / "stats.yaml"
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
    save_trajectories_npz(out_dir / "traj.npz", per_trajectory)
    _print_gamma_summary(mode, gamma_summary)
    print(f"  {mode} stats saved to {stats_path}")
    # Fold wrapper_state + applied_action into the gamma summary so the
    # top-level summary collates everything per mode.
    return {**gamma_summary, "applied_action": applied_action_summary,
            "wrapper_state": wrapper_state}


_GAMMA_MEANSTD_KEYS = (
    "violation_rate",
    "mean_pred_error",
    "avg_max_pred_error",
    "frac_used_cert",
    "gamma_reward",
    "mean_eta",
)
_GAMMA_SCALAR_KEYS = ("max_pred_error",)
_GAMMA_CELL_W = 22


def _print_gamma_summary(mode: str, summary: dict) -> None:
    """Print combined/success/failure gamma table.

    ``summary`` is the dict returned by ``summarize_gammas`` — three blocks
    (combined / success / failure), each with ``{mean, std}`` entries for
    the per-trajectory metrics and plain scalars for true-max / counts.
    """
    print(f"\nGamma metrics ({mode}, gamma_max={summary['combined']['gamma_max']:.4f}):")
    counts = (
        f"Trajectories: combined={summary['combined']['num_trajectories']}  "
        f"success={summary['success']['num_trajectories']}  "
        f"failure={summary['failure']['num_trajectories']}    "
        f"Total steps: combined={summary['combined']['total_steps']}  "
        f"success={summary['success']['total_steps']}  "
        f"failure={summary['failure']['total_steps']}"
    )
    print(counts)
    header = f"{'Metric':<24} {'Combined':>22} {'Success':>22} {'Failure':>22}"
    print(header)
    print("-" * len(header))
    for key in _GAMMA_MEANSTD_KEYS:
        cells = []
        for group in ("combined", "success", "failure"):
            ms = summary[group][key]
            if ms is None:
                cells.append("no bound (gamma_max<=0)".rjust(_GAMMA_CELL_W))
            elif isinstance(ms, dict):
                m = ms.get("mean", float("nan"))
                s = ms.get("std", float("nan"))
                cells.append(f"{m:10.5f} +/- {s:8.5f}")
            else:
                cells.append(f"{float(ms):10.5f}")
        print(f"{key:<24} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    for key in _GAMMA_SCALAR_KEYS:
        cells = [
            f"{float(summary[group][key]):10.5f}"
            for group in ("combined", "success", "failure")
        ]
        print(f"{key:<24} {cells[0]:>22} {cells[1]:>22} {cells[2]:>22}")
    print()


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
    koopman_dir: str | Path,
    lqr_dir: Optional[str | Path] = None,
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
    obs_scale=None,
    act_scale=None,
    aug_cfg=None,
) -> dict:
    """Run the three eval modes that apply and write all output files.

    ``base_policy`` is a single-step callable ``(obs) -> action``.
    ``residual_actor`` is the loaded ``StochasticActor`` (with state_dict
    already applied) or ``None``.

    Output layout:

    * When ``lqr`` is provided (the normal case) all results land under
      ``lqr_dir`` (required): per-mode subfolders ``eval_base_only/`` /
      ``eval_lqr/`` / ``eval_residual/`` each holding ``stats.yaml`` and
      ``traj.npz``, plus a top-level ``eval_gamma_metrics_summary.yaml``.
    * When ``lqr is None`` (pre-LQR gather/koopman context) the base
      policy is rolled out on a plain SyncVectorEnv and written to
      ``koopman_dir/eval_base_only/``.

    ``obs_scale`` / ``act_scale`` / ``aug_cfg`` are the koopman model's
    training-time normalization — required so the wrapper feeds the
    koopman model the same normalized inputs the LQR was fit against.

    Returns the gamma metrics summary dict (also written to
    ``eval_gamma_metrics_summary.yaml`` on disk in ``lqr_dir``).
    """
    from data.env_builder import make_single_env

    koopman_dir = Path(koopman_dir)
    koopman_dir.mkdir(parents=True, exist_ok=True)
    if lqr_dir is not None:
        lqr_dir = Path(lqr_dir)
        lqr_dir.mkdir(parents=True, exist_ok=True)

    def make_single():
        return make_single_env(env_name=env_name, env_kwargs=env_kwargs)

    # ``gamma_max`` from ``ctrl_performance.yaml`` is computed against
    # state-space prediction error when the koopman model uses prepended state
    # (the bound is α-bound branch in ``lqr_analysisv2``), and latent-space
    # otherwise. The eval has to use the same space or violation_rate is
    # incomparable with the LQR coverage report. Always derive from the model
    # for base/lqr; residual mode uses whatever the residual was trained with.
    bound_space = "state_decoded" if bool(getattr(koopman_model, "prepend_state", False)) else "latent"
    if residual_pred_error_space != bound_space:
        print(
            f"  [multi_mode] WARNING: residual_pred_error_space={residual_pred_error_space!r} "
            f"differs from gamma_max bound space {bound_space!r}; base/lqr modes will use "
            f"{bound_space!r} so their violation_rate is comparable to the LQR coverage report."
        )

    base_policy_name = getattr(base_policy, "__class__", type(base_policy)).__name__
    print(
        "\n[multi_mode] env={env}  base_policy={pol}  "
        "gamma_max={gm}  pred_error_space={sp}  residual={res}\n"
        "             koopman_dir={kd}\n"
        "             lqr_dir={ld}".format(
            env=env_name,
            pol=base_policy_name,
            gm=("none" if gamma_max is None else f"{gamma_max:.6g}"),
            sp=bound_space,
            res=("loaded" if residual_actor is not None else "none"),
            kd=str(koopman_dir),
            ld=("-" if lqr_dir is None else str(lqr_dir)),
        )
    )

    per_mode_gamma: dict[str, dict] = {}

    # ---- base mode ----
    # The wrapper needs ``gamma_max`` (it's a required field), so without an
    # LQR we fall back to a plain SyncVectorEnv + base policy and write to
    # ``koopman_dir/eval_base_only/``.
    if lqr is None or gamma_max is None:
        _run_base_only(
            koopman_dir=koopman_dir,
            eval_cfg=eval_cfg,
            env_name=env_name,
            env_kwargs=env_kwargs,
            base_policy=base_policy,
        )
        return {}

    if lqr_dir is None:
        raise ValueError(
            "run_multi_mode: lqr_dir is required when lqr is provided "
            "(it's where all eval results land)."
        )

    bound_common = dict(
        parent_dir=lqr_dir,
        eval_cfg=eval_cfg,
        make_single_env_fn=make_single,
        env_name=env_name,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        # base / lqr always use the bound's native space so they match the
        # LQR coverage report. residual mode below uses the training-time
        # value so the actor sees the gamma values it was reward-shaped on.
        pred_error_space=bound_space,
        z_ref_max_mode=residual_z_ref_max_mode,
        gamma_worst_case=residual_gamma_worst_case,
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
    )

    base_batch = _vectorize_single_step(base_policy, eval_cfg.num_parallel_evals)
    per_mode_gamma["base"] = _run_mode(
        "base",
        policy_batch=base_batch,
        obs_augmentation_override=None,
        **bound_common,
    )

    # ---- lqr mode ----
    latent_dim = int(koopman_model.encoder_latent_dim) + int(koopman_model.prepend_dim)
    per_mode_gamma["lqr"] = _run_mode(
        "lqr",
        policy_batch=_zero_z_ref_policy(latent_dim),
        obs_augmentation_override=None,
        **bound_common,
    )

    # ---- residual mode ----
    if residual_actor is not None:
        residual_common = {**bound_common, "pred_error_space": residual_pred_error_space}
        per_mode_gamma["residual"] = _run_mode(
            "residual",
            policy_batch=_residual_actor_policy(residual_actor, device),
            obs_augmentation_override=residual_obs_augmentation,
            **residual_common,
        )

    # ---- summary ----
    summary = compare_modes(per_mode_gamma)
    summary_path = lqr_dir / "eval_gamma_metrics_summary.yaml"
    with summary_path.open("w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    print(f"\nGamma metrics summary saved to {summary_path}")
    return summary


def _run_base_only(
    *,
    koopman_dir: Path,
    eval_cfg,
    env_name: str,
    env_kwargs: dict,
    base_policy,
) -> None:
    """Pre-LQR gather/koopman context: roll out the base policy on a plain
    ``SyncVectorEnv``. No gamma capture; writes only ``stats.yaml`` +
    ``traj.npz`` under ``koopman_dir/eval_base_only/``."""
    from data.env_builder import make_single_env

    out_dir = koopman_dir / _MODE_SUBDIR["base"]
    out_dir.mkdir(parents=True, exist_ok=True)

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

    with (out_dir / "stats.yaml").open("w") as f:
        yaml.dump({"task_metrics": results}, f, default_flow_style=False, sort_keys=False)
    save_trajectories_npz(out_dir / "traj.npz", per_trajectory)
    print(f"\nBase-policy eval (no LQR / no gamma) saved to {out_dir}/stats.yaml")
