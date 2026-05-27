"""CLI: ``python -m eval --config <yaml>``.

Loads the Koopman model + optional LQR controller + optional residual policy
and runs the eval modes implied by which artifacts are configured (see
``config/manager/eval_cfg.py``). Always optionally runs the Koopman accuracy
heatmap when ``eval_koopman_accuracy=True``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from config.manager import ConfigManager, EvalCfg
from data.dataloader import load_dataset
from policy import make_policy
from train_koopman.checkpointing import load_koopman_experiment, make_device


def _resolve_controller_dir(cfg: EvalCfg) -> Path | None:
    if not cfg.lqr_output_name:
        return None
    return Path("results") / cfg.koopman_experiment_name / "lqr" / cfg.lqr_output_name


def _load_lqr(controller_dir: Path) -> tuple["object", float]:
    """Return ``(lqr_obj_with_F, gamma_max)`` from ``controller_dir``."""
    lqr_pt = controller_dir / "lqr.pt"
    perf_yaml = controller_dir / "ctrl_performance.yaml"
    if not lqr_pt.is_file():
        raise FileNotFoundError(f"Expected LQR weights at {lqr_pt}")
    if not perf_yaml.is_file():
        raise FileNotFoundError(
            f"Expected ctrl_performance.yaml at {perf_yaml} (needed for gamma_max)"
        )
    raw = torch.load(lqr_pt, map_location="cpu")

    class _LoadedLQR:
        def __init__(self, F):
            self.F = F
            self.gain_norm = torch.linalg.matrix_norm(F, ord=2)

    # full_load tolerates legacy ``!!python/complex`` eigvals embedded in
    # controllability_fit.open_loop_eigvals by older training runs.
    perf = yaml.full_load(perf_yaml.read_text())
    bound = perf.get("bound") or {}
    if "gamma_max" not in bound:
        raise KeyError(
            f"'bound.gamma_max' not found in {perf_yaml}; was stability "
            "analysis run for this controller?"
        )
    return _LoadedLQR(raw["F"]), float(bound["gamma_max"])


def _load_residual_actor(
    cfg: EvalCfg,
    *,
    obs_space,
    act_space,
    device: torch.device,
) -> tuple["object", dict]:
    """Return ``(actor, train_cfg_dict)`` for the residual experiment.

    ``cfg.residual_train_cfg_path`` is the source of truth for the actor
    architecture and the wrapper kwargs the actor expects. The ``.pt`` file
    is loaded from ``train_residual/weights/<residual_experiment_name>/
    residual_train/best.pt``.
    """
    from models.residual_policy import StochasticActor

    if not cfg.residual_train_cfg_path:
        raise ValueError(
            "EvalCfg.residual_train_cfg_path must be set when "
            "residual_experiment_name is set (needed to recover actor architecture "
            "and wrapper kwargs)."
        )

    train_cfg = ConfigManager.load_stage(cfg.residual_train_cfg_path, "train_residual_cfg")
    weights_path = (
        Path("train_residual")
        / "weights"
        / cfg.residual_experiment_name
        / "residual_train"
        / "best.pt"
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Expected residual weights at {weights_path}")

    actor = StochasticActor(
        observation_space=obs_space,
        action_space=act_space,
        device=device,
        hidden_size=train_cfg.actor_hidden_size,
        hidden_layers=train_cfg.actor_hidden_layers,
    )
    state_dict = torch.load(weights_path, map_location=device)
    actor.load_state_dict(state_dict)
    return actor, {
        "obs_augmentation": train_cfg.obs_augmentation,
        "pred_error_space": train_cfg.pred_error_space,
        "z_ref_max_mode": train_cfg.z_ref_max_mode,
        "gamma_worst_case": float(train_cfg.gamma_worst_case),
    }


def _peek_residual_obs_act_space(
    *,
    eval_cfg: EvalCfg,
    env_name: str,
    env_kwargs: dict,
    koopman_model,
    lqr,
    gamma_max: float,
    pred_error_space: str,
    z_ref_max_mode: str,
    gamma_worst_case: float,
    obs_augmentation: str,
    device: torch.device,
    obs_scale=None,
    act_scale=None,
    aug_cfg=None,
):
    """Build a 1-env residual wrapper just to read its (obs, action) spaces.

    We need these to instantiate ``StochasticActor`` before running rollouts.
    The throwaway env is closed immediately.
    """
    from eval.rollout import build_residual_eval_env

    one_env = build_residual_eval_env(
        mode="residual",
        make_single_env_fn=lambda: _make_single_env(env_name, env_kwargs),
        num_envs=1,
        koopman_model=koopman_model,
        lqr=lqr,
        gamma_max=gamma_max,
        device=device,
        env_name=env_name,
        pred_error_space=pred_error_space,
        z_ref_max_mode=z_ref_max_mode,
        gamma_worst_case=gamma_worst_case,
        obs_augmentation_override=obs_augmentation,
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
    )
    try:
        return one_env.single_observation_space, one_env.single_action_space
    finally:
        one_env.close()


def _make_single_env(env_name, env_kwargs):
    from data.env_builder import make_single_env
    return make_single_env(env_name=env_name, env_kwargs=env_kwargs)


def run(cfg: EvalCfg) -> str:
    device = make_device()

    # All controller-dependent eval results land under the LQR controller dir
    # (one folder per mode). The koopman dir is only used as the fallback
    # output for the pre-LQR (gather/koopman-only) context and as the home of
    # the koopman-accuracy heatmap.
    koopman_dir = Path("results") / cfg.koopman_experiment_name
    koopman_dir.mkdir(parents=True, exist_ok=True)

    model, train_cfg, _state_dim, _action_dim = load_koopman_experiment(
        cfg.koopman_experiment_name, device
    )
    ds = load_dataset(train_cfg.dataset_name)
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = gather_snap["env_name"]
    env_kwargs = gather_snap.get("env_kwargs", {}) or {}

    base_policy_snap = gather_snap["base_policy"]
    base_policy = make_policy(
        base_policy_snap["name"], **(base_policy_snap.get("params", {}) or {})
    )

    # ---- Koopman accuracy (independent of LQR / residual) ----
    if cfg.eval_koopman_accuracy:
        _run_koopman_accuracy(
            out_dir=koopman_dir,
            model=model,
            ds=ds,
            train_cfg=train_cfg,
            env_name=env_name,
            env_kwargs=env_kwargs,
        )

    if not cfg.eval_policy_rollout:
        return str(koopman_dir)

    # ---- Multi-mode policy rollout ----
    # Compute the same obs / action scales the koopman model was trained with
    # so the wrapper feeds the encoder + predict step normalized inputs
    # (matching controller.controller_analysis.count_steps_under_threshold).
    from data.augmentation import compute_act_scale, compute_obs_scale

    aug_cfg = train_cfg.augmentation
    obs_scale = compute_obs_scale(aug_cfg, ds)
    act_scale = compute_act_scale(aug_cfg, ds)

    controller_dir = _resolve_controller_dir(cfg)
    lqr_obj = None
    gamma_max: float | None = None
    lqr_dir: Path | None = None
    residual_actor = None
    residual_kwargs: dict = {}

    if controller_dir is not None and controller_dir.is_dir():
        lqr_obj, gamma_max = _load_lqr(controller_dir)
        lqr_dir = controller_dir

    if lqr_obj is not None and cfg.residual_experiment_name:
        # Need to peek the wrapper-augmented obs/act space for the actor.
        # First need the residual training kwargs (loaded inside the helper).
        from config.manager import ConfigManager as _CM
        if not cfg.residual_train_cfg_path:
            raise ValueError(
                "EvalCfg.residual_train_cfg_path must be set when "
                "residual_experiment_name is set."
            )
        train_residual_cfg = _CM.load_stage(cfg.residual_train_cfg_path, "train_residual_cfg")
        residual_kwargs = {
            "obs_augmentation": train_residual_cfg.obs_augmentation,
            "pred_error_space": train_residual_cfg.pred_error_space,
            "z_ref_max_mode": train_residual_cfg.z_ref_max_mode,
            "gamma_worst_case": float(train_residual_cfg.gamma_worst_case),
        }
        obs_space, act_space = _peek_residual_obs_act_space(
            eval_cfg=cfg,
            env_name=env_name,
            env_kwargs=env_kwargs,
            koopman_model=model,
            lqr=lqr_obj,
            gamma_max=gamma_max,
            pred_error_space=residual_kwargs["pred_error_space"],
            z_ref_max_mode=residual_kwargs["z_ref_max_mode"],
            gamma_worst_case=residual_kwargs["gamma_worst_case"],
            obs_augmentation=residual_kwargs["obs_augmentation"],
            device=device,
            obs_scale=obs_scale,
            act_scale=act_scale,
            aug_cfg=aug_cfg,
        )
        residual_actor, _ = _load_residual_actor(
            cfg, obs_space=obs_space, act_space=act_space, device=device
        )

    from eval.multi_mode import run_multi_mode

    run_multi_mode(
        koopman_dir=koopman_dir,
        lqr_dir=lqr_dir,
        eval_cfg=cfg,
        env_name=env_name,
        env_kwargs=env_kwargs,
        base_policy=base_policy,
        koopman_model=model,
        device=device,
        lqr=lqr_obj,
        gamma_max=gamma_max,
        residual_actor=residual_actor,
        residual_obs_augmentation=residual_kwargs.get("obs_augmentation", "both"),
        residual_pred_error_space=residual_kwargs.get("pred_error_space", "latent"),
        residual_z_ref_max_mode=residual_kwargs.get("z_ref_max_mode", "action_bound"),
        residual_gamma_worst_case=residual_kwargs.get("gamma_worst_case", 0.0),
        obs_scale=obs_scale,
        act_scale=act_scale,
        aug_cfg=aug_cfg,
    )
    return str(koopman_dir)


def _run_koopman_accuracy(
    *, out_dir: Path, model, ds, train_cfg, env_name, env_kwargs
) -> None:
    from data.augmentation import augment_trajectories, compute_act_scale, compute_obs_scale
    from eval.koopman_accuracy import evaluate_model

    aug_cfg = train_cfg.augmentation
    aug_trajectories = augment_trajectories(ds, aug_cfg)
    obs_scale = compute_obs_scale(aug_cfg, ds)
    act_scale = compute_act_scale(aug_cfg, ds)
    koopman_obs_scale = (
        np.concatenate([obs_scale, act_scale])
        if aug_cfg.prepend_base_action and obs_scale is not None and act_scale is not None
        else obs_scale
    )
    fig, error_stats, heatmap_data = evaluate_model(
        model,
        aug_trajectories,
        train_horizon=train_cfg.horizon,
        eval_horizon=train_cfg.horizon + 1,
        env_name=env_name,
        obs_scale=koopman_obs_scale.tolist() if koopman_obs_scale is not None else None,
        obs_type=env_kwargs.get("obs_type", "cos_sin"),
    )
    if fig is not None:
        fig.savefig(out_dir / "koopman_prediction_error.png", dpi=150, bbox_inches="tight")
    with (out_dir / "koopman_eval_stats.yaml").open("w") as f:
        yaml.dump(
            {**error_stats, "heatmap": heatmap_data},
            f,
            default_flow_style=False,
            sort_keys=False,
        )
    print(f"Koopman accuracy saved to {out_dir}/koopman_eval_stats.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stable_koop artifacts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "eval_cfg")
    run(cfg)


if __name__ == "__main__":
    main()
