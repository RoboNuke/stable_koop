"""CLI: ``python -m eval.gym_rollout_video --config <yaml>``.

**Gymnasium-only.** This script builds the env via ``gym.make(env_name,
render_mode="rgb_array")`` and pulls frames from ``env.render()``. It is
NOT compatible with Isaac Lab / Forge envs — those need a different env
builder and a camera-sensor frame source.

Runs one rollout of the trained LQR controller (with or without a residual
SAC policy) and writes a gif/mp4 with on-frame diagnostics:

* Cumulative env reward (text overlay).
* A vertical bar in the bottom-right whose fill height is
  ``gamma_t / gamma_max`` (visually saturating at ``video.bar_pct_cap``).
* A red border around the frame on any step where
  ``gamma_t > gamma_max``.

The env is always wrapped in :class:`wrappers.residual.ResidualPolicyEnv`
so that ``gamma_t`` and ``gamma_max`` come from a single source of truth.
In base-policy mode the wrapper is configured to pass actions through and
expose only the raw env obs (``disable_action_augmentation=True``,
``obs_augmentation="none"``, ``reward_weight=0.0``); the Koopman LQR is
applied externally as ``u = F · encode(obs)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym

from config.manager import ConfigManager, EvalCfg, TrainResidualCfg
from data.augmentation import compute_act_scale, compute_obs_scale
from data.dataloader import load_dataset
from data.env_builder import make_single_env
from models.residual_policy import StochasticActor
from policy import make_policy
from train_koopman.checkpointing import load_koopman_experiment, make_device
from wrappers.gym_vec_adapter import GymVectorAdapter
from wrappers.residual import ResidualPolicyEnv


# ----------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------


class _LoadedLQR:
    """Minimal LQR adapter exposing ``.F`` for :class:`ResidualPolicyEnv`."""

    def __init__(self, F: torch.Tensor):
        self.F = F


def _load_lqr_and_gamma(koopman_exp: str, lqr_output_name: str) -> tuple[_LoadedLQR, float]:
    ctrl_dir = Path("results") / koopman_exp / "lqr" / lqr_output_name
    lqr_pt = ctrl_dir / "lqr.pt"
    perf_yaml = ctrl_dir / "ctrl_performance.yaml"
    if not lqr_pt.is_file():
        raise FileNotFoundError(f"Expected LQR weights at {lqr_pt}")
    if not perf_yaml.is_file():
        raise FileNotFoundError(f"Expected {perf_yaml} (needed for gamma_max)")
    F = torch.load(lqr_pt, map_location="cpu")["F"]
    # full_load: ctrl_performance.yaml may carry legacy !!python/complex eigvals
    # that safe_load chokes on (see controller.lqr.__main__._read_perf_yaml).
    perf = yaml.full_load(perf_yaml.read_text())
    bound = perf.get("bound") or {}
    if "gamma_max" not in bound:
        raise KeyError(f"'bound.gamma_max' missing in {perf_yaml}")
    return _LoadedLQR(F), float(bound["gamma_max"])


def _load_residual_actor(
    residual_experiment_name: str,
    train_residual_cfg: TrainResidualCfg,
    obs_space,
    action_space,
    device: torch.device,
) -> StochasticActor:
    best_path = (
        Path("train_residual") / "weights" / residual_experiment_name
        / "residual_train" / "best.pt"
    )
    if not best_path.is_file():
        raise FileNotFoundError(f"Expected residual actor weights at {best_path}")
    actor = StochasticActor(
        obs_space, action_space, device,
        hidden_size=train_residual_cfg.actor_hidden_size,
        hidden_layers=train_residual_cfg.actor_hidden_layers,
    )
    actor.load_state_dict(torch.load(best_path, map_location=device))
    actor.eval()
    return actor


# ----------------------------------------------------------------------
# Overlay rendering
# ----------------------------------------------------------------------


def _lerp_color(pct: float) -> tuple[int, int, int]:
    """Green → yellow → red gradient on ``pct ∈ [0, 1+]`` (clipped at 1)."""
    pct = float(max(0.0, min(1.0, pct)))
    r = int(round(255 * min(1.0, 2.0 * pct)))
    g = int(round(255 * min(1.0, 2.0 * (1.0 - pct))))
    return (r, g, 0)


def _overlay(
    frame: np.ndarray,
    *,
    total_reward: float,
    gamma_t: float,
    gamma_max: float,
    bar_pct_cap: float,
    font: ImageFont.ImageFont,
) -> np.ndarray:
    """Composite reward text, γ bar, and red border onto an RGB frame."""
    img = Image.fromarray(frame).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # --- Reward text (top-left) on a translucent backdrop. ---
    text = f"reward: {total_reward:+.2f}"
    pad = 6
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    x0, y0 = 8, 8
    draw.rectangle(
        [x0, y0, x0 + tw + 2 * pad, y0 + th + 2 * pad],
        fill=(0, 0, 0, 160),
    )
    draw.text((x0 + pad, y0 + pad), text, fill=(255, 255, 255, 255), font=font)

    # --- γ-ratio bar (bottom-right). ---
    # A non-positive gamma_max means "no useful stability bound"; treat any
    # positive gamma_t as overflow so the bar saturates and the border lights up.
    if gamma_max > 0:
        pct_raw = gamma_t / gamma_max
    else:
        pct_raw = bar_pct_cap if gamma_t > 0 else 0.0
    pct_display = min(max(pct_raw, 0.0), bar_pct_cap) / bar_pct_cap
    bar_w = max(8, W // 40)
    bar_h = max(40, H // 4)
    margin = 12
    bar_x1 = W - margin
    bar_x0 = bar_x1 - bar_w
    bar_y1 = H - margin
    bar_y0 = bar_y1 - bar_h
    # Empty outline.
    draw.rectangle([bar_x0, bar_y0, bar_x1, bar_y1], outline=(255, 255, 255, 220), width=2)
    # Filled portion from bottom.
    fill_h = int(round(pct_display * bar_h))
    if fill_h > 0:
        color = _lerp_color(pct_raw)
        draw.rectangle([bar_x0 + 1, bar_y1 - fill_h, bar_x1 - 1, bar_y1 - 1], fill=color + (220,))
    # Label above the bar.
    label = "g/gmax"
    try:
        lbbox = draw.textbbox((0, 0), label, font=font)
        lw = lbbox[2] - lbbox[0]
        lh = lbbox[3] - lbbox[1]
    except AttributeError:
        lw, lh = draw.textsize(label, font=font)
    draw.text(
        (bar_x1 - lw, bar_y0 - lh - 4),
        label,
        fill=(255, 255, 255, 230),
        font=font,
    )

    # --- Red border on violation. ---
    if gamma_t > gamma_max:
        thick = max(4, min(W, H) // 60)
        for k in range(thick):
            draw.rectangle([k, k, W - 1 - k, H - 1 - k], outline=(255, 0, 0, 255))

    return np.asarray(img.convert("RGB"))


def _load_font() -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, 18)
            except OSError:
                pass
    return ImageFont.load_default()


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------


def _make_residual_policy(actor: StochasticActor):
    """Return a deterministic policy callable: ``obs_t (1, O) -> action_t (1, A)``.

    Operates entirely on the wrapper's device-tensor obs (no numpy round
    trips); ``tanh(mean)`` of the Gaussian for deterministic eval.
    """

    @torch.no_grad()
    def policy(obs_t: torch.Tensor) -> torch.Tensor:
        mean_actions, _log_std, _ = actor.compute({"states": obs_t}, role="policy")
        return torch.tanh(mean_actions)

    return policy


def _make_gather_base_policy(base_policy_callable, device: torch.device):
    """Adapt a numpy gather-snap base policy to ``(1, O) → (1, A)`` tensor form.

    Hand-designed base policies (registered in :mod:`policy`) take per-sample
    numpy obs and return numpy actions. This thin shim moves a single-sample
    batch in/out so it can drop into the same rollout loop as the tensor
    policies.
    """

    def policy(obs_t: torch.Tensor) -> torch.Tensor:
        obs_np = obs_t.detach().cpu().numpy()[0]  # (O,)
        action_np = np.asarray(base_policy_callable(obs_np), dtype=np.float32)
        return torch.as_tensor(action_np, device=device, dtype=torch.float32).unsqueeze(0)

    return policy


def _make_zero_residual_policy(latent_dim: int, device: torch.device):
    """Return a constant-zero policy in residual (latent) action space.

    Used for "pure Koopman LQR" mode: the wrapper is configured with
    ``disable_action_augmentation=False`` so it applies
    ``u = -F · (z_t - z_ref_base - z_ref_res)`` itself (with the env's
    registered ``base_goal_fn`` providing ``z_ref_base`` and the proper
    obs/act scaling). Passing ``z_ref_res = 0`` makes the policy a pure
    LQR drive toward ``x_base``. Mirrors the ``"lqr"`` mode in
    :mod:`eval.rollout` (``_MODE_KWARGS["lqr"]``).
    """
    zeros = torch.zeros((1, latent_dim), device=device, dtype=torch.float32)

    def policy(_obs_t: torch.Tensor) -> torch.Tensor:
        return zeros

    return policy


def _render_from_vec(vec_env: gym.vector.SyncVectorEnv) -> np.ndarray | None:
    """Pull an rgb_array frame from the first sub-env of a SyncVectorEnv.

    Routed through ``vec_env.envs[0]`` because :class:`ResidualPolicyEnv` /
    :class:`GymVectorAdapter` don't expose ``render``.
    """
    raw = vec_env.envs[0].render()
    return None if raw is None else np.asarray(raw)


def _rollout(env, policy, *, vec_env, max_steps: int, seed: int | None,
             gamma_max: float, bar_pct_cap: float, font) -> list[np.ndarray]:
    reset_kwargs = {} if seed is None else {"seed": int(seed)}
    obs, _ = env.reset(**reset_kwargs)  # obs: (1, O) tensor on device
    total_reward = 0.0
    gamma_t = 0.0
    frames: list[np.ndarray] = []

    # First frame (post-reset, no prior step → γ = 0).
    raw = _render_from_vec(vec_env)
    if raw is not None:
        frames.append(
            _overlay(
                raw,
                total_reward=total_reward,
                gamma_t=gamma_t,
                gamma_max=gamma_max,
                bar_pct_cap=bar_pct_cap,
                font=font,
            )
        )

    for _step in range(max_steps):
        action = policy(obs)  # (1, A) tensor
        obs, _aug_reward, terminated, truncated, info = env.step(action)
        # info["env_reward"] is (N, 1); info["gamma_t"] is (N,); single-env here.
        total_reward += float(info["env_reward"].sum().item())
        gamma_t = float(info["gamma_t"].sum().item())
        raw = _render_from_vec(vec_env)
        if raw is not None:
            frames.append(
                _overlay(
                    raw,
                    total_reward=total_reward,
                    gamma_t=gamma_t,
                    gamma_max=gamma_max,
                    bar_pct_cap=bar_pct_cap,
                    font=font,
                )
            )
        if bool(terminated.any().item()) or bool(truncated.any().item()):
            break

    return frames


def _write_outputs(
    frames: list[np.ndarray], out_dir: Path, *, stem: str, formats: list[str], fps: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        if fmt == "gif":
            path = out_dir / f"{stem}.gif"
            # imageio infers GIF duration from fps for "gif" extension.
            imageio.mimsave(path, frames, fps=fps, loop=0)
            written.append(path)
        elif fmt == "mp4":
            path = out_dir / f"{stem}.mp4"
            # libx264 requires even H/W; crop one pixel if needed.
            even_frames = [_ensure_even(f) for f in frames]
            imageio.mimwrite(path, even_frames, fps=fps, codec="libx264", quality=8)
            written.append(path)
        else:
            raise ValueError(f"Unknown video format {fmt!r} (expected 'gif' or 'mp4')")
    return written


def _ensure_even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    eh = h - (h % 2)
    ew = w - (w % 2)
    if eh == h and ew == w:
        return frame
    return frame[:eh, :ew]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def run(cfg: EvalCfg) -> str:
    if not cfg.video.enabled:
        print("[gym_rollout_video] video.enabled=False — nothing to do.")
        return ""
    if not cfg.video.formats:
        raise ValueError("video.formats is empty; expected at least one of 'gif', 'mp4'.")
    for fmt in cfg.video.formats:
        if fmt not in ("gif", "mp4"):
            raise ValueError(f"Unknown video format {fmt!r}; expected 'gif' or 'mp4'.")

    mode = cfg.video.mode
    if mode not in ("base_only", "lqr", "residual"):
        raise ValueError(
            f"video.mode must be 'base_only', 'lqr', or 'residual'; got {mode!r}."
        )
    if not cfg.lqr_output_name:
        raise ValueError(
            "eval_cfg.lqr_output_name is empty; required for all modes to locate "
            "lqr.pt + ctrl_performance.yaml (γmax + F drive the overlay)."
        )
    if mode == "residual":
        if not cfg.residual_experiment_name:
            raise ValueError(
                "video.mode='residual' but residual_experiment_name is unset."
            )
        if not cfg.residual_train_cfg_path:
            raise ValueError(
                "video.mode='residual' but residual_train_cfg_path is unset; "
                "needed to mirror the wrapper kwargs the actor was trained with."
            )

    device = make_device()
    # Per-mode output dirs all live under the LQR control folder, since
    # γmax (the overlay's reference) comes from this LQR fit even when the
    # rollout is driven by the gather-snap base policy or a residual actor.
    # Side-by-side ``videos_*`` subdirs keep modes separate without scattering
    # results across multiple experiment folders:
    #   * base_only → results/<koopman>/lqr/<lqr_output>/videos_base_only/
    #   * lqr       → results/<koopman>/lqr/<lqr_output>/videos_lqr/
    #   * residual  → results/<koopman>/lqr/<lqr_output>/videos_residual/
    ctrl_dir = (
        Path("results") / cfg.koopman_experiment_name
        / "lqr" / cfg.lqr_output_name
    )
    out_dir = ctrl_dir / f"videos_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gym_rollout_video] mode={mode!r}; outputs → {out_dir}")
    print(f"[gym_rollout_video] loading koopman experiment {cfg.koopman_experiment_name!r}")
    model, train_cfg, _state_dim, _action_dim = load_koopman_experiment(
        cfg.koopman_experiment_name, device
    )
    ds = load_dataset(train_cfg.dataset_name)
    gather_snap = yaml.safe_load(ds.config_yaml)["gather_data_cfg"]
    env_name = gather_snap["env_name"]
    env_kwargs = gather_snap.get("env_kwargs", {}) or {}

    # Normalization scales matching the koopman's training-time augmentation.
    # Required so the residual wrapper feeds the encoder + predict step the
    # same normalized inputs as training (and γ_t lands in the same space as
    # the LQR's coverage report). Mirrors train_residual/__main__.py.
    obs_scale_np = compute_obs_scale(train_cfg.augmentation, ds)
    act_scale_np = compute_act_scale(train_cfg.augmentation, ds)
    print(
        f"[gym_rollout_video] obs_scale_source={train_cfg.augmentation.obs_scale_source!r}, "
        f"act_scale_source={train_cfg.augmentation.act_scale_source!r}"
    )

    print(f"[gym_rollout_video] loading LQR + γmax from results/{cfg.koopman_experiment_name}/lqr/{cfg.lqr_output_name}/")
    lqr, gamma_max = _load_lqr_and_gamma(cfg.koopman_experiment_name, cfg.lqr_output_name)

    # Mirror eval.rollout._MODE_KWARGS so all three eval modes assemble the
    # wrapper the same way. residual mode replays the wrapper kwargs the
    # actor was trained with; the other two are pure visualization runs with
    # ``reward_weight = 0``.
    if mode == "residual":
        tr_cfg = ConfigManager.load_stage(cfg.residual_train_cfg_path, "train_residual_cfg")
        wrapper_kwargs = dict(
            gamma_max=gamma_max,
            gamma_worst_case=tr_cfg.gamma_worst_case,
            reward_weight=tr_cfg.reward_weight,
            pred_error_space=tr_cfg.pred_error_space,
            z_ref_max_mode=tr_cfg.z_ref_max_mode,
            obs_augmentation=tr_cfg.obs_augmentation,
            disable_action_augmentation=tr_cfg.disable_action_augmentation,
        )
    else:
        tr_cfg = None
        # base_only → passthrough (policy provides env-space action).
        # lqr       → wrapper applies u = -F·(z - z_ref_base) itself.
        disable_action_aug = (mode == "base_only")
        wrapper_kwargs = dict(
            gamma_max=gamma_max,
            gamma_worst_case=0.0,
            reward_weight=0.0,
            pred_error_space="latent",
            z_ref_max_mode="action_bound",
            obs_augmentation="none",
            disable_action_augmentation=disable_action_aug,
        )

    print(f"[gym_rollout_video] building env {env_name!r} (render_mode='rgb_array')")
    # Match train_residual.sac._make_residual_env: gym → SyncVectorEnv →
    # GymVectorAdapter → ResidualPolicyEnv. num_envs=1 since this is a single
    # rollout for visualization. Hold a handle on vec_env for render() access
    # (the adapter + residual wrapper don't expose render).
    def _make_raw_env():
        return make_single_env(env_name, render_mode="rgb_array", env_kwargs=env_kwargs)

    vec_env = gym.vector.SyncVectorEnv([_make_raw_env])
    adapter = GymVectorAdapter(vec_env, device=device)
    from data.env_builder import get_base_goal_fn
    env = ResidualPolicyEnv(
        adapter,
        koopman_model=model,
        lqr=lqr,
        device=device,
        base_goal_fn=get_base_goal_fn(env_name),
        obs_scale=obs_scale_np,
        act_scale=act_scale_np,
        aug_cfg=train_cfg.augmentation,
        **wrapper_kwargs,
    )

    if mode == "residual":
        print(f"[gym_rollout_video] loading residual actor for {cfg.residual_experiment_name!r}")
        actor = _load_residual_actor(
            cfg.residual_experiment_name, tr_cfg,
            env.observation_space, env.action_space, device,
        )
        policy = _make_residual_policy(actor)
    elif mode == "base_only":
        base_snap = gather_snap["base_policy"]
        params = base_snap.get("params", {}) or {}
        print(
            f"[gym_rollout_video] using gather-snap base policy {base_snap['name']!r} "
            f"with params {params}"
        )
        base_callable = make_policy(base_snap["name"], **params)
        policy = _make_gather_base_policy(base_callable, device)
    else:  # lqr
        print(
            "[gym_rollout_video] using Koopman LQR controller "
            "(wrapper applies u = -F · (z - z_ref_base), residual z_ref = 0)"
        )
        policy = _make_zero_residual_policy(env.latent_dim, device)

    max_steps = cfg.video.max_steps if cfg.video.max_steps is not None else cfg.eval_max_steps
    base_seed = cfg.video.seed if cfg.video.seed is not None else cfg.eval_seed
    num_videos = int(cfg.video.num_videos)
    if num_videos < 1:
        raise ValueError(f"video.num_videos must be >= 1, got {num_videos}")
    font = _load_font()
    # env.gamma_max is a 0-d device tensor in the refactored wrapper.
    gamma_max_scalar = float(env.gamma_max.item())
    pad = max(3, len(str(num_videos - 1)))
    print(
        f"[gym_rollout_video] rolling out {num_videos} video(s), up to {max_steps} steps each "
        f"(base seed={base_seed}, γmax={gamma_max_scalar:.4e})"
    )
    try:
        for i in range(num_videos):
            seed = base_seed + i
            stem = "rollout" if num_videos == 1 else f"rollout_{i:0{pad}d}"
            frames = _rollout(
                env, policy,
                vec_env=vec_env,
                max_steps=max_steps,
                seed=seed,
                gamma_max=gamma_max_scalar,
                bar_pct_cap=cfg.video.bar_pct_cap,
                font=font,
            )
            if not frames:
                raise RuntimeError(
                    f"No frames captured for rollout {i} (seed={seed}) — "
                    "env.render() returned None. Check that the env supports "
                    "render_mode='rgb_array'."
                )
            written = _write_outputs(
                frames, out_dir, stem=stem, formats=cfg.video.formats, fps=cfg.video.fps,
            )
            for p in written:
                print(
                    f"[gym_rollout_video] [{i + 1}/{num_videos}] wrote {p}  "
                    f"({len(frames)} frames @ {cfg.video.fps} fps, seed={seed})"
                )
    finally:
        env.close()

    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a rollout video for a trained Koopman LQR controller.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ConfigManager.load_stage(args.config, "eval_cfg")
    run(cfg)


if __name__ == "__main__":
    main()
