"""Voxel-binned grid visualizer + the state-error rendering used during training.

The :func:`compute_voxel_errors` helper produces a 3D grid of mean one-step
state-prediction errors (binned by goal-relative object position). It is
reused by:

* :func:`save_voxel_visualization` (this module) — Koopman-training
  visualization. Inferno colormap of the error itself.
* :func:`controller.lqr.visualize.save_compliance_visualization` — LQR
  compliance visualization. Diverging red→green colormap of ``γ − error``.

Both rendering callers go through :func:`render_voxel_grid`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image

from config.manager import VisualizationCfg


# ---------------------------------------------------------------------------
# Voxelization (env-agnostic; just bins per-transition errors).
# ---------------------------------------------------------------------------


def compute_voxel_errors(
    model,
    ds,
    aug_trajectories,
    viz_cfg: VisualizationCfg,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute the ``(Nx, Ny, Nz)`` voxel grid of mean one-step state errors.

    Returns ``(grid, bounds_low, bounds_high)`` (NaN for empty voxels) or
    ``None`` if no transitions fell into any voxel.
    """
    obj_idx = list(viz_cfg.object_pos_obs_indices)
    if len(obj_idx) != 3:
        raise ValueError(
            f"visualization.object_pos_obs_indices must have length 3, "
            f"got {obj_idx!r}"
        )
    goal_idx = list(viz_cfg.goal_obs_indices) if viz_cfg.goal_obs_indices else None
    if goal_idx is not None and len(goal_idx) != 3:
        raise ValueError(
            f"visualization.goal_obs_indices must have length 3 or be null, "
            f"got {goal_idx!r}"
        )

    voxel_size = float(viz_cfg.voxel_size)
    if voxel_size <= 0:
        raise ValueError(f"visualization.voxel_size must be > 0, got {voxel_size}")

    positions_all: list[np.ndarray] = []
    errors_all: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (raw_states, _raw_acts, _ba, _r), (scaled_states, scaled_actions) in zip(
            ds.trajectories, aug_trajectories
        ):
            T_act = len(scaled_actions)
            if T_act == 0:
                continue

            if goal_idx is not None:
                goal = np.asarray(raw_states[0, goal_idx], dtype=np.float64)
            else:
                goal = np.zeros(3, dtype=np.float64)

            states_t = torch.tensor(scaled_states, dtype=torch.float32, device=device)
            actions_t = torch.tensor(scaled_actions, dtype=torch.float32, device=device)
            z_t = model.encode(states_t[:T_act])
            z_pred = model.predict(z_t, actions_t[:T_act])
            x_pred = model.decode(z_pred)
            x_next = states_t[1 : T_act + 1]
            errs = torch.linalg.norm(x_next - x_pred, dim=-1).cpu().numpy()

            positions = (
                np.asarray(raw_states[:T_act, obj_idx], dtype=np.float64) - goal
            )
            positions_all.append(positions)
            errors_all.append(errs)

    if not positions_all:
        return None

    positions = np.concatenate(positions_all, axis=0)  # (N, 3)
    errors = np.concatenate(errors_all, axis=0)  # (N,)

    voxel_indices = np.floor(positions / voxel_size).astype(np.int64)
    voxel_sums: dict[tuple[int, int, int], float] = defaultdict(float)
    voxel_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for idx, err in zip(voxel_indices, errors):
        key = (int(idx[0]), int(idx[1]), int(idx[2]))
        voxel_sums[key] += float(err)
        voxel_counts[key] += 1
    if not voxel_sums:
        return None

    min_idx = np.array([min(k[i] for k in voxel_sums) for i in range(3)])
    max_idx = np.array([max(k[i] for k in voxel_sums) for i in range(3)])
    shape = tuple((max_idx - min_idx + 1).tolist())
    grid = np.full(shape, np.nan, dtype=np.float64)
    for key, s in voxel_sums.items():
        local = tuple(int(key[i] - min_idx[i]) for i in range(3))
        grid[local] = s / voxel_counts[key]

    bounds_low = min_idx.astype(np.float64) * voxel_size
    bounds_high = (max_idx + 1).astype(np.float64) * voxel_size
    return grid, bounds_low, bounds_high


# ---------------------------------------------------------------------------
# Per-z-slice rendering (caller picks cmap / norm / labels).
# ---------------------------------------------------------------------------


def render_voxel_grid(
    *,
    out_dir: Path,
    grid: np.ndarray,
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    voxel_size: float,
    gif_seconds: float,
    cmap,
    norm: Normalize,
    value_label: str,
    title_template: Callable[[float], str],
    slices_subdir: str = "voxel_slices",
    gif_name: str = "voxel_sweep.gif",
) -> dict:
    """Write per-z-slice PNG heatmaps + a GIF; return a summary dict.

    ``cmap`` and ``norm`` are matplotlib objects controlling color mapping;
    ``value_label`` is the colorbar caption; ``title_template(z)`` builds
    the per-slice figure title (called with the z-center in workspace units).
    """
    out_dir = Path(out_dir)
    slices_dir = out_dir / slices_subdir
    slices_dir.mkdir(parents=True, exist_ok=True)

    n_z = grid.shape[2]
    z_centers = bounds_low[2] + (np.arange(n_z) + 0.5) * voxel_size
    extent = [bounds_low[0], bounds_high[0], bounds_low[1], bounds_high[1]]

    cmap_with_nan = cmap.copy()
    cmap_with_nan.set_bad(color="lightgrey")

    frame_paths: list[Path] = []
    for k, z in enumerate(z_centers):
        slc = grid[:, :, k].T  # transpose so x is the horizontal axis in imshow
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            np.ma.masked_invalid(slc),
            origin="lower",
            extent=extent,
            cmap=cmap_with_nan,
            norm=norm,
            aspect="equal",
        )
        ax.set_xlabel("x (goal-relative)")
        ax.set_ylabel("y (goal-relative)")
        ax.set_title(title_template(float(z)))
        fig.colorbar(im, ax=ax, label=value_label)
        path = slices_dir / f"z={z:+.3f}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(path)

    duration_ms = max(1, int(round(1000 * gif_seconds / len(frame_paths))))
    gif_path = out_dir / gif_name
    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )

    print(f"  saved {len(frame_paths)} z-slice PNGs to {slices_dir}")
    print(f"  saved GIF to {gif_path}  ({duration_ms} ms/frame, {len(frame_paths)} frames)")

    return {
        "slices_dir": str(slices_dir),
        "gif_path": str(gif_path),
        "num_z_slices": int(n_z),
        "voxel_size": float(voxel_size),
        "gif_seconds": float(gif_seconds),
        "bounds_low": bounds_low.tolist(),
        "bounds_high": bounds_high.tolist(),
        "grid_shape": list(grid.shape),
    }


# ---------------------------------------------------------------------------
# Koopman-training caller: error heatmap (inferno).
# ---------------------------------------------------------------------------


def save_voxel_visualization(
    *,
    out_dir: Path,
    model,
    ds,
    aug_trajectories,
    viz_cfg: VisualizationCfg,
    device,
) -> dict | None:
    """Voxel one-step state-error heatmap (training-time view)."""
    print("\n=== Voxel one-step error visualization ===")
    result = compute_voxel_errors(model, ds, aug_trajectories, viz_cfg, device)
    if result is None:
        print("  no transitions fell into any voxel; skipping.")
        return None
    grid, bounds_low, bounds_high = result
    print(
        f"  voxel_size: {viz_cfg.voxel_size}  shape: {tuple(grid.shape)}  "
        f"x:[{bounds_low[0]:.3f},{bounds_high[0]:.3f}]  "
        f"y:[{bounds_low[1]:.3f},{bounds_high[1]:.3f}]  "
        f"z:[{bounds_low[2]:.3f},{bounds_high[2]:.3f}]"
    )

    finite = grid[np.isfinite(grid)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    return render_voxel_grid(
        out_dir=out_dir,
        grid=grid,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        voxel_size=viz_cfg.voxel_size,
        gif_seconds=viz_cfg.gif_seconds,
        cmap=plt.cm.inferno,
        norm=Normalize(vmin=vmin, vmax=vmax),
        value_label="mean ||x_pred − x_next||",
        title_template=lambda z: f"One-step state error  |  z = {z:+.3f}",
    )
