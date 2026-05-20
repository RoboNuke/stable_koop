"""LQR compliance visualization: per-voxel ``γ_max − one-step error``.

Reuses the same 2D / 3D voxel binning as the training-time error heatmap,
but the color encodes *distance to the γ-bound*: positive = compliant,
negative = violation. The colormap is diverging red→green with a sharp
transition at 0, so any positive voxel reads green and any negative reads
red.

The visualization is driven by the trained Koopman's ``VisualizationCfg``
(carried in the saved checkpoint), so the controller fit doesn't need its
own config block.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from config.manager import VisualizationCfg
from train_koopman.visualize import compute_voxel_errors, render_voxel_grid


# Diverging colormap with a sharp red/green transition at the center stop.
# Any value just above 0 reads a clear light-green; just below 0, light-red.
# Larger magnitudes deepen toward dark-red / dark-green respectively.
_COMPLIANCE_CMAP = LinearSegmentedColormap.from_list(
    "compliance",
    [
        (0.0, "#660000"),     # deep red — large violation
        (0.499, "#ee5050"),   # light red — slight violation
        (0.501, "#50ee50"),   # light green — slight compliance
        (1.0, "#006600"),     # deep green — large compliance
    ],
)


def save_compliance_visualization(
    *,
    out_dir: Path,
    model,
    ds,
    aug_trajectories,
    viz_cfg: VisualizationCfg,
    device,
    gamma_max: float,
    label: str = "config_eta",
    title_suffix: str | None = None,
) -> dict | None:
    """Voxel ``γ_max − one-step error`` heatmap (LQR compliance view).

    ``label`` distinguishes multiple renders sharing an output dir (e.g.
    ``"no_eta"`` vs ``"config_eta"``); it's appended to the filename and,
    if ``title_suffix`` is omitted, also used to suffix the plot title.
    """
    print(f"\n=== Voxel γ-compliance visualization ({label}) ===")
    result = compute_voxel_errors(model, ds, aug_trajectories, viz_cfg, device)
    if result is None:
        print("  no transitions fell into any voxel; skipping.")
        return None
    grid_error, bounds_low, bounds_high = result
    grid_compliance = float(gamma_max) - grid_error
    finite = grid_compliance[np.isfinite(grid_compliance)]
    if finite.size == 0:
        print("  voxel grid empty after compliance transform; skipping.")
        return None
    max_abs = float(np.max(np.abs(finite)))
    if max_abs == 0.0:
        max_abs = 1e-9  # avoid TwoSlopeNorm degeneracy

    print(
        f"  γ_max = {gamma_max:.4e}  "
        f"compliance range = [{finite.min():.4e}, {finite.max():.4e}]  "
        f"violations = {int((finite < 0).sum())}/{finite.size} voxels"
    )

    suffix = title_suffix if title_suffix is not None else label
    return render_voxel_grid(
        out_dir=out_dir,
        grid=grid_compliance,
        bounds_low=bounds_low,
        bounds_high=bounds_high,
        voxel_size=viz_cfg.voxel_size,
        gif_seconds=viz_cfg.gif_seconds,
        cmap=_COMPLIANCE_CMAP,
        norm=TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs),
        value_label="γ_max − mean error  (positive = compliant)",
        title=f"γ-compliance ({suffix})",
        slices_subdir=f"compliance_slices_{label}",
        gif_name=f"compliance_sweep_{label}.gif",
        image_name=f"compliance_heatmap_{label}.png",
    )


def save_ctrl_sweep_plot(
    *,
    out_dir,
    sweep: dict,
    highlight_points: list[dict] | None = None,
) -> str:
    """Two-y-axis plot of the dense ctrl_percentage sweep.

    X-axis spans ``[0, u_max]`` (the ctrl_percentage values rescaled by
    ``u_max``). Two lines, each on its own y-axis:

      * % transitions covered (left)
      * γ / R (right)

    (γ_max itself is omitted because, with R constant, γ_max and γ/R are
    a scaled copy of each other — they overlap perfectly on rescaled axes.)

    ``highlight_points`` (the ``ctrl_pct_results`` list) is overlaid as
    dots on both lines with an annotation per point showing ``%covered``
    and ``γ/R``.
    """
    out_path = Path(out_dir) / "ctrl_budget_sweep.png"
    pcts = np.asarray(sweep["ctrl_percentages"], dtype=float)
    u_max = float(sweep["u_max"])
    x = pcts * u_max
    fracs_pct = 100.0 * np.asarray(sweep["frac_transitions_covered"], dtype=float)
    ratios = np.asarray(sweep["gamma_over_R"], dtype=float)

    # Font sizes: keep annotation text at least as large as axis labels.
    label_fs = 13
    annot_fs = 13
    title_fs = 14
    tick_fs = 11

    fig, ax_cov = plt.subplots(figsize=(11, 6))

    color_cov = "#2ca02c"
    color_ratio = "#d62728"

    (line_cov,) = ax_cov.plot(
        x, fracs_pct, color=color_cov, label="% transitions covered"
    )
    ax_cov.set_xlabel("control budget   (ctrl_pct · u_max)", fontsize=label_fs)
    ax_cov.set_ylabel("% transitions covered", color=color_cov, fontsize=label_fs)
    ax_cov.tick_params(axis="y", labelcolor=color_cov, labelsize=tick_fs)
    ax_cov.tick_params(axis="x", labelsize=tick_fs)
    ax_cov.set_xlim(0.0, u_max)
    ax_cov.set_ylim(-2.0, 102.0)

    ax_ratio = ax_cov.twinx()
    (line_ratio,) = ax_ratio.plot(x, ratios, color=color_ratio, label="γ / R")
    ax_ratio.set_ylabel("γ / R", color=color_ratio, fontsize=label_fs)
    ax_ratio.tick_params(axis="y", labelcolor=color_ratio, labelsize=tick_fs)

    # Highlight markers at each ctrl_percentage that has a heatmap, with an
    # annotation near the coverage dot summarizing the stats at that point.
    if highlight_points:
        for entry in highlight_points:
            pct = float(entry["ctrl_percentage"])
            x_at = pct * u_max
            cov = entry.get("coverage", {})
            frac = float(cov.get("frac_transitions_under_gamma", 0.0))
            ratio = float(cov.get("ratio_to_R", float("nan")))

            ax_cov.plot(x_at, 100.0 * frac, marker="o", color=color_cov,
                        markersize=7, zorder=5)
            ax_ratio.plot(x_at, ratio, marker="o", color=color_ratio,
                          markersize=7, zorder=5)

            # Position the annotation per-quadrant so heavy-overlap points
            # land in free space (offsets are all relative to the dot):
            #   * no-control (pct=0): SW of the dot — text extends down-
            #     and-left, partially overlapping the y-axis area instead
            #     of the title.
            #   * 100%-control (pct=1): NW of the dot — text extends up-
            #     and-left into open space instead of running outside the
            #     right edge of the plot.
            #   * everything else: default NE.
            text = f"{100.0 * frac:.1f}% cov\nγ/R = {ratio:.2f}"
            bbox = dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="0.6",
                alpha=0.85,
            )
            if pct <= 1e-6:
                offset_xy = (-8, -8)
                ha, va = "right", "top"
            elif pct >= 1.0 - 1e-6:
                offset_xy = (-8, 8)
                ha, va = "right", "bottom"
            else:
                offset_xy = (8, 8)
                ha, va = "left", "bottom"
            ax_cov.annotate(
                text,
                xy=(x_at, 100.0 * frac),
                xytext=offset_xy,
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=annot_fs,
                color="black",
                bbox=bbox,
                zorder=6,
            )

    ax_cov.set_title(
        f"Coverage + γ/R vs control budget   (η = ctrl_pct · ||B||·u_max,  "
        f"u_max = {u_max:.2e})",
        fontsize=title_fs,
    )
    ax_cov.legend(
        [line_cov, line_ratio],
        [l.get_label() for l in (line_cov, line_ratio)],
        loc="best",
        fontsize=label_fs,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved control-budget sweep plot to {out_path}")
    return str(out_path)
