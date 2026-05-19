"""LQR compliance visualization: per-voxel ``γ_max − one-step error``.

Same 3D voxel binning as the training-time error heatmap, but the color
encodes *distance to the γ-bound*: positive = compliant, negative =
violation. The colormap is diverging red→green with a sharp transition at
0, so any positive voxel reads green and any negative reads red.

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
) -> dict | None:
    """Voxel ``γ_max − one-step error`` heatmap (LQR compliance view)."""
    print("\n=== Voxel γ-compliance visualization ===")
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
        title_template=lambda z: f"γ-compliance  |  z = {z:+.3f}",
    )
