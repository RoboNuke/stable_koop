"""Hand-designed lifting functions (pure tensor maps, no learnable params).

Currently used: the trigonometric pendulum lifting embedded inside
:class:`KoopmanAutoencoder` for the ``encoder_type="trig"`` pathway. The lifting
operates on either ``[cos θ, sin θ, θ̇]`` (``obs_type="cos_sin"``) or
``[θ, θ̇]`` (``obs_type="theta"``) and emits pairs ``(cos θ · θ̇ⁿ, sin θ · θ̇ⁿ)``
for ``n = 0..k-1``.

The functional ``trig_lift`` here is the canonical implementation; the
:class:`KoopmanAutoencoder` orchestrator calls into it for the ``trig`` path.
"""

from __future__ import annotations

import torch


def trig_lift(x: torch.Tensor, latent_dim: int, obs_type: str) -> torch.Tensor:
    """Trigonometric lifting used by the pendulum experiment.

    Emits ``latent_dim`` features, ``latent_dim`` must be even.
    """
    assert latent_dim % 2 == 0, f"trig lift requires even latent_dim, got {latent_dim}"
    if obs_type == "cos_sin":
        cos_th = x[..., 0:1]
        sin_th = x[..., 1:2]
        thdot = x[..., 2:3]
        n_start = 1  # cos/sin and θ̇ already prepended, avoid redundancy
    else:
        theta = x[..., 0:1]
        thdot = x[..., 1:2]
        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        n_start = 0
    n_pairs = latent_dim // 2
    parts = []
    for n in range(n_start, n_start + n_pairs):
        scale = thdot ** n
        parts.append(scale * cos_th)
        parts.append(scale * sin_th)
    return torch.cat(parts, dim=-1)
