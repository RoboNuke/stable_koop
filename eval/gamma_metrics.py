"""Aggregate per-step gamma data from a multi-env rollout.

``gamma_t`` / ``eta_t`` / ``stability_term`` are emitted on the per-step
``info`` dict by :class:`wrappers.residual.ResidualPolicyEnv`. The rollout
loop appends only the *valid* values per env (steps where the env was still
in its original trajectory — i.e. before termination), so the post-auto-reset
gamma at the done step itself is naturally excluded.

The summary uses the unweighted stability term ``gamma_max - gamma_t - eta_t``
as the gamma reward.
"""

from __future__ import annotations

import numpy as np


def summarize_gammas(
    *,
    gamma: np.ndarray,
    eta: np.ndarray,
    stability_term: np.ndarray,
    gamma_max: float,
) -> dict:
    """Reduce flat per-step arrays to summary scalars.

    All three inputs must be 1-D, the same length, and contain only *valid*
    steps (the rollout drops the spurious post-auto-reset entries).

    Returns a dict with:
      * ``num_valid_steps``       — total counted steps
      * ``violation_rate``        — fraction of steps with ``gamma > gamma_max``
      * ``mean_normalized_gamma`` — mean of ``gamma / gamma_max``
      * ``mean_gamma_reward``     — mean of ``gamma_max - gamma - eta``
      * ``max_gamma`` / ``p95_gamma`` / ``mean_gamma`` — raw gamma summary
      * ``mean_eta``              — mean eta
      * ``gamma_max``             — passthrough for downstream readers
    """
    g = np.asarray(gamma, dtype=np.float64).ravel()
    e = np.asarray(eta, dtype=np.float64).ravel()
    s = np.asarray(stability_term, dtype=np.float64).ravel()
    n_valid = g.size
    if n_valid == 0:
        return {
            "num_valid_steps": 0,
            "violation_rate": float("nan"),
            "mean_normalized_gamma": float("nan"),
            "mean_gamma_reward": float("nan"),
            "max_gamma": float("nan"),
            "p95_gamma": float("nan"),
            "mean_gamma": float("nan"),
            "mean_eta": float("nan"),
            "gamma_max": float(gamma_max),
        }

    return {
        "num_valid_steps": int(n_valid),
        "violation_rate": float((g > gamma_max).mean()),
        "mean_normalized_gamma": float((g / gamma_max).mean()),
        "mean_gamma_reward": float(s.mean()),
        "max_gamma": float(g.max()),
        "p95_gamma": float(np.percentile(g, 95)),
        "mean_gamma": float(g.mean()),
        "mean_eta": float(e.mean()),
        "gamma_max": float(gamma_max),
    }


def compare_modes(per_mode: dict[str, dict]) -> dict:
    """Side-by-side comparison block for ``gamma_metrics_summary.yaml``.

    ``per_mode`` is ``{"base": summary, "lqr": summary, ...}`` where each
    summary is the output of :func:`summarize_gammas`. Returns a dict that
    yaml-dumps cleanly: a per-mode block plus a flat ``ranking`` listing
    modes by ``violation_rate`` (ascending) for quick eyeballing.
    """
    block: dict = {mode: dict(summary) for mode, summary in per_mode.items()}
    def _key(item):
        rate = item[1].get("violation_rate", float("inf"))
        return float("inf") if np.isnan(rate) else rate

    ranking = sorted(per_mode.items(), key=_key)
    block["ranking_by_violation_rate"] = [mode for mode, _ in ranking]
    return block
