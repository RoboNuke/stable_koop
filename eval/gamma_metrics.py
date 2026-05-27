"""Aggregate per-step gamma data from a multi-env rollout.

``gamma_t`` / ``eta_t`` / ``stability_term`` are emitted on the per-step
``info`` dict by :class:`wrappers.residual.ResidualPolicyEnv`. The rollout
loop appends only the *valid* values per env (steps where the env was still
in its original trajectory — i.e. before termination), so the post-auto-reset
gamma at the done step itself is naturally excluded.

Each metric is computed per trajectory first, then grouped into
``combined`` / ``success`` / ``failure`` blocks. Where it makes sense each
block reports ``{mean, std}`` over the trajectories in the group;
trajectory-count / step-count totals and the true ``max_pred_error`` are
scalars.
"""

from __future__ import annotations

import numpy as np


def _per_traj_stats(per_trajectory, gamma_max: float) -> list[dict]:
    """Compute per-trajectory gamma stats once; everything else aggregates over these."""
    out: list[dict] = []
    for t in per_trajectory:
        g = np.asarray(t.get("gamma_t", []), dtype=np.float64).ravel()
        e = np.asarray(t.get("eta_t", []), dtype=np.float64).ravel()
        s = np.asarray(t.get("stability_term", []), dtype=np.float64).ravel()
        if g.size == 0:
            continue
        out.append({
            "success": bool(t.get("success", False)),
            "n_steps": int(g.size),
            "mean_pred_error": float(g.mean()),
            "max_pred_error": float(g.max()),
            "mean_eta": float(e.mean()) if e.size else 0.0,
            "gamma_reward": float(s.mean()) if s.size else 0.0,
            "violation_rate": float((g > gamma_max).mean()),
            # frac_used_cert is only meaningful when the certificate is
            # positive; ``None`` here signals downstream "no bound".
            "frac_used_cert": (
                float((g / gamma_max).mean()) if gamma_max > 0 else None
            ),
            "_g": g,  # kept for true-max across the whole group
        })
    return out


def _mean_std(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std())}


def _group_block(traj_stats: list[dict], gamma_max: float) -> dict:
    # ``frac_used_cert`` is ``None`` when ``gamma_max <= 0`` so downstream
    # printers / yaml readers can render "no bound" instead of bogus numbers.
    frac_block: dict | None
    if gamma_max <= 0:
        frac_block = None
    elif not traj_stats:
        frac_block = {"mean": float("nan"), "std": float("nan")}
    else:
        frac_block = _mean_std([t["frac_used_cert"] for t in traj_stats])

    if not traj_stats:
        return {
            "num_trajectories": 0,
            "total_steps": 0,
            "violation_rate": {"mean": float("nan"), "std": float("nan")},
            "mean_pred_error": {"mean": float("nan"), "std": float("nan")},
            "avg_max_pred_error": {"mean": float("nan"), "std": float("nan")},
            "max_pred_error": float("nan"),
            "frac_used_cert": frac_block,
            "gamma_reward": {"mean": float("nan"), "std": float("nan")},
            "mean_eta": {"mean": float("nan"), "std": float("nan")},
            "gamma_max": float(gamma_max),
        }
    return {
        "num_trajectories": len(traj_stats),
        "total_steps": int(sum(t["n_steps"] for t in traj_stats)),
        "violation_rate": _mean_std([t["violation_rate"] for t in traj_stats]),
        "mean_pred_error": _mean_std([t["mean_pred_error"] for t in traj_stats]),
        "avg_max_pred_error": _mean_std([t["max_pred_error"] for t in traj_stats]),
        "max_pred_error": float(np.concatenate([t["_g"] for t in traj_stats]).max()),
        "frac_used_cert": frac_block,
        "gamma_reward": _mean_std([t["gamma_reward"] for t in traj_stats]),
        "mean_eta": _mean_std([t["mean_eta"] for t in traj_stats]),
        "gamma_max": float(gamma_max),
    }


def summarize_gammas(*, per_trajectory: list[dict], gamma_max: float) -> dict:
    """Return ``{combined, success, failure}`` blocks of gamma metrics.

    Each block carries:
      * ``num_trajectories`` / ``total_steps`` — group sizes
      * ``violation_rate``     {mean, std} — per-traj ``(γ_t > γ_max).mean()``
      * ``mean_pred_error``    {mean, std} — per-traj mean of ``γ_t``
      * ``avg_max_pred_error`` {mean, std} — mean of per-traj max ``γ_t``
      * ``max_pred_error``     — true max ``γ_t`` across every step in group
      * ``gamma_reward``       {mean, std} — per-traj mean of stability term
      * ``mean_eta``           {mean, std} — per-traj mean of ``η_t``
      * ``gamma_max``          — passthrough scalar
    """
    traj_stats = _per_traj_stats(per_trajectory, gamma_max)
    succ = [t for t in traj_stats if t["success"]]
    fail = [t for t in traj_stats if not t["success"]]
    return {
        "combined": _group_block(traj_stats, gamma_max),
        "success": _group_block(succ, gamma_max),
        "failure": _group_block(fail, gamma_max),
    }


def compare_modes(per_mode: dict[str, dict]) -> dict:
    """Side-by-side comparison block for ``eval_gamma_metrics_summary.yaml``.

    ``per_mode`` is ``{"base": summary, "lqr": summary, ...}`` where each
    summary is the output of :func:`summarize_gammas`. Returns a dict that
    yaml-dumps cleanly: a per-mode block plus a flat ``ranking_by_violation_rate``.
    """
    block: dict = {mode: dict(summary) for mode, summary in per_mode.items()}

    def _key(item):
        rate = item[1].get("combined", {}).get("violation_rate", {}).get("mean", float("inf"))
        if isinstance(rate, float) and np.isnan(rate):
            return float("inf")
        return float(rate)

    ranking = sorted(per_mode.items(), key=_key)
    block["ranking_by_violation_rate"] = [mode for mode, _ in ranking]
    return block
