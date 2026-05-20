"""Evaluation entry points + per-env scorer registry.

The framework (``eval/rollout.py``, ``eval/multi_mode.py``,
``eval/koopman_accuracy.py``) is environment-agnostic. Per-env scoring
lives in dedicated modules (e.g. :mod:`eval.pendulum_metrics`) and
registers itself here.

To add an environment: write a module exposing ``check_success(states, cfg) -> bool``
and ``compute_metrics(states, actions) -> dict``, then call
:func:`register_env_scorer("Your-Env-v0", check_success=..., compute_metrics=...)`
from that module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EnvScorer:
    """Env-specific scoring hooks dispatched by env_name."""

    check_success: Callable
    """``check_success(states: np.ndarray, cfg: dict) -> bool``."""
    compute_metrics: Callable
    """``compute_metrics(states: np.ndarray, actions: np.ndarray) -> dict``."""
    make_koopman_heatmap: Optional[Callable] = None
    """Optional ``(model, trajectories, train_horizon, eval_horizon, ...)
    -> (fig, heatmap_data)`` hook used by the koopman-accuracy CLI when an
    env wants a custom visualization."""


ENV_SCORERS: dict[str, EnvScorer] = {}


def register_env_scorer(
    env_name: str,
    *,
    check_success: Callable,
    compute_metrics: Callable,
    make_koopman_heatmap: Optional[Callable] = None,
) -> None:
    """Register the per-env scorer used by :func:`eval.rollout.do_rollout`."""
    ENV_SCORERS[env_name] = EnvScorer(
        check_success=check_success,
        compute_metrics=compute_metrics,
        make_koopman_heatmap=make_koopman_heatmap,
    )


def get_env_scorer(env_name: str) -> EnvScorer:
    if env_name not in ENV_SCORERS:
        raise KeyError(
            f"No eval scorer registered for env {env_name!r}; "
            f"registered envs: {sorted(ENV_SCORERS)}"
        )
    return ENV_SCORERS[env_name]


# Self-register the pendulum scorer. New envs add their own
# register_env_scorer(...) call from their own module.
from eval.pendulum_metrics import (  # noqa: E402
    make_koopman_angle_heatmap,
    pendulum_check_success,
    pendulum_compute_metrics,
)

register_env_scorer(
    "Pendulum-v1",
    check_success=pendulum_check_success,
    compute_metrics=pendulum_compute_metrics,
    make_koopman_heatmap=make_koopman_angle_heatmap,
)

from eval.inverted_pendulum_metrics import (  # noqa: E402
    inverted_pendulum_check_success,
    inverted_pendulum_compute_metrics,
)

# v4 and v5 are functionally identical for InvertedPendulum (same obs layout,
# termination at |theta|>0.2, max_episode_steps=1000) — only gymnasium's
# internal MuJoCo bindings differ. Register the scorer under both names so
# whichever the gather YAML picks works out of the box.
for _ip_name in ("InvertedPendulum-v4", "InvertedPendulum-v5"):
    register_env_scorer(
        _ip_name,
        check_success=inverted_pendulum_check_success,
        compute_metrics=inverted_pendulum_compute_metrics,
    )


from eval.rollout import do_rollout, load_eval_stats  # noqa: E402
from eval.koopman_accuracy import evaluate_model  # noqa: E402

__all__ = [
    "ENV_SCORERS",
    "EnvScorer",
    "do_rollout",
    "evaluate_model",
    "get_env_scorer",
    "load_eval_stats",
    "register_env_scorer",
]
