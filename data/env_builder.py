"""Env-agnostic gym environment construction.

The framework does not know about any specific environment. Each environment
registers its own wrapper builder against ``ENV_WRAPPERS``; the dispatcher
here looks up the env_name and applies that builder.

To add an environment: write a wrapper builder taking ``(env, **kwargs) -> env``,
then call :func:`register_env_wrappers("Your-Env-v0", builder)` (typically from
that env's own module).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np
import torch

# Map env_name -> wrapper-builder. The builder takes the raw env plus
# arbitrary kwargs from the YAML and returns the wrapped env.
ENV_WRAPPERS: dict[str, Callable[..., gym.Env]] = {}

# Map env_name -> base-goal getter. The getter takes the wrapped (single)
# env and returns the per-env base goal ``x_base`` in raw observation space.
# Shape ``(x_dim,)`` is broadcast to the batch; ``(num_envs, x_dim)`` is
# used row-wise. Consumed by ``wrappers.residual.ResidualPolicyEnv`` to form
# the latent reference ``z_ref_t_base = koopman.encode(x_base / obs_scale)``
# so the LQR drives the system toward a known target rather than the
# latent origin.
BaseGoalFn = Callable[[gym.Env], Union[np.ndarray, torch.Tensor]]
ENV_BASE_GOALS: dict[str, BaseGoalFn] = {}


def register_env_wrappers(env_name: str, builder: Callable[..., gym.Env]) -> None:
    """Register ``builder`` as the wrapper stack for ``env_name``.

    Re-registration replaces the previous builder so tests can rebind freely.
    """
    ENV_WRAPPERS[env_name] = builder


def register_env_base_goal(env_name: str, fn: BaseGoalFn) -> None:
    """Register ``fn`` as the base-goal getter for ``env_name``."""
    ENV_BASE_GOALS[env_name] = fn


def get_base_goal_fn(env_name: str) -> BaseGoalFn:
    """Return the registered base-goal getter, or raise ``KeyError``."""
    if env_name not in ENV_BASE_GOALS:
        raise KeyError(
            f"No base-goal getter registered for {env_name!r}. Register one "
            f"via register_env_base_goal(); see data/README.md for the "
            f"per-env extension contract."
        )
    return ENV_BASE_GOALS[env_name]


# Self-register the pendulum wrapper builder + base goal. New envs add their
# own `register_env_wrappers(...)` / `register_env_base_goal(...)` calls at
# import time from their own module.
from wrappers.pendulum import apply_pendulum_wrappers, pendulum_base_goal  # noqa: E402

register_env_wrappers("Pendulum-v1", apply_pendulum_wrappers)
register_env_base_goal("Pendulum-v1", pendulum_base_goal)

# InvertedPendulum-v4/-v5 have no custom wrapper builder but need a base goal.
from wrappers.inverted_pendulum import inverted_pendulum_base_goal  # noqa: E402

register_env_base_goal("InvertedPendulum-v4", inverted_pendulum_base_goal)
register_env_base_goal("InvertedPendulum-v5", inverted_pendulum_base_goal)


def make_single_env(
    env_name: str,
    *,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[dict] = None,
) -> gym.Env:
    """Build one env and apply the registered wrapper stack (if any).

    ``env_kwargs`` is the env-specific blob from the YAML; it is passed
    verbatim to the registered builder. Envs with no registered wrappers
    accept only an empty ``env_kwargs``.
    """
    env_kwargs = env_kwargs or {}
    env = gym.make(env_name, render_mode=render_mode) if render_mode else gym.make(env_name)
    builder = ENV_WRAPPERS.get(env_name)
    if builder is not None:
        env = builder(env, **env_kwargs)
    elif env_kwargs:
        raise ValueError(
            f"No wrappers registered for env {env_name!r}; got env_kwargs "
            f"{sorted(env_kwargs)}. Either drop the kwargs from the YAML or "
            f"register a builder via register_env_wrappers()."
        )
    return env


def make_eval_env(
    env_name: str,
    *,
    num_parallel_evals: int = 1,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[dict] = None,
):
    """Build a vectorized env if ``num_parallel_evals > 1``, else a single env."""
    if num_parallel_evals > 1:
        return gym.vector.SyncVectorEnv(
            [
                lambda: make_single_env(env_name, render_mode=render_mode, env_kwargs=env_kwargs)
                for _ in range(num_parallel_evals)
            ]
        )
    return make_single_env(env_name, render_mode=render_mode, env_kwargs=env_kwargs)
