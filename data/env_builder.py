"""Env-agnostic gym environment construction.

The framework does not know about any specific environment. Each environment
registers its own wrapper builder against ``ENV_WRAPPERS``; the dispatcher
here looks up the env_name and applies that builder.

To add an environment: write a wrapper builder taking ``(env, **kwargs) -> env``,
then call :func:`register_env_wrappers("Your-Env-v0", builder)` (typically from
that env's own module).
"""

from __future__ import annotations

from typing import Callable, Optional

import gymnasium as gym

# Map env_name -> wrapper-builder. The builder takes the raw env plus
# arbitrary kwargs from the YAML and returns the wrapped env.
ENV_WRAPPERS: dict[str, Callable[..., gym.Env]] = {}


def register_env_wrappers(env_name: str, builder: Callable[..., gym.Env]) -> None:
    """Register ``builder`` as the wrapper stack for ``env_name``.

    Re-registration replaces the previous builder so tests can rebind freely.
    """
    ENV_WRAPPERS[env_name] = builder


# Self-register the pendulum wrapper builder. New envs add their own
# `register_env_wrappers(...)` call at import time from their own module.
from wrappers.pendulum import apply_pendulum_wrappers  # noqa: E402

register_env_wrappers("Pendulum-v1", apply_pendulum_wrappers)


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
