# Per-env extension contract

The framework is environment-agnostic: every env-specific behavior is supplied
through a small set of registered callables. To add a new environment you
implement these callables (typically in `wrappers/<env>/` and
`eval/<env>_metrics.py`) and self-register them at import time.

This file is the single index of what is required and where it lives.

## Pipeline at a glance

```
gather_data  ──>  train_koopman  ──>  controller/lqr  ──>  train_residual  ──>  eval
     │                                                          │                 │
     │                                                          │                 │
     ▼                                                          ▼                 ▼
ENV_WRAPPERS                                            ENV_BASE_GOALS    ENV_SCORERS
  (build env)                                           (z_ref_t_base)    (success / metrics)
policy._REGISTRY                                                          ENV_BASE_GOALS
  (hand-designed                                                          (residual wrapper)
   base policy)
```

## Required registrations

Every environment that is run end-to-end through the pipeline must register
all three callables below. They are all keyed by the gymnasium `env_name`.

| # | What | Where it's registered | Signature | Purpose |
|---|------|----------------------|-----------|---------|
| 1 | **Wrapper builder** | `data/env_builder.py::register_env_wrappers` | `(env: gym.Env, **env_kwargs) -> gym.Env` | Applies env-specific obs transforms / dynamics tweaks / spawn limiting. Receives the YAML's `env_kwargs` block verbatim. |
| 2 | **Base-goal getter** | `data/env_builder.py::register_env_base_goal` | `(env: gym.Env) -> np.ndarray` of shape `(x_dim,)` *or* `(num_envs, x_dim)` | Returns `x_base` in **raw env-obs space**. The residual wrapper encodes it (via `koopman.encode(x_base / obs_scale)`) into `z_ref_t_base` each step so the LQR drives the system toward this target rather than the latent origin. Re-queried every step — may vary in time. |
| 3 | **Scorer** | `eval/__init__.py::register_env_scorer` | `EnvScorer(check_success, compute_metrics)` | Pair of callables defining (a) the trajectory-level success criterion and (b) the per-trajectory task metrics surfaced in the eval report. |

If `register_env_base_goal` is missing for an env, `ResidualPolicyEnv` raises
`KeyError` at construction. The hard error is intentional — silently defaulting
to zero would change the LQR target without warning.

## Optional hooks

| What | Where | Purpose |
|------|-------|---------|
| **Hand-designed base policy** | `policy/__init__.py::_REGISTRY` (entries like `"pd"`, `"energy"`, `"LQR_policy"`) | Used by `data/gather_data.py` to collect the dataset that trains the Koopman model. Selected by name from the gather YAML's `base_policy.name`. |
| **Custom `gym.register(...)` call** | Anywhere imported before `gym.make(env_name)` | Needed when the env isn't shipped by gymnasium. Existing example: `eval/__init__.py` registers `InvertedPendulum-v4/-v5` against the local entry point. |
| **Per-env eval plotting** | `eval/<env>_metrics.py` (e.g. heatmaps) | Optional; the scorer-returned metrics already feed the standard reporter. |

## Worked example — Pendulum-v1

These three commits suffice to make `Pendulum-v1` runnable end-to-end:

1. **Wrapper builder** — `wrappers/pendulum/__init__.py`:
   ```python
   def apply_pendulum_wrappers(env, *, obs_type="cos_sin", limited_spawn=False, ...):
       ...
       return env
   ```
   Registered from `data/env_builder.py`:
   ```python
   register_env_wrappers("Pendulum-v1", apply_pendulum_wrappers)
   ```

2. **Base-goal getter** — `wrappers/pendulum/__init__.py`:
   ```python
   def pendulum_base_goal(env):
       obs_dim = env.observation_space.shape[-1]
       if obs_dim == 3:    # [cos θ, sin θ, θ̇]
           return np.array([1.0, 0.0, 0.0], dtype=np.float32)
       if obs_dim == 2:    # [θ, θ̇]
           return np.array([0.0, 0.0], dtype=np.float32)
       raise ValueError(...)
   ```
   Registered from `data/env_builder.py`:
   ```python
   register_env_base_goal("Pendulum-v1", pendulum_base_goal)
   ```

3. **Scorer** — `eval/pendulum_metrics.py` defines `pendulum_check_success`
   and `pendulum_compute_metrics`. Registered from `eval/__init__.py`:
   ```python
   register_env_scorer(
       "Pendulum-v1",
       check_success=pendulum_check_success,
       compute_metrics=pendulum_compute_metrics,
   )
   ```

## Adding a new env (checklist)

1. Add `wrappers/<env>/__init__.py` with the wrapper builder and base-goal
   getter (skip the builder if the stock gym env needs no wrapping — see
   `wrappers/inverted_pendulum/__init__.py`).
2. Import from `data/env_builder.py` and call `register_env_wrappers` and
   `register_env_base_goal`.
3. Add `eval/<env>_metrics.py` with `check_success` and `compute_metrics`,
   then register from `eval/__init__.py`.
4. (Optional) Add a hand-designed base policy under `policy/<env>/` and
   register it in `policy/__init__.py::_REGISTRY` so the gather YAML can
   select it by name.
5. (Optional) Add a YAML config for each pipeline stage under
   `config/<stage>/<env>.yaml`.
