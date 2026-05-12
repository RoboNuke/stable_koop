# stable_koop

Koopman + LQR + learned residual policy framework. Demonstrated on the
``Pendulum-v1`` swing-up task: a learned Koopman lifting (encoder + linear
operator A + control matrix B) approximates the closed-loop dynamics under a
hand-designed base policy; an LQR controller is fit on the latent linear
system to provide a stabilizing residual; an SAC policy learns a corrective
residual on top.

The codebase is environment-agnostic in shape — pendulum-specific pieces
(policies, wrappers, success criteria) are isolated under ``policy/pendulum/``
and ``wrappers/pendulum/`` so new environments can be added without touching
the framework.

## Repo layout

```
stable_koop/
├── config/
│   ├── manager/             # ConfigManager + per-stage dataclasses
│   └── exp_cfgs/            # Per-stage YAMLs, grouped by stage subfolder
├── models/                  # All nn.Module definitions
├── policy/                  # Hand-designed policies + name→class registry
│   └── pendulum/
├── controller/
│   ├── controller_analysis.py
│   └── lqr/                 # LQR class, analysis, CLI entry
├── data/                    # gather_data + dataloader + env_builder
├── train_koopman/           # losses, training_loop, two_phase, joint
├── train_residual/          # SAC residual policy
├── eval/                    # koopman_accuracy, policy_rollout, CLI
├── launch/                  # bash launchers, one per stage + run_all.sh
├── wrappers/                # env-agnostic + pendulum-specific wrappers
└── tests/
```

## Quickstart

Create the conda env once:

```
mamba env create -f envy.yml
```

Run the full pipeline end-to-end for the bundled pendulum experiment:

```
bash launch/run_all.sh pendulum
```

This runs ``gather_data → train_koopman → fit_controller → train_residual → eval``
in sequence using ``config/exp_cfgs/<stage>/pendulum.yaml`` for each stage.

## Running individual stages

Each stage is independently runnable against the previous stage's saved
outputs:

```
bash launch/gather_data.sh   config/exp_cfgs/gather_data/pendulum.yaml
bash launch/train_koopman.sh config/exp_cfgs/train_koopman/pendulum.yaml
bash launch/fit_controller.sh config/exp_cfgs/controller/lqr/pendulum.yaml
bash launch/train_residual.sh config/exp_cfgs/train_residual/pendulum.yaml
bash launch/eval.sh          config/exp_cfgs/eval/pendulum.yaml
```

Stage outputs land under (each gitignored):

| Stage           | Output                                          |
|-----------------|-------------------------------------------------|
| gather_data     | ``data/datasets/<dataset_name>.npz``            |
| train_koopman   | ``train_koopman/weights/<experiment_name>/``    |
| fit_controller  | ``controller/lqr/weights/<output_name>/``       |
| train_residual  | ``train_residual/weights/<experiment_name>/``   |
| eval            | ``eval/results/<results_name>/``                |

## Adding a new experiment

Each stage YAML names its inputs and outputs by string keys:

* ``gather_data_cfg.dataset_name`` → file under ``data/datasets/``
* ``train_koopman_cfg.dataset_name`` + ``.experiment_name``
* ``lqr_controller_cfg.koopman_experiment_name`` + ``.output_name``
* ``train_residual_cfg.koopman_experiment_name`` + ``.lqr_name`` + ``.experiment_name``
* ``eval_cfg.koopman_experiment_name`` + ``.results_name``

Copy the five ``pendulum.yaml`` files under ``config/exp_cfgs/*/`` to a new
experiment name, edit the strings to match, and run ``launch/run_all.sh <name>``.

## Tests

```
bash launch/test.sh
```

The launcher unsets ``PYTHONPATH`` before invoking ``pytest`` to keep a
shell-sourced ROS humble ``PYTHONPATH`` (Python 3.10 site-packages) from
leaking into ``koop_env``. All ``launch/*.sh`` do the same so the pipeline
stages aren't affected either.
