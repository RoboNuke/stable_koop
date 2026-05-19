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

This runs ``gather_data → train_koopman (+ controller fit) → train_residual → eval``
in sequence. Each stage uses ``config/exp_cfgs/<stage>/pendulum.yaml``,
except training, which uses the combined
``config/exp_cfgs/train_koopman/<controller_type>/pendulum.yaml`` file that
carries both the Koopman training and controller-fit headers.

## Running individual stages

Each stage is independently runnable against the previous stage's saved
outputs:

```
bash launch/gather_data.sh   config/exp_cfgs/gather_data/pendulum.yaml --both
bash launch/train_koopman.sh config/exp_cfgs/train_koopman/lqr/pendulum.yaml
bash launch/train_residual.sh config/exp_cfgs/train_residual/pendulum.yaml
bash launch/eval.sh          config/exp_cfgs/eval/pendulum.yaml
```

``train_koopman`` accepts two extra flags:

* ``--skip_train`` — skip Koopman training and fit only the controller
  against an already-saved model. The script prompts interactively for a
  fresh controller experiment name so the new fit lands in a new subdir
  under ``<koopman_dir>/lqr/``.
* ``--koopman_path <path>`` — override the model checkpoint location.
  Accepts a directory (containing ``koopman_ckpt.pt``) or the ``.pt``
  file directly. When absent, the controller cfg's ``koopman_experiment_name``
  field is used (``results/<name>/``).

Stage outputs land under (each gitignored):

| Stage             | Output                                                            |
|-------------------|-------------------------------------------------------------------|
| gather_data       | ``data/datasets/<dataset_name>.npz``                              |
| train_koopman     | ``results/<experiment_name>/`` (ckpt, config.yaml, model_performance.yaml) |
| (controller fit)  | ``results/<koopman_exp>/lqr/<output_name>/`` (lqr.pt, config.yaml, ctrl_performance.yaml) |
| train_residual    | ``train_residual/weights/<experiment_name>/``                     |
| eval              | ``eval/results/<results_name>/``                                  |

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
