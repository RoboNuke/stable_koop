#!/usr/bin/env bash
# Run the full stable_koop pipeline end-to-end for one experiment name.
#
# Expects per-stage YAMLs under config/exp_cfgs/<stage>/<experiment>.yaml for
# every stage. Pass the experiment name as the only argument.
#
# Usage: bash launch/run_all.sh pendulum
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="${1:-pendulum}"
echo "[run_all] experiment=$EXP_NAME"

bash launch/gather_data.sh "config/exp_cfgs/gather_data/${EXP_NAME}.yaml"
bash launch/train_koopman.sh "config/exp_cfgs/train_koopman/${EXP_NAME}.yaml"
bash launch/fit_controller.sh "config/exp_cfgs/controller/lqr/${EXP_NAME}.yaml"
bash launch/train_residual.sh "config/exp_cfgs/train_residual/${EXP_NAME}.yaml"
bash launch/eval.sh "config/exp_cfgs/eval/${EXP_NAME}.yaml"

echo "[run_all] done."
