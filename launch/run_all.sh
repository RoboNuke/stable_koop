#!/usr/bin/env bash
# Run the full stable_koop pipeline end-to-end for one experiment name.
#
# Expects per-stage YAMLs under config/exp_cfgs/<stage>/<experiment>.yaml for
# every stage. Pass the experiment name as the only argument.
#
# Usage: bash launch/run_all.sh pendulum [--no-residual]
#   --no-residual   Skip the train_residual stage.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

EXP_NAME="pendulum"
RUN_RESIDUAL=1
for arg in "$@"; do
    case "$arg" in
        --no-residual) RUN_RESIDUAL=0 ;;
        -*) echo "[run_all] unknown flag: $arg" >&2; exit 2 ;;
        *) EXP_NAME="$arg" ;;
    esac
done
echo "[run_all] experiment=$EXP_NAME residual=$RUN_RESIDUAL"

bash launch/gather_data.sh "config/exp_cfgs/gather_data/${EXP_NAME}.yaml" --both
bash launch/train_koopman.sh "config/exp_cfgs/train_koopman/lqr/${EXP_NAME}.yaml"
if [[ "$RUN_RESIDUAL" -eq 1 ]]; then
    bash launch/train_residual.sh "config/exp_cfgs/train_residual/${EXP_NAME}.yaml"
else
    echo "[run_all] skipping train_residual (--no-residual)"
fi
bash launch/eval.sh "config/exp_cfgs/eval/${EXP_NAME}.yaml"

echo "[run_all] done."
