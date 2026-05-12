#!/usr/bin/env bash
# Run evaluation stages enabled by the EvalCfg.
# Usage: bash launch/eval.sh config/exp_cfgs/eval/pendulum.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
python -m eval --config "$@"
