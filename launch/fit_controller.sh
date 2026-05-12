#!/usr/bin/env bash
# Fit the LQR controller on a saved Koopman model and run stability analysis.
# Usage: bash launch/fit_controller.sh config/exp_cfgs/controller/lqr/pendulum.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
python -m controller.lqr --config "$@"
