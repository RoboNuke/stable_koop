#!/usr/bin/env bash
# Fit the LQR controller on a saved Koopman model and run stability analysis.
# Usage: bash launch/fit_controller.sh config/exp_cfgs/controller/lqr/pendulum.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
unset PYTHONPATH  # keep ROS humble site-packages out of koop_env (Python 3.10 vs 3.11)
python -m controller.lqr --config "$@"
