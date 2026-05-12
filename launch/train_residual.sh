#!/usr/bin/env bash
# Train the SAC residual policy.
# Usage: bash launch/train_residual.sh config/exp_cfgs/train_residual/pendulum.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
python -m train_residual --config "$@"
