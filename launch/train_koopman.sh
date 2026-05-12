#!/usr/bin/env bash
# Run Koopman training against a per-stage YAML.
# Usage: bash launch/train_koopman.sh config/exp_cfgs/train_koopman/pendulum.yaml
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
python -m train_koopman --config "$@"
