#!/usr/bin/env bash
# Run the gather-data stage against a per-stage YAML.
# Usage: bash launch/gather_data.sh config/exp_cfgs/gather_data/pendulum.yaml [extra args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
python -m data.gather_data --config "$@"
