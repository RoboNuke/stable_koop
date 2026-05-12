#!/usr/bin/env bash
# Run the unit test suite.
# Usage: bash launch/test.sh [extra pytest args...]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
source /home/hunter/miniconda3/bin/activate koop_env
unset PYTHONPATH  # keep ROS humble site-packages out of koop_env (Python 3.10 vs 3.11)
python -m pytest tests/ "$@"
