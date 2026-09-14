#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
source_ros

exec python3 "$INTEGRATION_ROOT/run_wp2_manipulator_home_test.py" --control-only
