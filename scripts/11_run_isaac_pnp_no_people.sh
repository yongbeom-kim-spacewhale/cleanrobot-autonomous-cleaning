#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
require_isaac
export STAGE_USD_PATH="$INTEGRATION_ROOT/isaac_sim/Integrated_CleanRobot_Personal_D112/CleanRobot_Park_NoPedestrian.usda"

exec "$INTEGRATION_ROOT/run_isaac_pnp_integrated.sh"
