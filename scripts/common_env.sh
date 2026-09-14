#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS2_WS_ROOT="$PROJECT_ROOT/ros2_ws"
INTEGRATION_ROOT="$PROJECT_ROOT/simulator/cleanrobot_amr_d112/manipulator_integration"

export CLEANROBOT_PROJECT_ROOT="$PROJECT_ROOT"
export CLEANROBOT_INTEGRATION_ROOT="$INTEGRATION_ROOT"
export COBOT3_WS_ROOT="$INTEGRATION_ROOT/cobot3_ws"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-108}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

require_isaac() {
    if [[ -z "${ISAAC_SIM_ROOT:-}" ]]; then
        echo "ERROR: export ISAAC_SIM_ROOT=/path/to/isaac_sim/release" >&2
        exit 2
    fi
    if [[ ! -x "$ISAAC_SIM_ROOT/python.sh" ]]; then
        echo "ERROR: $ISAAC_SIM_ROOT/python.sh를 찾을 수 없습니다." >&2
        exit 2
    fi
}

source_ros() {
    set +u
    source /opt/ros/humble/setup.bash
    if [[ ! -f "$ROS2_WS_ROOT/install/setup.bash" ]]; then
        echo "ERROR: ROS 2 워크스페이스가 빌드되지 않았습니다." >&2
        echo "먼저 $PROJECT_ROOT/scripts/01_build_ros2.sh를 실행하세요." >&2
        exit 2
    fi
    source "$ROS2_WS_ROOT/install/setup.bash"
    set -u
}
