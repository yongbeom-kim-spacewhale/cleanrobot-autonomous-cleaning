#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
if [[ ! -f "$WS_ROOT/install/setup.bash" ]]; then
    echo "ERROR: 먼저 $WS_ROOT/scripts/01_build_yolo.sh를 실행하세요." >&2
    exit 2
fi
source "$WS_ROOT/install/setup.bash"
set -u

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-108}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"

exec ros2 run yolo_pub yolo_pub
