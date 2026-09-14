#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
set +u
source /opt/ros/humble/setup.bash
set -u

cd "$ROS2_WS_ROOT"
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

echo "BUILD_OK: source $ROS2_WS_ROOT/install/setup.bash"
