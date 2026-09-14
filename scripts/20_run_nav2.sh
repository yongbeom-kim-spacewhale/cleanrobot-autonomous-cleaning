#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"
source_ros

exec ros2 launch cleanrobot_amr_navigation cleanrobot_amr_bringup.launch.py
