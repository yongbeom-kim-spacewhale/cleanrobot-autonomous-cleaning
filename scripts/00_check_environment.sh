#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_env.sh"

require_isaac
set +u
source /opt/ros/humble/setup.bash
set -u

required=(
    "$INTEGRATION_ROOT/run_isaac_pnp_integrated.sh"
    "$INTEGRATION_ROOT/run_wp2_manipulator_home_test.py"
    "$INTEGRATION_ROOT/isaac_sim/Integrated_CleanRobot_Personal_D112/CleanRobot_Park.usda"
    "$INTEGRATION_ROOT/isaac_sim/Integrated_CleanRobot_Personal_D112/CleanRobot_Park_NoPedestrian.usda"
    "$ROS2_WS_ROOT/src/cleanrobot_amr_navigation/package.xml"
    "$PROJECT_ROOT/external_vision_ws/src/yolo_pub/package.xml"
    "$PROJECT_ROOT/external_vision_ws/src/yolo_pub/resource/finetune_v8n_best.pt"
)
for path in "${required[@]}"; do
    [[ -f "$path" ]] || { echo "MISSING: $path" >&2; exit 1; }
done

for package in nav2_bringup nav2_simple_commander rmw_fastrtps_cpp rviz2; do
    ros2 pkg prefix "$package" >/dev/null
done

echo "CHECK_OK: Ubuntu=$(lsb_release -ds)"
echo "CHECK_OK: ROS_DISTRO=${ROS_DISTRO:-unknown}"
echo "CHECK_OK: ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "CHECK_OK: RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "CHECK_OK: ISAAC_SIM_ROOT=$ISAAC_SIM_ROOT"
echo "CHECK_OK: 필수 파일 및 ROS 패키지 확인 완료"
