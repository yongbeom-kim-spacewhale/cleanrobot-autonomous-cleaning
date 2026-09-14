#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

set +u
source /opt/ros/humble/setup.bash
set -u

if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: GPU·CUDA 환경에 맞는 PyTorch를 먼저 설치하세요." >&2
    exit 2
fi

python3 -m pip install -r "$WS_ROOT/src/yolo_pub/requirements.txt"
rosdep install --from-paths "$WS_ROOT/src" --ignore-src -r -y
cd "$WS_ROOT"
PYTHONNOUSERSITE=1 colcon build --packages-select yolo_pub --symlink-install

echo "BUILD_OK: source $WS_ROOT/install/setup.bash"
