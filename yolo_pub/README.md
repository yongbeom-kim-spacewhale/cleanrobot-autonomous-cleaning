# yolo_pub

ROS 2 Humble용 YOLO-seg 노드입니다. Isaac Sim 또는 실제 카메라의 RGB-D 영상을 구독해
로봇팔에는 클래스별 XYZ 점군을, AMR에는 화면 좌표와 거리를 JSON으로 발행합니다.

## 구성

```text
yolo_pub.py
  기본 실행 노드
  - 로봇팔: 클래스별 PointCloud2만 발행
  - AMR: 주행 목표 JSON 발행

yolo_pub_with_bbox_center.py
  bbox 중심 JSON이 필요한 이전 호환용 노드
```

## 설치

ROS 2 의존성은 `package.xml`에 있고, pip 의존성은 `requirements.txt`에 있습니다.

```bash
cd ~/cobot3_ws
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/yolo_pub/requirements.txt
PYTHONNOUSERSITE=1 colcon build --packages-select yolo_pub --symlink-install
```

`requirements.txt`에는 `torch`를 넣지 않았습니다. NVIDIA CUDA용 PyTorch는 GPU/CUDA에 맞는
버전을 먼저 설치해야 합니다. 설치 후 다음으로 GPU 인식을 확인합니다.

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 실행

```bash
cd ~/cobot3_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run yolo_pub yolo_pub
```

bbox 중심 JSON까지 필요한 경우에는 아래 실행 항목을 사용합니다.

```bash
ros2 run yolo_pub yolo_pub_with_bbox_center
```

두 실행 항목은 같은 입력/출력 토픽을 공유하므로 동시에 실행하지 않습니다.

## 기본 토픽

| 방향 | 토픽 | 타입 | 내용 |
|---|---|---|---|
| 구독 | `/robot_arm/rgb` | `sensor_msgs/Image` | 로봇팔 RGB |
| 구독 | `/robot_arm/depth` | `sensor_msgs/Image` | 로봇팔 depth (`32FC1`, m) |
| 구독 | `/robot_arm/camera_info` | `sensor_msgs/CameraInfo` | 로봇팔 카메라 내부 파라미터 |
| 구독 | `/tf/robot_arm` | `tf2_msgs/TFMessage` | 로봇팔 카메라 → `base_link` TF |
| 구독 | `/front_stereo_camera/amr/rgb` | `sensor_msgs/Image` | AMR RGB |
| 구독 | `/front_stereo_camera/amr/depth` | `sensor_msgs/Image` | AMR depth (`32FC1`, m) |
| 발행 | `/robot_arm/trash_points/class_0` | `sensor_msgs/PointCloud2` | can XYZ 점군 |
| 발행 | `/robot_arm/trash_points/class_1` | `sensor_msgs/PointCloud2` | paper XYZ 점군 |
| 발행 | `/robot_arm/trash_points/class_2` | `sensor_msgs/PointCloud2` | plastic XYZ 점군 |
| 발행 | `/robot_arm/trash_points/class_3` | `sensor_msgs/PointCloud2` | general_waste XYZ 점군 |
| 발행 | `/amr/trash_target` | `std_msgs/String` | AMR 주행 목표 JSON |

AMR JSON 예시:

```json
{
  "detected": true,
  "bbox_center_x": 324.5,
  "bbox_center_y": 210.0,
  "distance_m": 2.1834
}
```

검출 후보가 없거나 같은 목표가 3회 연속 검출되지 않으면 다음만 발행됩니다.

```json
{"detected": false}
```

## 기본 동작

- 추론·출력 주기: 카메라별 최대 `5 Hz` (`process_interval_sec=0.2`)
- 로봇팔 거리 필터: 마스크 무게중심 주변 `5×5 ∩ segmentation mask` depth 중앙값이 `0.70 m` 이내
- AMR 거리 필터: 기본 `8.0 m` 이내
- AMR 안전화: bbox 중심이 `80 px` 이내인 동일 후보가 `3`회 연속 검출되어야 `detected: true`
- 로봇팔 점군: 객체당 최대 `4000`개 XYZ 점, 가능하면 `base_link` 기준으로 TF 변환

## 파라미터 변경 예시

```bash
ros2 run yolo_pub yolo_pub --ros-args \
  -p arm_max_distance_m:=0.60 \
  -p amr_max_distance_m:=6.0 \
  -p process_interval_sec:=0.2 \
  -p robot_arm_tf_topic:=/tf/robot_arm
```

Isaac Sim의 로봇팔 TF는 `/tf/robot_arm`, AMR TF는 `/tf/amr`처럼 분리하는 것을 권장합니다.
기본 노드는 로봇팔 점군 변환에만 TF가 필요하므로 `/tf/robot_arm`만 구독합니다.
