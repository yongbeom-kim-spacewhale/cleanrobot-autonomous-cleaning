# yolo_pub

ROS 2 Humble에서 RGB-D 영상에 YOLO segmentation을 적용하는 노드다. 로봇팔 카메라에서는
클래스별 3D 점군을, AMR 카메라에서는 주행 후보의 화면 좌표와 거리를 JSON으로 발행한다.

## 요구 사항

- Ubuntu 22.04, ROS 2 Humble
- Python 3.10
- NVIDIA GPU 사용 시 해당 GPU/CUDA 조합에 맞는 PyTorch
- RGB, depth, CameraInfo, TF를 발행하는 카메라 또는 Isaac Sim ROS 2 Bridge

ROS 패키지 의존성은 `package.xml`, Python 의존성은 `requirements.txt`에 있다. PyTorch는
CUDA 조합마다 설치 방법이 달라 requirements에 고정하지 않았다. 먼저 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에서 맞는 명령으로 설치한다.

```bash
# <workspace>는 이 패키지의 src/를 포함하는 ROS 2 작업공간이다.
cd <workspace>
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/yolo_pub/requirements.txt
PYTHONNOUSERSITE=1 colcon build --packages-select yolo_pub --symlink-install
source install/setup.bash
```

GPU 인식 확인:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()`가 `False`이면 CPU로 실행되며 추론이 크게 느려질 수 있다.

## 실행

기본 가중치는 패키지의 `resource/yolo-v8n-best.pt`에서 찾으므로 사용자 홈 경로에 의존하지 않는다.

```bash
cd <workspace>
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run yolo_pub yolo_pub
```

다른 가중치를 쓸 때는 절대경로 대신 현재 작업공간 기준 경로도 줄 수 있다.

```bash
ros2 run yolo_pub yolo_pub --ros-args \
  -p model_path:="$PWD/models/best.pt"
```

## 기본 토픽

| 방향 | 기본 토픽 | 타입 | 설명 |
|---|---|---|---|
| 구독 | `/robot_arm/rgb` | `sensor_msgs/Image` | 로봇팔 RGB (`rgb8`, `bgr8`, `rgba8`, `bgra8`) |
| 구독 | `/robot_arm/depth` | `sensor_msgs/Image` | 로봇팔 depth (`32FC1`, 미터) |
| 구독 | `/robot_arm/depth/camera_info` | `sensor_msgs/CameraInfo` | depth 영상에 맞는 카메라 내부 파라미터 |
| 구독 | `/tf` | `tf2_msgs/TFMessage` | 동적 TF. `target_frame`에서 카메라 frame까지 경로가 있어야 한다. |
| 구독 | `/tf_static` | `tf2_msgs/TFMessage` | 고정 TF. 늦게 실행해도 받을 수 있도록 transient-local QoS를 쓴다. |
| 구독 | `/clock` | `rosgraph_msgs/Clock` | Isaac Sim reset 감지용. 시간이 되감기면 TF 버퍼를 자동 초기화한다. |
| 구독 | `/front_stereo_camera/amr/rgb` | `sensor_msgs/Image` | AMR RGB |
| 구독 | `/front_stereo_camera/amr/depth` | `sensor_msgs/Image` | AMR depth (`32FC1`, 미터) |
| 발행 | `/robot_arm/trash_points/class_<id>` | `sensor_msgs/PointCloud2` | 클래스별 XYZ 점군 |
| 발행 | `/robot_arm/yolo/overlay` | `sensor_msgs/Image` | 구독자가 있을 때만 만든 YOLO 디버그 영상 |
| 발행 | `/amr/trash_target` | `std_msgs/String` | AMR 주행 목표 JSON |

모델의 클래스 ID가 그대로 `class_<id>` 토픽 접미사가 된다. 포함된 가중치의 클래스는
`0: can`, `1: paper`, `2: plastic`, `3: general_waste`다.

AMR 목표 예시:

```json
{"detected":true,"bbox_center_x":324.5,"bbox_center_y":210.0,"distance_m":2.1834}
```

후보가 없거나 같은 후보가 연속 프레임 조건을 만족하지 않으면 `{"detected":false}`를 발행한다.

## 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---:|---|
| `model_path` | 패키지 내 `resource/yolo-v8n-best.pt` | YOLO segmentation 가중치 경로 |
| `image_topic`, `depth_topic`, `camera_info_topic` | 로봇팔 토픽 | 로봇팔 RGB-D 입력 |
| `point_cloud_prefix` | `/robot_arm/trash_points` | 클래스별 점군 토픽 접두사 |
| `target_frame` | `map` | 발행 점군 좌표계. 집기만 하면 보통 `base_link`가 편하다. |
| `robot_arm_tf_topic` | `/tf` | 동적 TF 토픽 |
| `arm_max_distance_m` | `2.0` | 로봇팔 검출과 점군의 최대 depth 거리(m) |
| `arm_process_interval_sec` | `0.04` | 로봇팔 최대 처리 주기(25 Hz) |
| `confidence`, `imgsz` | `0.4`, `640` | YOLO confidence threshold와 입력 크기 |
| `debug_image_topic` | `/robot_arm/yolo/overlay` | 디버그 overlay 출력 토픽 |
| `amr_image_topic`, `amr_depth_topic` | AMR 토픽 | AMR RGB-D 입력 |
| `amr_max_distance_m` | `8.0` | AMR 탐색 거리(m) |
| `amr_process_interval_sec` | `0.20` | AMR 최대 처리 주기(5 Hz) |
| `amr_required_consecutive_frames` | `3` | AMR 목표 확정에 필요한 연속 검출 횟수 |

예시: 로봇팔은 `base_link` 기준 점군으로, AMR은 5 Hz로 실행한다.

```bash
ros2 run yolo_pub yolo_pub --ros-args \
  -p target_frame:=base_link \
  -p arm_max_distance_m:=1.5 \
  -p arm_process_interval_sec:=0.04 \
  -p amr_process_interval_sec:=0.20
```

## TF와 Isaac Sim

`target_frame:=map`을 쓸 때는 카메라 frame까지 하나의 TF tree로 이어져야 한다. 일반적인
구성은 `map → odom → tf_amr → chassis_link → base_link → … → camera_optical_frame`이다.
`map → odom`은 Nav2의 AMCL/SLAM 등 localization이 발행해야 하며, Isaac Stage 안의 map Prim만으로는 만들어지지 않는다.

Isaac Sim이 `/clock`을 발행하면 Stop/Reset 뒤 Play했을 때 시간이 0으로 되감기는 것을
자동 감지해 TF 버퍼를 다시 만든다. `/clock`을 쓰지 않는 실제 로봇에서는 이 구독은 아무 동작도 하지 않는다.

TF 연결 상태는 다음으로 확인한다.

```bash
ros2 run tf2_ros tf2_echo map cobot3_wrist_color_optical_frame
```

`target_frame:=base_link`를 사용한다면 위 명령의 `map`을 `base_link`로 바꾼다.

## 디버깅

YOLO overlay는 image_view로 볼 수 있다.

```bash
ros2 run image_view image_view --ros-args -r image:=/robot_arm/yolo/overlay
```

추론 시간, TF 오류, AMR 오류는 `yolo_pub` 터미널 로그에 출력된다. Isaac Sim을 reset한 뒤에는
`Isaac Sim time reset detected; TF buffer reinitialized.` 한 줄이 출력되는 것이 정상이다.
