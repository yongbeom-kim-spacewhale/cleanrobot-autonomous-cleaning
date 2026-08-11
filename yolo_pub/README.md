# yolo_pub

ROS 2 Humble용 RGB-D YOLO-seg 추론 패키지다. 하나의 노드가 로봇팔과 AMR 카메라를
각각 구독하여, 로봇팔에는 클래스별 3D 점군을, AMR에는 주행 목표 JSON을 발행.

## 제출 범위

이 폴더는 과제 목록 중 **ROS 2 패키지**에 해당한다.

- 포함: `yolo_pub` ROS 2 Python 패키지, 학습 가중치(`resource/*.pt`), Python 의존성 명세
- 미포함: Isaac Sim USD/URDF 에셋, Isaac Sim standalone 실행 코드, 시뮬레이션 메인 USD

## 운영체제·개발 환경

| 항목 | 기준 |
|---|---|
| OS | Ubuntu 22.04 |
| ROS | ROS 2 Humble |
| Python | 3.10 |
| GPU | CUDA를 지원하는 NVIDIA GPU 권장 (개발·검증: RTX 5080) |
| 입력 카메라 | Intel RealSense D455와 호환되는 RGB-D 토픽 또는 Isaac Sim ROS 2 Bridge |

CPU에서도 실행할 수 있으나 YOLO-seg 및 point cloud 처리 속도가 크게 낮아질 수 있다.

## 의존성

### ROS 2 의존성

`package.xml`에 다음 ROS 의존성이 선언되어 있다.

```text
rclpy, rosgraph_msgs, ament_index_python, sensor_msgs, sensor_msgs_py,
std_msgs, tf2_ros, tf2_msgs
```

### Python 의존성

`requirements.txt`에 다음 Python 패키지가 명시되어 있다.

```text
ultralytics, numpy, opencv-python
```

PyTorch는 GPU·CUDA·드라이버 조합에 따라 설치 명령이 달라서 `requirements.txt`에 고정하지 않음.
CUDA 지원 PyTorch를 먼저 설치한 뒤 나머지 의존성 파일 설치.

```bash
# CUDA 환경에 맞는 PyTorch 설치 명령은 공식 안내에서 선택한다.
# https://pytorch.org/get-started/locally/

python3 -m pip install -r src/yolo_pub/requirements.txt
```

GPU 인식 확인:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 빌드 및 실행

`<workspace>`는 `src/yolo_pub`를 포함하는 ROS 2 작업공간.

```bash
cd <workspace>
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
python3 -m pip install -r src/yolo_pub/requirements.txt
PYTHONNOUSERSITE=1 colcon build --packages-select yolo_pub --symlink-install
source install/setup.bash
ros2 run yolo_pub yolo_pub
```

`PYTHONNOUSERSITE=1`은 빌드 시에만 사용. 실행할 때는 CUDA PyTorch와 Ultralytics가 설치된
사용자 Python 패키지 경로가 필요하므로 지정하지 않음.

기본 가중치는 패키지 내부의 다음 파일을 사용.

```text
resource/finetune_v8n_best.pt
```

## ROS 2 인터페이스

### 구독 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/robot_arm/rgb` | `sensor_msgs/Image` | 로봇팔 RGB (`rgb8`, `bgr8`, `rgba8`, `bgra8`) |
| `/robot_arm/depth` | `sensor_msgs/Image` | 로봇팔 depth (`32FC1`, m) |
| `/cobot3/wrist_camera/depth/camera_info` | `sensor_msgs/CameraInfo` | depth 카메라 내부 파라미터 |
| `/front_stereo_camera/amr/rgb` | `sensor_msgs/Image` | AMR 전방 RGB |
| `/front_stereo_camera/amr/depth` | `sensor_msgs/Image` | AMR 전방 depth (`32FC1`, m) |
| `/tf` | `tf2_msgs/TFMessage` | 로봇팔 점군 좌표 변환용 동적 TF |
| `/tf_static` | `tf2_msgs/TFMessage` | 고정 TF |
| `/clock` | `rosgraph_msgs/Clock` | Isaac Sim reset 감지용. 없으면 카메라 timestamp를 그대로 처리 |

RGB와 depth는 **동일한 ROS timestamp**를 가진 쌍만 추론에 사용한다.

### 발행 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/robot_arm/trash_points/class_0` | `sensor_msgs/PointCloud2` | can 객체 XYZ 점군 |
| `/robot_arm/trash_points/class_1` | `sensor_msgs/PointCloud2` | paper 객체 XYZ 점군 |
| `/robot_arm/trash_points/class_2` | `sensor_msgs/PointCloud2` | plastic 객체 XYZ 점군 |
| `/robot_arm/trash_points/class_3` | `sensor_msgs/PointCloud2` | general_waste 객체 XYZ 점군 |
| `/robot_arm/yolo/overlay` | `sensor_msgs/Image` | 로봇팔 YOLO 결과 오버레이 |
| `/amr/yolo/overlay` | `sensor_msgs/Image` | AMR YOLO 결과 오버레이 |
| `/amr/trash_target` | `std_msgs/String` | AMR 주행 목표 JSON |

AMR 목표 메시지 예시:

```json
{"detected":true,"bbox_center_x":324.5,"bbox_center_y":210.0,"distance_m":2.1834}
```

후보가 없거나 동일 후보가 3회 연속으로 검출되지 않으면 다음을 발행한다.

```json
{"detected":false}
```

## 현재 동작 기준

- 클래스: `0: can`, `1: paper`, `2: plastic`, `3: general_waste`
- YOLO confidence: `0.7`, 입력 크기: `640`
- 로봇팔: 최대 `25 Hz`, 거리 `2.0 m` 이내 객체의 점군 발행
- AMR: 최대 `12.5 Hz`, 거리 `8.0 m` 이내 후보 중 화면 중앙 객체 선택
- AMR 목표: 화면상 중심 차이가 `80 px` 이내인 동일 후보를 3회 연속 확인한 뒤 발행
- 로봇팔 점군 target frame: `map` (TF가 없으면 depth camera frame 기준으로 발행)

## 확인 방법

실행 후 AMR 목표 메시지 확인:

```bash
ros2 topic echo /amr/trash_target
```

AMR 카메라 입력 확인:

```bash
ros2 topic info /front_stereo_camera/amr/rgb
ros2 topic info /front_stereo_camera/amr/depth
```

YOLO 오버레이 확인:

```bash
ros2 run image_view image_view --ros-args -r image:=/amr/yolo/overlay
```

Isaac Sim이 정지 상태이면 카메라 publisher가 없으므로, 씬을 Play한 뒤 토픽을 확인해야 한다.
