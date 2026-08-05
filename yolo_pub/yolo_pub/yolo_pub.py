"""로봇팔·AMR RGB-D를 YOLO-seg로 처리하는 ROS 2 노드.

처리 흐름
---------
1. RGB와 depth를 동일한 ROS timestamp로 동기화한다.
2. 로봇팔 영상은 설정한 거리 이내의 객체만 클래스별 XYZ PointCloud2로 발행한다.
3. AMR 영상은 화면 중앙에 가까운 후보를 고르고, 3회 연속 검출되어야 주행 목표로 발행한다.
4. 로봇팔 점군은 가능하면 카메라 좌표계에서 target_frame 좌표계로 TF 변환한다.

디버그 overlay는 구독자가 있을 때만 발행해 평상시 추론 부하를 늘리지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException
from ultralytics import YOLO


def transform_xyz(points: np.ndarray, translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Nx3 카메라 좌표 점들을 ROS TF의 회전(quaternion)·이동으로 변환한다."""
    # Quaternion-vector 회전의 벡터식이다. scipy 의존성 없이 대량 점을 한 번에 처리한다.
    q = quaternion[:3]
    w = quaternion[3]
    twice_cross = 2.0 * np.cross(q, points)
    return points + w * twice_cross + np.cross(q, twice_cross) + translation


def _transform_xyz_self_check() -> None:
    rotated = transform_xyz(np.array([[1.0, 0.0, 0.0]]), np.zeros(3), np.array([0.0, 0.0, 1.0, 0.0]))
    assert np.allclose(rotated, [[-1.0, 0.0, 0.0]])


_transform_xyz_self_check()


class YoloViewer(Node):
    """두 RGB-D 카메라를 구독하여 로봇팔 점군과 AMR 주행 목표를 만드는 단일 노드.
    로봇팔과 AMR은 서로 다른 카메라이므로, timestamp 버퍼·처리 주기·거리 필터를
    별도로 관리한다. 단일 YOLO 모델 인스턴스는 두 입력이 순차적으로 공유한다.
    """

    def __init__(self) -> None:
        super().__init__('yolo_viewer')
        # 설치된 ROS 패키지 resource/의 기본 세그멘테이션 가중치 경로.
        default_model = Path(get_package_share_directory('yolo_pub')) / 'resource' / 'yolo-v8n-best.pt'

        # 런타임 파라미터: launch 파일 또는 ros2 run --ros-args -p 로 재정의 가능하다.
        # 아래 기본값은 현재 Isaac Sim 장면의 ROS 토픽 이름에 맞춘 값이다.
        self.declare_parameter('model_path', str(default_model))
        self.declare_parameter('image_topic', '/robot_arm/rgb')
        self.declare_parameter('depth_topic', '/robot_arm/depth')
        self.declare_parameter('camera_info_topic', '/robot_arm/depth/camera_info')
        # 로봇팔 출력: class_0~3의 접미사가 붙은 PointCloud2 토픽이 생성된다.
        self.declare_parameter('point_cloud_prefix', '/robot_arm/trash_points')
        # 구독자가 있을 때만 YOLO box/mask overlay를 만들어 디버깅한다.
        self.declare_parameter('debug_image_topic', '/robot_arm/yolo/overlay')
        # 로봇팔은 집기 직전 카메라이므로 이 거리보다 먼 객체는 검출 결과 전체에서 제외한다.
        self.declare_parameter('arm_max_distance_m', 2.00)
        # PointCloud2의 기준 frame. 카메라 frame과 다르면 /tf로 이 frame에 맞춘다.
        self.declare_parameter('target_frame', 'map')
        # target_frame까지 이어지는 동적 TF 토픽. 완전한 전역 트리가 필요하면 /tf를 사용한다.
        self.declare_parameter('robot_arm_tf_topic', '/tf')
        # YOLO 공통 추론 설정. imgsz는 YOLO 내부 입력 크기이지 출력 pixel 좌표계가 아니다.
        self.declare_parameter('confidence', 0.4)
        self.declare_parameter('imgsz', 640)
        # 마스크에 포함된 바닥/벤치/그림자 점을 주변 평면으로 제거하는 설정이다.
        self.declare_parameter('remove_support_plane', True)
        self.declare_parameter('support_ring_pixels', 20)
        self.declare_parameter('support_plane_distance_m', 0.015)
        self.declare_parameter('gripper_filter_enabled', True)
        self.declare_parameter('gripper_roi_top_ratio', 0.58)
        self.declare_parameter('gripper_roi_side_ratio', 0.32)
        self.declare_parameter('gripper_overlap_threshold', 0.30)
        # AMR 입력·출력: AMR은 point cloud 대신 화면 좌표와 거리만 수신한다.
        self.declare_parameter('amr_image_topic', '/front_stereo_camera/amr/rgb')
        self.declare_parameter('amr_depth_topic', '/front_stereo_camera/amr/depth')
        self.declare_parameter('amr_target_topic', '/amr/trash_target')
        self.declare_parameter('amr_max_distance_m', 8.0)
        # 로봇팔은 집기 중 최신 점군이 중요하므로 최대 25 Hz로 처리한다.
        self.declare_parameter('arm_process_interval_sec', 0.04)
        # AMR 탐색은 주행 목표 확인용이므로 최대 5 Hz로 제한해 추론 부하를 낮춘다.
        self.declare_parameter('amr_process_interval_sec', 0.20)
        # AMR은 일회성 오검출을 피하기 위해 같은 목표가 이 횟수만큼 연속 확인되어야 한다.
        self.declare_parameter('amr_required_consecutive_frames', 3)
        self.declare_parameter('amr_track_tolerance_px', 80.0)

        model_path = Path(self.get_parameter('model_path').value)
        if not model_path.is_file():
            raise FileNotFoundError(f'YOLO weights not found: {model_path}')
        # 가중치는 한 번만 GPU/CPU 메모리에 적재하여 로봇팔과 AMR 추론이 공유한다.
        self.model = YOLO(str(model_path))
        self.confidence = float(self.get_parameter('confidence').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        # RGB와 depth는 같은 header timestamp끼리만 결합한다.
        # 콜백 도착 순서가 뒤바뀌어도 서로 다른 시각의 RGB-depth가 섞이지 않는다.
        # 처리한 항목은 pop()으로 제거해 버퍼가 계속 커지는 것을 막는다.
        self.rgb_by_stamp, self.depth_by_stamp = {}, {}
        # AMR RGB-D는 로봇팔 영상과 절대 섞이지 않도록 별도 timestamp 버퍼를 쓴다.
        self.amr_rgb_by_stamp, self.amr_depth_by_stamp = {}, {}
        # CameraInfo가 아직 오지 않은 경우에 쓰는 D455 근사 내부 파라미터.
        # CameraInfo를 받으면 info_callback()이 실제 K 행렬 값으로 덮어쓴다.
        self.fx, self.fy, self.cx, self.cy = 554.3307, 569.7543, 320.0, 240.0
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.remove_support_plane = bool(self.get_parameter('remove_support_plane').value)
        self.support_ring_pixels = int(self.get_parameter('support_ring_pixels').value)
        self.support_plane_distance_m = float(self.get_parameter('support_plane_distance_m').value)
        self.gripper_filter_enabled = bool(self.get_parameter('gripper_filter_enabled').value)
        self.gripper_roi_top_ratio = float(self.get_parameter('gripper_roi_top_ratio').value)
        self.gripper_roi_side_ratio = float(self.get_parameter('gripper_roi_side_ratio').value)
        self.gripper_overlap_threshold = float(self.get_parameter('gripper_overlap_threshold').value)
        # 이 거리보다 먼 객체는 로봇팔 검출 결과와 점군에서 제외한다.
        self.arm_max_distance_m = float(self.get_parameter('arm_max_distance_m').value)
        # AMR은 접근 전 탐색용이므로 로봇팔 거리 필터와 독립된 범위를 쓴다.
        self.amr_max_distance_m = float(self.get_parameter('amr_max_distance_m').value)
        # RGB-D 입력은 계속 받되, 각 카메라의 추론과 결과 토픽만 독립 주기로 제한한다.
        self.arm_process_interval_ns = int(float(self.get_parameter('arm_process_interval_sec').value) * 1_000_000_000)
        self.amr_process_interval_ns = int(float(self.get_parameter('amr_process_interval_sec').value) * 1_000_000_000)
        self.last_arm_process_ns: int | None = None
        self.last_amr_process_ns: int | None = None
        self.amr_required_consecutive_frames = int(self.get_parameter('amr_required_consecutive_frames').value)
        self.amr_track_tolerance_px = float(self.get_parameter('amr_track_tolerance_px').value)
        # 직전 AMR 후보의 화면 좌표와 연속 검출 횟수. target 안정화에만 사용한다.
        self.amr_previous_candidate: dict[str, float] | None = None
        self.amr_consecutive_count = 0
        # 고정 seed: 같은 입력에서는 support-plane RANSAC 결과가 재현된다.
        self.plane_rng = np.random.default_rng(0)
        self.last_tf_warning_ns = 0
        # 지정한 동적 TF 토픽과 /tf_static을 합쳐 target_frame 변환에 쓸 TF tree를 만든다.
        self.tf_buffer = Buffer()
        # TF buffer는 child frame별 timestamp가 단조 증가해야 한다. Isaac/ROS 재전송으로
        # 늦게 도착한 과거 TF는 넣지 않아 TF_OLD_DATA 경고와 최신 변환 오염을 막는다.
        self.latest_dynamic_tf_stamp_ns: dict[str, int] = {}
        # /clock이 Isaac Sim reset으로 뒤로 가면 동적 TF는 전부 무효가 된다.
        # 고정 TF는 버퍼 재생성을 위해 별도로 보관한다.
        self.last_clock_ns: int | None = None
        self.static_transforms = {}
        topic = str(self.get_parameter('image_topic').value)
        # Isaac Sim ROS2 Camera Helper의 reliable QoS와 맞춘다.
        # depth=10은 RGB와 depth 콜백 도착 순서가 약간 바뀌어도 짝을 찾기 위한 여유다.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(Image, topic, self.rgb_callback, qos)
        self.depth_subscription = self.create_subscription(Image, str(self.get_parameter('depth_topic').value), self.depth_callback, qos)
        self.info_subscription = self.create_subscription(CameraInfo, str(self.get_parameter('camera_info_topic').value), self.info_callback, qos)
        robot_arm_tf_topic = str(self.get_parameter('robot_arm_tf_topic').value)
        # ROS2 Humble TransformListener는 구독 topic을 바꾸는 인자가 없어 TFMessage를 직접 받는다.
        self.tf_subscription = self.create_subscription(TFMessage, robot_arm_tf_topic, self.robot_arm_tf_callback, 100)
        # 고정 TF는 이미 발행된 메시지도 받아야 하므로 transient local QoS를 사용한다.
        tf_static_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static_subscription = self.create_subscription(TFMessage, '/tf_static', self.robot_arm_tf_static_callback, tf_static_qos)
        self.clock_subscription = self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        amr_image_topic = str(self.get_parameter('amr_image_topic').value)
        self.amr_subscription = self.create_subscription(Image, amr_image_topic, self.amr_rgb_callback, qos)
        self.amr_depth_subscription = self.create_subscription(
            Image, str(self.get_parameter('amr_depth_topic').value), self.amr_depth_callback, qos
        )
        cloud_prefix = str(self.get_parameter('point_cloud_prefix').value).rstrip('/')
        amr_target_topic = str(self.get_parameter('amr_target_topic').value)
        # AMR 담당 측과 합의한 경량 인터페이스: std_msgs/String 안에 JSON 한 개를 넣는다.
        self.amr_target_pub = self.create_publisher(String, amr_target_topic, qos)
        # 클래스별 점군은 XYZ만 담는다. 같은 클래스는 한 화면에 하나라는 운영 조건을 전제로 한다.
        # 현재 학습 모델의 names dict key = class id. 기대값은 0:can, 1:paper, 2:plastic, 3:general_waste.
        self.class_ids = tuple(sorted(self.model.names))
        self.points_pubs = {
            # ROS 2 토픽 token은 숫자로 시작할 수 없으므로 class_ 접두사를 사용한다.
            class_id: self.create_publisher(PointCloud2, f'{cloud_prefix}/class_{class_id}', qos)
            for class_id in self.class_ids
        }
        self.debug_image_pub = self.create_publisher(
            Image, str(self.get_parameter('debug_image_topic').value), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.get_logger().info(
            f'Loaded {model_path.name}; RGB-D={topic}; points={cloud_prefix}/class_{{class_id}} '
            f'in {self.target_frame}; '
            f'TF={robot_arm_tf_topic}; AMR RGB-D={amr_image_topic}, target={amr_target_topic}'
        )

    @staticmethod
    def to_bgr(message: Image) -> np.ndarray:
        """ROS Image의 stride/encoding을 고려해 OpenCV·YOLO용 BGR 배열로 변환한다.
        ROS Image의 ``step``은 한 행의 byte 수라 행 끝 padding을 포함할 수 있다.
        따라서 buffer를 먼저 step 기준으로 펼친 뒤 실제 image 폭만 잘라 사용한다.
        """
        if message.encoding not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            raise ValueError(f'Unsupported image encoding: {message.encoding}')
        channels = 4 if message.encoding in ('rgba8', 'bgra8') else 3
        # step에는 행 끝 padding도 포함될 수 있으므로 width*channels만 잘라 사용한다.
        row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
        # Ultralytics/OpenCV에는 BGR 3채널 배열을 넘긴다. bgr8은 복사 없이 그대로 반환한다.
        if message.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if message.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def stamp(message: Image) -> tuple[int, int]:
        """RGB-depth 쌍을 찾기 위한 (초, 나노초) timestamp 키를 반환한다."""
        return message.header.stamp.sec, message.header.stamp.nanosec

    def robot_arm_tf_callback(self, message: TFMessage) -> None:
        """동적 TF를 등록하되, 늦게 도착한 과거 timestamp는 버린다."""
        for transform in message.transforms:
            stamp = transform.header.stamp
            stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
            child_frame = transform.child_frame_id
            previous_stamp_ns = self.latest_dynamic_tf_stamp_ns.get(child_frame)
            if previous_stamp_ns is not None and stamp_ns <= previous_stamp_ns:
                continue
            self.tf_buffer.set_transform(transform, 'robot_arm_tf')
            self.latest_dynamic_tf_stamp_ns[child_frame] = stamp_ns

    def robot_arm_tf_static_callback(self, message: TFMessage) -> None:
        """/tf_static의 고정 transform을 TF buffer에 등록한다."""
        for transform in message.transforms:
            self.tf_buffer.set_transform_static(transform, 'tf_static')
            self.static_transforms[(transform.header.frame_id, transform.child_frame_id)] = transform

    def clock_callback(self, message: Clock) -> None:
        """Isaac Sim reset으로 시뮬레이션 시간이 되감기면 TF buffer를 새로 만든다."""
        clock = message.clock
        clock_ns = clock.sec * 1_000_000_000 + clock.nanosec
        if self.last_clock_ns is not None and clock_ns < self.last_clock_ns:
            self.tf_buffer = Buffer()
            self.latest_dynamic_tf_stamp_ns.clear()
            for transform in self.static_transforms.values():
                self.tf_buffer.set_transform_static(transform, 'tf_static')
            self.get_logger().info('Isaac Sim time reset detected; TF buffer reinitialized.')
        self.last_clock_ns = clock_ns

    def should_process(self, message: Image, last_attribute: str, interval_ns: int) -> bool:
        """카메라별 추론·출력을 지정한 주기로 제한한다.
        이 함수는 RGB-D 짝을 이미 버퍼에서 제거한 뒤 호출한다. 즉, 생략된 프레임도
        메모리에 남지 않고, 마지막으로 처리한 frame보다 충분히 새 프레임만 YOLO로 간다.
        """
        current_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        previous_ns = getattr(self, last_attribute)
        # Isaac Sim 시간을 reset해 timestamp가 감소하면 새 구간의 첫 프레임은 즉시 허용한다.
        if previous_ns is not None and current_ns >= previous_ns and current_ns - previous_ns < interval_ns:
            return False
        setattr(self, last_attribute, current_ns)
        return True

    def stable_amr_target(self, candidate: dict[str, float] | None) -> dict[str, float] | None:
        """같은 화면상 목표가 연속 검출된 경우에만 AMR 주행 목표로 확정한다.
        동일성은 두 후보의 bbox 중심 pixel 거리로 판단한다. 연속성이 끊기거나 화면에서
        사라지면 count를 즉시 0으로 초기화한다. 기본 5 Hz에서 3회 연속이면 약 0.6초다.
        """
        if candidate is None:
            self.amr_previous_candidate = None
            self.amr_consecutive_count = 0
            return None
        previous = self.amr_previous_candidate
        # np.hypot(dx, dy)는 중심 두 점의 유클리드 pixel 거리를 안정적으로 계산한다.
        if previous is not None and np.hypot(
            candidate['bbox_center_x'] - previous['bbox_center_x'],
            candidate['bbox_center_y'] - previous['bbox_center_y'],
        ) <= self.amr_track_tolerance_px:
            self.amr_consecutive_count += 1
        else:
            self.amr_consecutive_count = 1
        self.amr_previous_candidate = candidate
        return candidate if self.amr_consecutive_count >= self.amr_required_consecutive_frames else None

    def info_callback(self, message: CameraInfo) -> None:
        """CameraInfo의 K 행렬에서 실제 로봇팔 카메라 내부 파라미터를 갱신한다.
        K = [fx, 0, cx; 0, fy, cy; 0, 0, 1] 이며, 이후 depth pixel을 3D로 역투영할 때 쓴다.
        """
        if len(message.k) == 9 and message.k[0] > 0 and message.k[4] > 0:
            self.fx, self.fy, self.cx, self.cy = message.k[0], message.k[4], message.k[2], message.k[5]

    def pixels_to_xyz(self, rows: np.ndarray, cols: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """depth 영상의 픽셀 (u, v)를 카메라 좌표계의 미터 단위 XYZ로 역투영한다."""
        z = depth[rows, cols]
        # pinhole model: X=(u-cx)Z/fx, Y=(v-cy)Z/fy, Z=depth
        return np.column_stack(((cols - self.cx) * z / self.fx, (rows - self.cy) * z / self.fy, z))

    @staticmethod
    def mask_center_depth(depth: np.ndarray, mask: np.ndarray) -> float | None:
        """마스크 무게중심 주변 5×5 내부 depth(최대 25개)의 중앙값을 반환한다.

        단일 중심 pixel은 depth 센서의 hole/noise에 취약하므로 중앙값을 쓴다. 5×5 창의
        바닥 pixel이 섞이지 않도록 세그멘테이션 마스크와의 교집합만 표본으로 사용한다.
        """
        rows, cols = np.where(mask)
        if not len(rows):
            return None
        # 세그멘테이션 마스크의 평균 좌표를 객체 중심으로 사용한다.
        center_row = int(round(float(rows.mean())))
        center_col = int(round(float(cols.mean())))
        # 반경 2 = 중심 포함 5×5. 화면 끝에서는 가능한 픽셀만 사용한다.
        row_start, row_end = max(0, center_row - 2), min(depth.shape[0], center_row + 3)
        col_start, col_end = max(0, center_col - 2), min(depth.shape[1], center_col + 3)
        samples = depth[row_start:row_end, col_start:col_end]
        # 5×5 창 안에서도 세그멘테이션 마스크에 속한 픽셀만 남겨 바닥 depth 혼입을 막는다.
        sample_mask = mask[row_start:row_end, col_start:col_end]
        samples = samples[sample_mask & np.isfinite(samples) & (samples > 0.05) & (samples < 10.0)]
        return float(np.median(samples)) if len(samples) else None

    def support_plane(self, depth: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float] | None:
        """마스크 주위의 바닥·벤치 평면을 RANSAC으로 추정한다.

        예측 마스크 안에 그림자 또는 접촉면이 포함되어도, 전역 Z값이 아닌 해당 물체
        주변 평면과의 거리로 제거하기 위한 함수다.
        """
        # support_ring_pixels=20이면 마스크 외곽 최대 20 px 주변에서 바닥/벤치 표본을 모은다.
        radius = max(1, self.support_ring_pixels)
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        # 객체 바깥의 얇은 ring만 표본으로 쓴다. 객체 자체를 평면으로 오인하지 않는다.
        ring = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool) & ~mask
        ring &= np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
        rows, cols = np.where(ring)
        if len(rows) < 100:
            return None
        if len(rows) > 2000:
            # RANSAC 비용을 일정하게 제한한다. 등간격 표본이면 평면에는 충분하다.
            choose = np.linspace(0, len(rows) - 1, 2000, dtype=int)
            rows, cols = rows[choose], cols[choose]
        samples = self.pixels_to_xyz(rows, cols, depth)
        best_inliers = None
        for _ in range(80):
            # 무작위 3점으로 후보 평면을 만들고, 가장 많은 inlier를 갖는 평면을 고른다.
            a, b, c = samples[self.plane_rng.choice(len(samples), 3, replace=False)]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length < 1e-7:
                continue
            # plane equation: normal · X + offset = 0. normal은 길이 1로 정규화한다.
            normal /= length
            inliers = np.abs(samples @ normal - normal @ a) < self.support_plane_distance_m
            if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_inliers = inliers
        if best_inliers is None or np.count_nonzero(best_inliers) < max(80, len(samples) * 0.4):
            return None
        inlier_points = samples[best_inliers]
        center = inlier_points.mean(axis=0)
        # 최종 inlier 전체를 SVD로 다시 적합해 후보 3점보다 안정적인 법선을 구한다.
        normal = np.linalg.svd(inlier_points - center, full_matrices=False)[2][-1]
        return normal, -float(normal @ center)

    def is_gripper_plastic(self, mask: np.ndarray, class_id: int) -> bool:
        """고정 wrist 카메라에서 그리퍼 영역과 겹친 plastic 오검출인지 판정한다.

        RG2 그리퍼가 카메라 하단 좌우에 고정되어 있고 흰색 plastic과 비슷해 보이므로,
        plastic(class 2) 마스크의 일정 비율이 그 영역에 있으면 결과 전체에서 제외한다.
        """
        if not self.gripper_filter_enabled or class_id != 2:  # class 2 = plastic
            return False
        height, width = mask.shape
        top = int(height * self.gripper_roi_top_ratio)
        side = int(width * self.gripper_roi_side_ratio)
        # 카메라 기준 하단 좌·우에 항상 보이는 RG2 그리퍼의 두 영역을 정의한다.
        gripper_zone = np.zeros_like(mask, dtype=bool)
        gripper_zone[top:, :side] = True
        gripper_zone[top:, width - side:] = True
        return np.count_nonzero(mask & gripper_zone) / max(np.count_nonzero(mask), 1) >= self.gripper_overlap_threshold

    def rgb_callback(self, message: Image) -> None:
        """로봇팔 RGB를 timestamp별로 보관하고, 이미 도착한 depth가 있으면 처리를 시도한다."""
        self.rgb_by_stamp[self.stamp(message)] = message
        self.process_pair(message)

    def depth_callback(self, message: Image) -> None:
        """로봇팔 depth를 timestamp별로 보관하고, 이미 도착한 RGB가 있으면 처리를 시도한다."""
        self.depth_by_stamp[self.stamp(message)] = message
        rgb = self.rgb_by_stamp.get(self.stamp(message))
        if rgb is not None:
            self.process_pair(rgb)

    def amr_rgb_callback(self, message: Image) -> None:
        """AMR RGB를 별도 버퍼에 보관하고, 같은 시각 depth가 있으면 주행 목표를 만든다."""
        self.amr_rgb_by_stamp[self.stamp(message)] = message
        self.process_amr_pair(message)

    def amr_depth_callback(self, message: Image) -> None:
        """AMR depth를 별도 버퍼에 보관하고, 같은 시각 RGB가 있으면 주행 목표를 만든다."""
        self.amr_depth_by_stamp[self.stamp(message)] = message
        rgb = self.amr_rgb_by_stamp.get(self.stamp(message))
        if rgb is not None:
            self.process_amr_pair(rgb)

    def process_amr_pair(self, rgb_message: Image) -> None:
        """AMR RGB-D에서 화면 중앙에 가장 가까운 쓰레기 한 개를 주행 목표로 발행한다.

        결과 JSON은 ``detected``, ``bbox_center_x``, ``bbox_center_y``, ``distance_m``이다.
        여러 객체가 있으면 화면 중앙에 가까운 것이 현재 로봇 진행 방향에 가장 자연스럽다는
        단순 규칙을 적용한다. 클래스와 point cloud는 AMR 제어에 필요하지 않아 발행하지 않는다.
        """
        key = self.stamp(rgb_message)
        depth_message = self.amr_depth_by_stamp.pop(key, None)
        if depth_message is None:
            return
        self.amr_rgb_by_stamp.pop(key, None)
        if not self.should_process(rgb_message, 'last_amr_process_ns', self.amr_process_interval_ns):
            return
        # 첫 두 연속 프레임 또는 후보 없음의 기본값. AMR은 false일 때 목표를 신뢰하면 안 된다.
        target = {'detected': False}
        candidate = None
        try:
            bgr = self.to_bgr(rgb_message)
            depth = np.frombuffer(depth_message.data, dtype=np.float32).reshape(
                depth_message.height, depth_message.step // 4
            )[:, :depth_message.width]
            # retina_masks=True: 원본 RGB 해상도에 맞는 고해상도 마스크를 받아 depth와 직접 맞춘다.
            result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.confidence, retina_masks=True, verbose=False)[0]
            candidates = []
            if result.masks is not None and result.boxes is not None:
                for index, mask in enumerate(result.masks.data.cpu().numpy()):
                    mask = cv2.resize(
                        mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    distance = self.mask_center_depth(depth, mask)
                    # depth 없음 또는 AMR 탐색 범위 밖 객체는 주행 후보에서 제외한다.
                    if distance is None or distance > self.amr_max_distance_m:
                        continue
                    center_x, center_y, _, _ = result.boxes.xywh[index].tolist()
                    # AMR은 현재 진행 방향에서 가장 가까운 화면 중앙 객체 하나만 추적한다.
                    candidates.append((abs(center_x - rgb_message.width / 2.0), center_x, center_y, distance))
            if candidates:
                _, center_x, center_y, distance = min(candidates)
                candidate = {
                    'bbox_center_x': round(center_x, 2),
                    'bbox_center_y': round(center_y, 2),
                    'distance_m': round(distance, 4),
                }
        except Exception as error:
            self.get_logger().error(f'AMR inference failed: {error}')
        # 3회 연속 조건을 통과한 후보만 detected:true가 된다.
        stable_candidate = self.stable_amr_target(candidate)
        if stable_candidate is not None:
            target = {'detected': True, **stable_candidate}
        # 검출이 없을 때도 false를 발행해야 AMR 제어기가 이전 목표로 계속 주행하지 않는다.
        self.amr_target_pub.publish(String(data=json.dumps(target, separators=(',', ':'))))

    def process_pair(self, rgb_message: Image) -> None:
        """로봇팔 RGB-depth 한 쌍으로 추론 후, 클래스별 PointCloud2를 발행한다.

        이 함수는 bbox JSON을 만들지 않는다. 점군 토픽 자체가 class별로 분리되어 있어
        로봇팔은 클래스별 점군의 중심/상단점을 직접 계산해 집기 목표로 사용할 수 있다.
        """
        key = self.stamp(rgb_message)
        # pop으로 처리 완료 프레임을 즉시 제거해 버퍼가 무한히 커지는 것을 막는다.
        depth_message = self.depth_by_stamp.pop(key, None)
        if depth_message is None:
            return
        self.rgb_by_stamp.pop(key, None)
        if not self.should_process(rgb_message, 'last_arm_process_ns', self.arm_process_interval_ns):
            return
        try:
            bgr = self.to_bgr(rgb_message)
            # Isaac depth 토픽은 32FC1(미터)이다. step은 byte 단위이므로 4로 나눈다.
            depth = np.frombuffer(depth_message.data, dtype=np.float32).reshape(depth_message.height, depth_message.step // 4)[:, :depth_message.width]
            # YOLO의 bbox/mask 좌표는 retina_masks=True로 원본 depth 해상도와 맞춰 사용한다.
            result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.confidence, retina_masks=True, verbose=False)[0]
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.cpu().numpy()
                keep = []
                for index, mask in enumerate(masks):
                    # YOLO mask 해상도를 depth 해상도로 맞춘다. nearest로 mask 경계를 보존한다.
                    full_mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    # 마스크 중심 주변 5×5 depth 중앙값으로 거리를 판단한다.
                    # 전체 마스크보다 그림자·바닥이 거리 판정에 섞일 가능성이 작다.
                    median_depth = self.mask_center_depth(depth, full_mask)
                    class_id = int(result.boxes.cls[index].item())
                    # 거리 필터 통과 + 그리퍼 ROI 밖인 결과만 화면/점군 처리 대상으로 남긴다.
                    keep.append(
                        median_depth is not None
                        and median_depth <= self.arm_max_distance_m
                        and not self.is_gripper_plastic(full_mask, class_id)
                    )
                if not all(keep):
                    # 화면 overlay와 아래 점군 생성 모두에서 그리퍼 검출을 완전히 제외한다.
                    result.boxes = result.boxes[keep]
                    result.masks = result.masks[keep]
            # 점군과 같은 거리·그리퍼 필터를 통과한 결과만 디버그 화면에 표시한다.
            if self.debug_image_pub.get_subscription_count():
                debug_overlay = np.ascontiguousarray(result.plot())
                debug_image = Image()
                debug_image.header = rgb_message.header
                debug_image.height, debug_image.width = debug_overlay.shape[:2]
                debug_image.encoding, debug_image.step = 'bgr8', debug_overlay.strides[0]
                debug_image.data = debug_overlay.tobytes()
                self.debug_image_pub.publish(debug_image)
            # 로봇팔에는 클래스별 XYZ 점군만 발행한다. bbox 중심 JSON은 분리 버전에만 있다.
            # class_id를 key로 하는 XYZ list. 점 안에는 class/confidence를 중복 저장하지 않는다.
            points_by_class = {class_id: [] for class_id in self.class_ids}
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.cpu().numpy()
                for index, mask in enumerate(masks):
                    mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    # 로봇팔 거리 필터를 통과한 유효 마스크 depth만 점군으로 만든다.
                    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth <= self.arm_max_distance_m)
                    rows, cols = np.where(valid)
                    # 해당 mask pixel만 pinhole 모델로 카메라 frame XYZ(m)로 변환한다.
                    xyz = self.pixels_to_xyz(rows, cols, depth)
                    if self.remove_support_plane:
                        plane = self.support_plane(depth, mask)
                        if plane is not None:
                            normal, offset = plane
                            # 추정 평면에 너무 가까운 점은 그림자/바닥/벤치 접촉면으로 본다.
                            keep = np.abs(xyz @ normal + offset) > self.support_plane_distance_m
                            rows, cols, xyz = rows[keep], cols[keep], xyz[keep]
                    if len(rows) > 4000:
                        # 객체 하나당 최대 4000점으로 고정해 대역폭과 RViz 부하를 제한한다.
                        choose = np.linspace(0, len(rows) - 1, 4000, dtype=int)
                        rows, cols, xyz = rows[choose], cols[choose], xyz[choose]
                    if not len(xyz):
                        continue
                    class_id = int(result.boxes.cls[index].item())
                    points_by_class.setdefault(class_id, []).extend((float(x), float(y), float(z)) for x, y, z in xyz)
            # TF 변환에 실패하면 안전하게 원래 카메라 frame으로 발행한다.
            cloud_header = rgb_message.header
            if any(points_by_class.values()) and rgb_message.header.frame_id != self.target_frame:
                # 점군은 camera frame에서 만들어졌으므로 TF로 로봇 base_link 등 기준 프레임으로 옮긴다.
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.target_frame, rgb_message.header.frame_id, Time.from_msg(rgb_message.header.stamp)
                    ).transform
                    # TF의 translation + quaternion을 numpy 배열로 옮겨 대량 점 변환에 사용한다.
                    translation = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
                    quaternion = np.array([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w])
                    # 클래스별 점군을 한 번씩 일괄 변환한다. 점에는 XYZ 이외 필드를 넣지 않는다.
                    for class_id, points in points_by_class.items():
                        if points:
                            xyz = transform_xyz(np.asarray(points), translation, quaternion)
                            points_by_class[class_id] = [tuple(map(float, point)) for point in xyz]
                    cloud_header = Header(stamp=rgb_message.header.stamp, frame_id=self.target_frame)
                except TransformException as error:
                    now_ns = self.get_clock().now().nanoseconds
                    # TF가 잠깐 늦더라도 2초마다 한 번만 경고해 터미널 로그 폭주를 방지한다.
                    if now_ns - self.last_tf_warning_ns > 2_000_000_000:
                        self.get_logger().warning(f'TF unavailable ({rgb_message.header.frame_id} → {self.target_frame}): {error}')
                        self.last_tf_warning_ns = now_ns
            # RViz에서 클래스별 토픽을 따로 표시한다. 빈 점군도 보내 이전 프레임의 잔상을 지운다.
            for class_id, publisher in self.points_pubs.items():
                publisher.publish(point_cloud2.create_cloud_xyz32(cloud_header, points_by_class.get(class_id, [])))
        except Exception as error:
            self.get_logger().error(f'Inference failed: {error}')

def main() -> None:
    """ROS 2 노드의 생성·spin·정상 종료 진입점."""
    rclpy.init()
    node = YoloViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
