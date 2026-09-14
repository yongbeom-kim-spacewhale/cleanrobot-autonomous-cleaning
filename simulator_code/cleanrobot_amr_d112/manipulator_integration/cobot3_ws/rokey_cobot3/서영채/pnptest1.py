import os
import sys
from pathlib import Path


# 현재 AMR/비전/Nav2가 사용하는 Domain 108로 Isaac/ROS 초기화 전에 맞춘다.
ROS_DOMAIN_ID = "108"
os.environ["ROS_DOMAIN_ID"] = ROS_DOMAIN_ID
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_COBOT3_WS_ROOT = SCRIPT_PATH.parents[2]
COBOT3_WS_ROOT = Path(
    os.environ.get("COBOT3_WS_ROOT", str(DEFAULT_COBOT3_WS_ROOT))
).expanduser().resolve()

integration_root_value = os.environ.get("CLEANROBOT_INTEGRATION_ROOT")
if not integration_root_value:
    raise RuntimeError(
        "CLEANROBOT_INTEGRATION_ROOT 환경변수가 필요합니다. "
        "압축을 푼 배포본의 manipulator_integration 폴더를 지정하세요."
    )
CLEANROBOT_INTEGRATION_ROOT = Path(
    integration_root_value
).expanduser().resolve()

STAGE_USD_PATH = os.environ.get(
    "STAGE_USD_PATH",
    str(
        CLEANROBOT_INTEGRATION_ROOT
        / "isaac_sim"
        / "Integrated_CleanRobot_Personal_D112"
        / "CleanRobot_Park.usda"
    ),
)
RMPFLOW_DIR = str(COBOT3_WS_ROOT / "isaacpjt" / "M0609" / "rmpflow")
M0609_URDF_PATH = str(
    COBOT3_WS_ROOT
    / "isaacpjt"
    / "M0609"
    / "doosan-robot2"
    / "urdf"
    / "m0609_isaac_sim.urdf"
)
M0609_DESCRIPTION_PATH = str(
    COBOT3_WS_ROOT
    / "isaacpjt"
    / "M0609"
    / "rmpflow"
    / "m0609_description.yaml"
)
M0609_RMPFLOW_CONFIG_PATH = str(
    COBOT3_WS_ROOT
    / "rokey_cobot3"
    / "서영채"
    / "m0609_rmpflow_slow.yaml"
)

required_files = {
    "통합 USD": STAGE_USD_PATH,
    "M0609 URDF": M0609_URDF_PATH,
    "M0609 description": M0609_DESCRIPTION_PATH,
    "M0609 RMPFlow config": M0609_RMPFLOW_CONFIG_PATH,
    "M0609 RMPFlow controller": str(
        Path(RMPFLOW_DIR) / "m0609_rmpflow_controller.py"
    ),
}
missing_files = {
    label: path
    for label, path in required_files.items()
    if not Path(path).is_file()
}
if missing_files:
    missing_text = "\n".join(
        f"  {label}: {path}" for label, path in missing_files.items()
    )
    raise FileNotFoundError(
        "통합 실행 필수 파일을 찾지 못했습니다:\n" + missing_text
    )

print(
    "[PORTABLE PATHS]\n"
    f"  ROS_DOMAIN_ID={ROS_DOMAIN_ID}\n"
    f"  CLEANROBOT_INTEGRATION_ROOT={CLEANROBOT_INTEGRATION_ROOT}\n"
    f"  COBOT3_WS_ROOT={COBOT3_WS_ROOT}\n"
    f"  STAGE_USD_PATH={STAGE_USD_PATH}"
)

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "create_new_stage": False,
    }
)

from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import open_stage, is_stage_loading

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import struct
import time
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import omni.timeline
import omni.usd
import rclpy
from pxr import Usd, UsdPhysics
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool

from isaacsim.core.api import World
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.prims import (
    SingleGeometryPrim,
    SingleXFormPrim,
)
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator


if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_rmpflow_controller import RMPFlowController


# Isaac Sim 모듈과 확장을 불러온 뒤 통합 Stage를 연다.
if not open_stage(STAGE_USD_PATH):
    raise RuntimeError(f"Stage를 열지 못했습니다: {STAGE_USD_PATH}")

while is_stage_loading():
    simulation_app.update()

for _ in range(5):
    simulation_app.update()

loaded_stage = omni.usd.get_context().get_stage()
print(f"[STAGE] {loaded_stage.GetRootLayer().identifier}")


# ---------------------------------------------------------------------
# Prim 경로
# ---------------------------------------------------------------------

CLEAN_ROBOT_PRIM_PATH = "/World/CleanRobot"
ARTICULATION_PRIM_PATH = f"{CLEAN_ROBOT_PRIM_PATH}/chassis_link"
MANIPULATOR_PRIM_PATH = f"{CLEAN_ROBOT_PRIM_PATH}/Manipulator/M0609"
RMPFLOW_BASE_PRIM_PATH = f"{MANIPULATOR_PRIM_PATH}/base_link"
PHYSICS_EE_PRIM_PATH = f"{MANIPULATOR_PRIM_PATH}/link_6"
TCP_PRIM_PATH = (
    f"{MANIPULATOR_PRIM_PATH}/"
    "Gripper/angle_bracket/grasp_tcp"
)
WRIST_DEPTH_CAMERA_PRIM_PATH = (
    f"{MANIPULATOR_PRIM_PATH}/Gripper/angle_bracket/"
    "WristD455/RSD455/Camera_Pseudo_Depth"
)
WASTEBINS_PRIM_PATH = f"{ARTICULATION_PRIM_PATH}/WasteBins"
LEFT_INNER_FINGER_PRIM_PATH = (
    f"{MANIPULATOR_PRIM_PATH}/Gripper/left_inner_finger"
)
RIGHT_INNER_FINGER_PRIM_PATH = (
    f"{MANIPULATOR_PRIM_PATH}/Gripper/right_inner_finger"
)
PEDESTRIAN_PRIM_PATH = "/World/Characters/Nathan_DynamicRoot"

RMPFLOW_EE_LINK_NAME = "grasp_tcp"


# ---------------------------------------------------------------------
# 로봇·그리퍼 설정
# ---------------------------------------------------------------------

GRIPPER_MASTER_JOINT = "finger_joint"
GRIPPER_JOINTS = [GRIPPER_MASTER_JOINT]
GRIPPER_OPEN = [0.0]
GRIPPER_CLOSE = [1.18]

# URDF 링크·collision mesh로 계산한 RG2 master 관절각별 안쪽 손가락 간격.
GRIPPER_ANGLE_RAD = np.array(
    [0.0, 0.1967, 0.3933, 0.5900, 0.7867, 0.9833, 1.1800],
    dtype=float,
)
GRIPPER_GAP_M = np.array(
    [0.09958, 0.09026, 0.07731, 0.06123, 0.04263, 0.02224, 0.00083],
    dtype=float,
)
GRASP_COMPRESSION_M = 0.001
GRASP_OPEN_CLEARANCE_M = 0.020
MAX_FINGER_CENTER_CORRECTION_M = 0.050
GRASP_WIDTH_PERCENTILE_LOW = 2.5
GRASP_WIDTH_PERCENTILE_HIGH = 97.5
GRASP_MIN_OBJECT_WIDTH_M = 0.015
GRASP_CENTER_PERCENTILE_LOW = 5.0
GRASP_CENTER_PERCENTILE_HIGH = 95.0

# 큰 순간 토크가 AMR 차체로 전달되지 않도록 보수적으로 제한한다.
DRIVE_STIFFNESS = 2e5
DRIVE_DAMPING = 4e3
DRIVE_MAX_FORCE = 3e3
GRIPPER_DRIVE_STIFFNESS = 5e3
GRIPPER_DRIVE_DAMPING = 5e2
GRIPPER_DRIVE_MAX_FORCE = 40.0
GRIPPER_MAX_SPEED_RAD_S = 0.8

FINGER_STATIC = 1.8
FINGER_DYNAMIC = 1.4


# ---------------------------------------------------------------------
# 위치 전용 테스트 설정
# ---------------------------------------------------------------------

TARGET_TOPICS = {
    "can": "/robot_arm/trash_points/class_0",
    "plastic": "/robot_arm/trash_points/class_2",
}
EXPECTED_TARGET_FRAME = "map"

MIN_POINT_COUNT = 30
POSITION_PERCENTILE_LOW = 2.5
POSITION_PERCENTILE_HIGH = 97.5
PCA_MIN_ANISOTROPY_RATIO = 1.20

# Stage가 Z-up이라는 전제에서 물체 위쪽으로 접근한다.
APPROACH_HEIGHT_M = 0.12
# 1차 점군 중심 위에 손목 카메라를 배치할 관측 거리.
OBSERVATION_DISTANCE_M = 0.35
MOTION_STABILIZE_SECONDS = 0.3
MOTION_STABILIZE_STEPS = int(MOTION_STABILIZE_SECONDS / (1.0 / 60.0))
GRIPPER_STABILIZE_SECONDS = 0.5
GRIPPER_STABILIZE_STEPS = int(GRIPPER_STABILIZE_SECONDS / (1.0 / 60.0))
OBSERVATION_SETTLE_SECONDS = 0.3
OBSERVATION_SETTLE_STEPS = int(
    OBSERVATION_SETTLE_SECONDS / (1.0 / 60.0)
)
STARTUP_SETTLE_SECONDS = 0.5
STARTUP_SETTLE_STEPS = int(STARTUP_SETTLE_SECONDS / (1.0 / 60.0))
POSITION_TOLERANCE_M = 0.02
OPENING_AXIS_TOLERANCE_DEG = 5.0
APPROACH_AXIS_TOLERANCE_DEG = 5.0
MAX_MOVE_STEPS = 1000

# 접근점과 중간 하강은 정밀 정렬하고, 최종점에서는 RMPFlow의 잔여 오차가
# 그리퍼 닫기 자체를 막지 않도록 별도의 안전 허용범위를 사용한다.
PREGRASP_POSITION_TOLERANCE_M = 0.008
PREGRASP_AXIS_TOLERANCE_DEG = 5.0
FINAL_GRASP_POSITION_TOLERANCE_M = 0.030
FINAL_GRASP_OPENING_TOLERANCE_DEG = 40.0
FINAL_GRASP_APPROACH_TOLERANCE_DEG = 20.0
GRASP_Z_SAFETY_OFFSET_M = 0.015
GRASP_DESCENT_STEP_M = 0.02
GRASP_DESCENT_STABILIZE_STEPS = 6


# ---------------------------------------------------------------------
# 로봇팔 단독 워크플로 설정
# ---------------------------------------------------------------------

ARM_ONLY_MODE = False
AMR_ALIGNED_TOPIC = "/amr/arm_alignment_complete"
ARM_TASK_COMPLETE_TOPIC = "/robot_arm/task_complete"

ARM_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# 항상 사용하는 45도 전방 기본 관측 자세.
# URDF FK 기준 TCP +Z(카메라 광축)가 chassis_link 전방·아래 45도를 향한다.
DETECTION_JOINT_DEG = np.array([0.0, 0.0, 90.0, 0.0, 45.0, 0.0])

JOINT_POSITION_TOLERANCE_RAD = np.deg2rad(1.0)
JOINT_MOVE_MAX_SPEED_RAD_S = np.deg2rad(20.0)
JOINT_MOVE_MAX_ACCEL_RAD_S2 = np.deg2rad(40.0)
JOINT_6_SEARCH_MAX_SPEED_RAD_S = np.deg2rad(40.0)
JOINT_6_SEARCH_MAX_ACCEL_RAD_S2 = np.deg2rad(80.0)
JOINT_MOVE_DT_S = 1.0 / 60.0
MAX_JOINT_MOVE_STEPS = 1200
REVERSE_TRAJECTORY_SPEED_RATIO = 0.8
DETECTION_POINTCLOUD_TIMEOUT_S = 3.0
OBSERVATION_MAX_CAMERA_TRANSLATION_M = 0.20
OBSERVATION_POSITION_TOLERANCE_M = 0.045
OBSERVATION_ROTATION_TOLERANCE_DEG = 5.0
MAX_OBSERVATION_MOVE_STEPS = 600
REFINED_POINTCLOUD_TIMEOUT_S = 5.0
MAX_GRASP_RETRIES = 2

# WasteBins 로컬 좌표. 칸 배치는 실제 라벨에 맞춰 이 사전만 조정하면 된다.
BIN_DROP_LOCAL_POSITIONS = {
    "can": np.array([-0.095, 0.235, 0.98], dtype=float),
    "paper": np.array([-0.405, 0.235, 0.98], dtype=float),
    "plastic": np.array([-0.095, -0.235, 0.98], dtype=float),
    "general_waste": np.array([-0.405, -0.235, 0.98], dtype=float),
}
BIN_APPROACH_HEIGHT_M = 0.15

# 45도 전방 관측 자세를 유지한 채 joint_1만 통 방향으로 회전한다.
BIN_JOINT_1_DEG = {
    "plastic": -150.0,
    "can": 150.0,
}


def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )

    print(f"[ROBOT] num_dof={robot.num_dof}")
    print(f"[ROBOT] dof_names={robot.dof_names}")


def pointcloud_to_numpy(msg: PointCloud2):
    """PointCloud2의 x, y, z 필드를 Nx3 numpy 배열로 변환한다."""

    if int(msg.width) <= 0 or int(msg.height) <= 0 or len(msg.data) == 0:
        return np.empty((0, 3), dtype=np.float64)

    if int(msg.point_step) <= 0:
        raise ValueError(f"잘못된 point_step: {msg.point_step}")

    offsets = {
        field.name: int(field.offset)
        for field in msg.fields
        if field.name in ("x", "y", "z")
    }

    missing_fields = {"x", "y", "z"} - set(offsets)
    if missing_fields:
        raise ValueError(
            f"PointCloud2 필드 누락: {sorted(missing_fields)}"
        )

    expected_count = int(msg.width) * int(msg.height)
    available_count = len(msg.data) // int(msg.point_step)
    point_count = min(expected_count, available_count)

    if point_count <= 0:
        return np.empty((0, 3), dtype=np.float64)

    endian = ">" if msg.is_bigendian else "<"
    unpack_float = struct.Struct(endian + "f").unpack_from
    data = bytes(msg.data)

    points = np.empty((point_count, 3), dtype=np.float64)

    for index in range(point_count):
        base = index * int(msg.point_step)
        points[index] = [
            unpack_float(data, base + offsets[axis])[0]
            for axis in ("x", "y", "z")
        ]

    return points[np.all(np.isfinite(points), axis=1)]


class ArmState(Enum):
    WAIT_AMR = auto()
    MOVE_DETECTION_POSE = auto()
    WAIT_TARGET = auto()
    ROTATE_JOINT_6_SEARCH = auto()
    WAIT_TARGET_JOINT_6_180 = auto()
    ALIGN_CAMERA = auto()
    REFINE_POINTCLOUD = auto()
    PICK = auto()
    VERIFY_GRASP = auto()
    REVERSE_TRAJECTORY = auto()
    ROTATE_TO_BIN = auto()
    PLACE = auto()
    RETURN_DETECTION = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class GraspPlan:
    target_class: str
    grasp_position: np.ndarray
    approach_position: np.ndarray
    target_orientation: np.ndarray
    selected_short_xy: np.ndarray
    object_width_m: float
    target_finger_gap_m: float
    target_gripper_angle_rad: float


class AMRSignalInterface:
    """AMR 정렬 완료 입력과 로봇팔 전체 작업 완료 출력만 담당한다."""

    def __init__(self, node, arm_only_mode=ARM_ONLY_MODE):
        self._node = node
        self._arm_only_mode = bool(arm_only_mode)
        self._aligned_received = self._arm_only_mode
        self._complete_sent = False

        signal_qos = QoSProfile(depth=1)
        signal_qos.reliability = ReliabilityPolicy.RELIABLE
        signal_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self._aligned_subscription = node.create_subscription(
            Bool,
            AMR_ALIGNED_TOPIC,
            self._on_amr_aligned,
            signal_qos,
        )
        self._complete_publisher = node.create_publisher(
            Bool,
            ARM_TASK_COMPLETE_TOPIC,
            signal_qos,
        )

        if self._arm_only_mode:
            print(
                "[AMR SIGNAL] ARM_ONLY_MODE=True: "
                "AMR 정렬 완료 신호가 이미 들어온 것으로 가정합니다."
            )
        else:
            self._complete_publisher.publish(Bool(data=False))
            print(
                "[AMR SIGNAL] 실제 연동 모드: "
                f"{AMR_ALIGNED_TOPIC}=True를 기다립니다."
            )
            print(
                f"[ARM SIGNAL SENT] {ARM_TASK_COMPLETE_TOPIC}=False"
            )

    def _on_amr_aligned(self, message):
        self._aligned_received = bool(message.data)
        print(f"[AMR SIGNAL RECEIVED] aligned={self._aligned_received}")

    def is_aligned(self):
        return self._arm_only_mode or self._aligned_received

    def publish_complete(self):
        if self._complete_sent:
            return
        self._complete_publisher.publish(Bool(data=True))
        self._complete_sent = True
        print(f"[ARM SIGNAL SENT] {ARM_TASK_COMPLETE_TOPIC}=True")

    def reset_cycle(self):
        self._aligned_received = self._arm_only_mode
        self._complete_sent = False
        if not self._arm_only_mode:
            self._complete_publisher.publish(Bool(data=False))
            print(
                f"[ARM SIGNAL SENT] {ARM_TASK_COMPLETE_TOPIC}=False "
                "(next AMR cycle ready)"
            )


class PerceptionManager:
    """점군 구독·capture gate·현재 목표 보관만 담당한다."""

    def __init__(self, node, task):
        self._node = node
        self._task = task
        self._last_wait_log_time = 0.0
        self._subscriptions = []
        for target_class, topic in TARGET_TOPICS.items():
            subscription = node.create_subscription(
                PointCloud2,
                topic,
                lambda message, captured_class=target_class: (
                    self._pointcloud_callback(message, captured_class)
                ),
                qos_profile_sensor_data,
            )
            self._subscriptions.append(subscription)
            print(
                f"[POINTCLOUD SUBSCRIBE] class={target_class}, "
                f"topic={topic!r}"
            )

    def _pointcloud_callback(self, message, target_class):
        try:
            if not self._task.accept_target_points:
                return
            if (
                self._task.capture_target_class is not None
                and target_class != self._task.capture_target_class
            ):
                return

            point_frame = message.header.frame_id.lstrip("/")
            if point_frame != EXPECTED_TARGET_FRAME:
                now = time.monotonic()
                if now - self._last_wait_log_time >= 1.0:
                    print(
                        "[POINTCLOUD REJECT] "
                        f"expected_frame={EXPECTED_TARGET_FRAME!r}, "
                        f"received_frame={message.header.frame_id!r}"
                    )
                    self._last_wait_log_time = now
                return

            points = pointcloud_to_numpy(message)
            if len(points) < MIN_POINT_COUNT:
                now = time.monotonic()
                if now - self._last_wait_log_time >= 1.0:
                    print(
                        "[POINTCLOUD WAIT] "
                        f"class={target_class}, "
                        f"frame={message.header.frame_id!r}, "
                        f"valid_points={len(points)}"
                    )
                    self._last_wait_log_time = now
                return

            if self._task.target_points is not None:
                return

            self._task.target_points = points.copy()
            self._task.target_frame = point_frame
            self._task.target_class = target_class
            self._task.accept_target_points = False
            print(
                f"[POINTCLOUD RECEIVED: {self._task.capture_label}] "
                f"class={target_class}, "
                f"frame={point_frame!r}, points={len(points)}"
            )
        except Exception as error:
            print(
                f"[POINTCLOUD ERROR] {type(error).__name__}: {error}"
            )

    def begin_capture(self, label, target_class=None):
        if target_class is not None and target_class not in TARGET_TOPICS:
            raise KeyError(f"지원하지 않는 점군 클래스: {target_class}")
        self._task.target_points = None
        self._task.target_frame = None
        self._task.target_class = None
        self._task.capture_target_class = target_class
        self._task.capture_label = label
        self._task.accept_target_points = True
        if target_class is None:
            topics = TARGET_TOPICS
        else:
            topics = {target_class: TARGET_TOPICS[target_class]}
        print(f"[POINTCLOUD WAIT: {label}] topics={topics}")

    def disable_capture(self, clear=True):
        self._task.accept_target_points = False
        self._task.capture_target_class = None
        self._task.capture_label = "disabled"
        if clear:
            self._task.target_points = None
            self._task.target_frame = None
            self._task.target_class = None

    def has_target(self):
        return self._task.target_points is not None

    def take_target(self):
        if self._task.target_points is None:
            return None, None, None
        points = self._task.target_points.copy()
        frame = self._task.target_frame
        target_class = self._task.target_class
        self.disable_capture(clear=True)
        return points, frame, target_class


class ArmWorkflow:
    """기능 클래스의 결과를 바탕으로 다음 상태만 관리한다."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = ArmState.WAIT_AMR
        self.grasp_attempts = 0
        self.collected_count = 0
        self.initial_points = None
        self.initial_frame = None
        self.target_class = None
        self.coarse_target_position = None
        self.observation_pose = None
        self.grasp_plan = None
        self.failure_reason = ""
        self._completion_announced = False
        self._error_recovery_done = False

    def transition(self, next_state, reason=""):
        previous = self.state
        self.state = next_state
        print(
            f"[STATE] {previous.name} -> {next_state.name}"
            + (f" | {reason}" if reason else "")
        )


class M0609Task(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)

        self._stage = None
        self._robot = None
        self.robot_prim = None

        self.target_points = None
        self.target_frame = None
        self.target_class = None
        self.accept_target_points = False
        self.capture_target_class = None
        self.capture_label = "disabled"

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        self._stage = omni.usd.get_context().get_stage()
        self._load_usd()
        self._setup_physics()
        self._register_robot(scene)
        self._apply_finger_material()

    def _load_usd(self):
        print(f"[TASK STAGE] {self._stage.GetRootLayer().identifier}")

        self.robot_prim = self._stage.GetPrimAtPath(
            MANIPULATOR_PRIM_PATH
        )

        required_paths = [
            ARTICULATION_PRIM_PATH,
            MANIPULATOR_PRIM_PATH,
            RMPFLOW_BASE_PRIM_PATH,
            PHYSICS_EE_PRIM_PATH,
            TCP_PRIM_PATH,
            WRIST_DEPTH_CAMERA_PRIM_PATH,
            WASTEBINS_PRIM_PATH,
            LEFT_INNER_FINGER_PRIM_PATH,
            RIGHT_INNER_FINGER_PRIM_PATH,
        ]

        for prim_path in required_paths:
            prim = self._stage.GetPrimAtPath(prim_path)

            print(
                f"[CHECK] {prim_path} | "
                f"valid={prim.IsValid()} | "
                f"type={prim.GetTypeName() if prim.IsValid() else ''}"
            )

            if not prim.IsValid():
                raise RuntimeError(
                    f"Stage에서 Prim을 찾지 못했습니다: {prim_path}"
                )

    def _setup_physics(self):
        for prim in Usd.PrimRange(self.robot_prim):
            for drive_type in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(
                    prim,
                    drive_type,
                )

                if not drive:
                    continue

                is_gripper_prim = str(prim.GetPath()).startswith(
                    f"{MANIPULATOR_PRIM_PATH}/Gripper/"
                )
                if prim.GetName() == GRIPPER_MASTER_JOINT:
                    drive.GetStiffnessAttr().Set(GRIPPER_DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(GRIPPER_DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(GRIPPER_DRIVE_MAX_FORCE)
                elif is_gripper_prim:
                    # mimic 관절의 USD 원래 Drive 설정을 보존한다. 팔 관절용
                    # 강한 Drive를 덮으면 좌우 손가락의 대칭 운동을 방해한다.
                    continue
                else:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)

    def _register_robot(self, scene):
        gripper = ParallelGripper(
            end_effector_prim_path=PHYSICS_EE_PRIM_PATH,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(
                GRIPPER_OPEN,
                dtype=float,
            ),
            joint_closed_positions=np.array(
                GRIPPER_CLOSE,
                dtype=float,
            ),
            action_deltas=None,
            use_mimic_joints=True,
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ARTICULATION_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=PHYSICS_EE_PRIM_PATH,
                gripper=gripper,
            )
        )

    def _apply_finger_material(self):
        finger_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/finger_material",
            static_friction=FINGER_STATIC,
            dynamic_friction=FINGER_DYNAMIC,
            restitution=0.0,
        )

        finger_paths = {
            "left_inner_finger": LEFT_INNER_FINGER_PRIM_PATH,
            "right_inner_finger": RIGHT_INNER_FINGER_PRIM_PATH,
        }

        for name, path in finger_paths.items():
            SingleGeometryPrim(
                prim_path=path,
                name=f"{name}_geom",
            ).apply_physics_material(finger_material)

    def post_reset(self):
        self.target_points = None
        self.target_frame = None
        self.target_class = None
        self.accept_target_points = False
        self.capture_target_class = None
        self.capture_label = "disabled"

    @staticmethod
    def estimate_target_position(points):
        """1차 관측 점군에서 이상치를 제외한 대략적인 중심만 계산한다."""
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"포인트 배열 형식 오류: shape={points.shape}")
        if len(points) < MIN_POINT_COUNT:
            raise ValueError(
                f"목표 위치 계산용 포인트 부족: {len(points)} < {MIN_POINT_COUNT}"
            )

        low = np.percentile(points, POSITION_PERCENTILE_LOW, axis=0)
        high = np.percentile(points, POSITION_PERCENTILE_HIGH, axis=0)
        robust_mask = np.all((points >= low) & (points <= high), axis=1)
        robust_points = points[robust_mask]
        if len(robust_points) < MIN_POINT_COUNT:
            robust_points = points

        target_position = np.mean(robust_points, axis=0)
        print(
            "[COARSE OBSERVATION TARGET]\n"
            f"  point_count={len(points)}\n"
            f"  robust_point_count={len(robust_points)}\n"
            f"  target_position={np.round(target_position, 6)}"
        )
        return target_position

    @staticmethod
    def _principal_axis_xy(centered_xy):
        """중심이 제거된 XY 점군의 PCA 주축과 고유값 비율을 계산한다."""
        covariance = np.cov(np.asarray(centered_xy, dtype=float).T)
        if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"유효하지 않은 XY 공분산 행렬: {covariance}")

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
        major_index = int(np.argmax(eigenvalues))
        minor_index = 1 - major_index
        major_value = float(eigenvalues[major_index])
        minor_value = float(eigenvalues[minor_index])
        anisotropy_ratio = (
            float("inf")
            if minor_value <= 1.0e-12
            else major_value / minor_value
        )
        if anisotropy_ratio < PCA_MIN_ANISOTROPY_RATIO:
            raise ValueError(
                "PCA XY 방향을 신뢰할 수 없습니다. "
                f"eigenvalues={np.round(eigenvalues, 8)}, "
                f"ratio={anisotropy_ratio:.3f} < {PCA_MIN_ANISOTROPY_RATIO:.3f}"
            )

        major_axis = eigenvectors[:, major_index]
        norm = float(np.linalg.norm(major_axis))
        if norm < 1.0e-9:
            raise ValueError("PCA 주축을 계산할 수 없습니다")
        return major_axis / norm, eigenvalues, anisotropy_ratio

    @staticmethod
    def estimate_target_position_and_pca(points):
        """점군 중심과 XY 평면의 PCA 주축·단축을 계산한다.

        단축은 주축 [x, y]를 같은 XY 평면에서 90도 회전시켜
        [-y, x]로 만든다.
        """
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"포인트 배열 형식 오류: shape={points.shape}")
        if len(points) < MIN_POINT_COUNT:
            raise ValueError(
                f"목표 위치 계산용 포인트 부족: {len(points)} < {MIN_POINT_COUNT}"
            )

        low = np.percentile(points, POSITION_PERCENTILE_LOW, axis=0)
        high = np.percentile(points, POSITION_PERCENTILE_HIGH, axis=0)
        robust_mask = np.all((points >= low) & (points <= high), axis=1)
        robust_points = points[robust_mask]
        if len(robust_points) < MIN_POINT_COUNT:
            robust_points = points

        density_mean_position = np.mean(robust_points, axis=0)
        target_position = density_mean_position.copy()
        robust_xy = robust_points[:, :2]
        center_xy = np.mean(robust_xy, axis=0)
        (
            major_axis_xy,
            pca_eigenvalues,
            pca_anisotropy_ratio,
        ) = M0609Task._principal_axis_xy(robust_xy - center_xy)

        # PCA 고유벡터 부호를 고정해 실행마다 180도 반전되는 것을 줄인다.
        if major_axis_xy[0] < 0.0 or (
            abs(major_axis_xy[0]) < 1.0e-9 and major_axis_xy[1] < 0.0
        ):
            major_axis_xy *= -1.0

        # 같은 XY 평면에서 주축에 수직인 단축.
        short_axis_xy = np.array(
            [-major_axis_xy[1], major_axis_xy[0]],
            dtype=float,
        )
        short_axis_xy /= np.linalg.norm(short_axis_xy)

        # 카메라에서 보이는 면에 점이 몰리면 단순 평균 중심이 한쪽으로
        # 치우친다. PCA 축별 robust 경계의 중점을 사용해 XY 파지 중심을
        # 물체 형상의 가운데로 보정한다.
        major_projected = robust_xy @ major_axis_xy
        short_projected = robust_xy @ short_axis_xy
        major_low, major_high = np.percentile(
            major_projected,
            [GRASP_CENTER_PERCENTILE_LOW, GRASP_CENTER_PERCENTILE_HIGH],
        )
        short_low, short_high = np.percentile(
            short_projected,
            [GRASP_CENTER_PERCENTILE_LOW, GRASP_CENTER_PERCENTILE_HIGH],
        )
        target_position[:2] = (
            major_axis_xy * (0.5 * (major_low + major_high))
            + short_axis_xy * (0.5 * (short_low + short_high))
        )

        pca_major_axis = np.array(
            [major_axis_xy[0], major_axis_xy[1], 0.0],
            dtype=float,
        )
        pca_short_axis = np.array(
            [short_axis_xy[0], short_axis_xy[1], 0.0],
            dtype=float,
        )

        print(
            "[POSITION + PCA TARGET]\n"
            f"  point_count={len(points)}\n"
            f"  robust_point_count={len(robust_points)}\n"
            f"  density_mean_position={np.round(density_mean_position, 6)}\n"
            f"  target_position={np.round(target_position, 6)}\n"
            f"  pca_eigenvalues={np.round(pca_eigenvalues, 8)}\n"
            f"  pca_anisotropy_ratio={pca_anisotropy_ratio:.4f}\n"
            f"  pca_major={np.round(pca_major_axis, 6)}\n"
            f"  pca_short={np.round(pca_short_axis, 6)}\n"
            f"  dot(major, short)={np.dot(pca_major_axis, pca_short_axis):.8f}"
        )
        return target_position, pca_major_axis, pca_short_axis

    @staticmethod
    def gripper_gap_from_angle(angle_rad):
        """RG2 master 관절각을 안쪽 손가락 간격(m)으로 보간한다."""
        angle_rad = float(
            np.clip(angle_rad, GRIPPER_ANGLE_RAD[0], GRIPPER_ANGLE_RAD[-1])
        )
        return float(np.interp(angle_rad, GRIPPER_ANGLE_RAD, GRIPPER_GAP_M))

    @staticmethod
    def gripper_angle_from_gap(gap_m):
        """목표 안쪽 손가락 간격(m)을 RG2 master 관절각으로 보간한다."""
        gap_m = float(np.clip(gap_m, GRIPPER_GAP_M[-1], GRIPPER_GAP_M[0]))
        return float(
            np.interp(
                gap_m,
                GRIPPER_GAP_M[::-1],
                GRIPPER_ANGLE_RAD[::-1],
            )
        )

    @staticmethod
    def estimate_adaptive_grasp(points, pca_short_axis):
        """점군의 PCA 단축 폭으로 목표 손가락 간격과 master 관절각을 정한다."""
        points = np.asarray(points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < MIN_POINT_COUNT:
            raise ValueError(
                f"파지 폭 계산용 포인트 부족: {len(points)} < {MIN_POINT_COUNT}"
            )

        opening_axis_xy = np.asarray(pca_short_axis[:2], dtype=float)
        axis_norm = float(np.linalg.norm(opening_axis_xy))
        if axis_norm < 1.0e-9:
            raise ValueError("PCA 단축으로 그리퍼 폭을 계산할 수 없습니다")
        opening_axis_xy /= axis_norm

        projected = points[:, :2] @ opening_axis_xy
        low, high = np.percentile(
            projected,
            [GRASP_WIDTH_PERCENTILE_LOW, GRASP_WIDTH_PERCENTILE_HIGH],
        )
        object_width_m = float(high - low)

        if object_width_m < GRASP_MIN_OBJECT_WIDTH_M:
            raise ValueError(
                "점군에서 계산한 물체 폭이 너무 작아 안전하게 파지할 수 없습니다. "
                f"object_width={object_width_m:.4f}m"
            )
        if object_width_m >= GRIPPER_GAP_M[0]:
            raise ValueError(
                "물체 폭이 RG2 최대 개방 간격보다 큽니다. "
                f"object_width={object_width_m:.4f}m, "
                f"max_gap={GRIPPER_GAP_M[0]:.4f}m"
            )

        target_gap_m = float(
            np.clip(
                object_width_m - GRASP_COMPRESSION_M,
                GRIPPER_GAP_M[-1],
                GRIPPER_GAP_M[0],
            )
        )
        target_angle_rad = M0609Task.gripper_angle_from_gap(target_gap_m)

        print(
            "[ADAPTIVE GRASP WIDTH]\n"
            f"  opening_axis_xy={np.round(opening_axis_xy, 6)}\n"
            f"  object_width={object_width_m:.4f}m\n"
            f"  compression={GRASP_COMPRESSION_M:.4f}m\n"
            f"  target_finger_gap={target_gap_m:.4f}m\n"
            f"  target_master_angle={target_angle_rad:.4f}rad "
            f"({np.degrees(target_angle_rad):.2f}deg)"
        )
        return object_width_m, target_gap_m, target_angle_rad

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion):
        """Isaac quaternion [w, x, y, z]를 회전행렬로 변환한다."""
        w, x, y, z = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm([w, x, y, z]))
        if norm < 1.0e-10:
            raise ValueError("유효하지 않은 quaternion")
        w, x, y, z = np.array([w, x, y, z], dtype=float) / norm
        return np.array(
            [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ],
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ],
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def rotation_matrix_to_quaternion(rotation):
        """3x3 회전행렬을 Isaac quaternion [w, x, y, z]로 변환한다."""
        matrix = np.asarray(rotation, dtype=float)
        trace = float(np.trace(matrix))

        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = np.sqrt(
                1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
            ) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = np.sqrt(
                1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
            ) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(
                1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
            ) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale

        quaternion = np.array([w, x, y, z], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        return quaternion



class SimulationStopped(Exception):
    """Isaac Sim Stop 또는 창 종료로 현재 작업이 취소되었음을 나타낸다."""


class M0609Motion:
    def __init__(
        self,
        app,
        world,
        robot,
        controller,
        rmpflow_base,
        tcp_prim,
        wrist_camera_prim,
        left_inner_finger_prim,
        right_inner_finger_prim,
    ):
        self._app = app
        self._world = world
        self._robot = robot
        self._controller = controller
        self._rmpflow_base = rmpflow_base
        self._tcp_prim = tcp_prim
        self._wrist_camera_prim = wrist_camera_prim
        self._left_inner_finger_prim = left_inner_finger_prim
        self._right_inner_finger_prim = right_inner_finger_prim
        self._pause_announced = False
        self.last_grasp_result = None
        self._held_gripper_angle_rad = None
        self._record_arm_trajectory = False
        self._arm_trajectory = []

        self._active_joint_names = self._controller.rmp_flow.get_active_joints()
        self._active_joint_indices = np.array(
            [self._robot.get_dof_index(name) for name in self._active_joint_names],
            dtype=np.int32,
        )

    def report_gripper_geometry(self, label):
        """실제 손가락 중심과 master 관절 기반 개방폭을 출력한다."""
        left_position, _ = self._left_inner_finger_prim.get_world_pose()
        right_position, _ = self._right_inner_finger_prim.get_world_pose()
        tcp_position, _ = self._tcp_prim.get_world_pose()
        left_position = np.asarray(left_position, dtype=float)
        right_position = np.asarray(right_position, dtype=float)
        tcp_position = np.asarray(tcp_position, dtype=float)
        midpoint = 0.5 * (left_position + right_position)
        center_distance = float(np.linalg.norm(right_position - left_position))
        master_angle = float(self._robot.gripper.get_joint_positions()[0])
        estimated_gap = M0609Task.gripper_gap_from_angle(master_angle)
        gripper_joint_states = {
            name: float(self._robot.get_joint_positions()[index])
            for index, name in enumerate(self._robot.dof_names)
            if "finger" in name or "knuckle" in name
        }
        print(
            f"[GRIPPER GEOMETRY: {label}]\n"
            f"  master_angle={master_angle:.4f}rad\n"
            f"  estimated_inner_gap={estimated_gap:.4f}m\n"
            f"  finger_center_distance={center_distance:.4f}m\n"
            f"  finger_midpoint={np.round(midpoint, 6)}\n"
            f"  midpoint_from_tcp={np.round(midpoint - tcp_position, 6)}\n"
            f"  gripper_joint_rad={gripper_joint_states}"
        )
        return midpoint, center_distance, estimated_gap

    def wait_for_timeline(self):
        """Pause 중에는 기다리고, Stop/창 종료 시 현재 작업을 취소한다."""
        while self._app.is_running():
            if self._world.is_stopped():
                raise SimulationStopped(
                    "Isaac Sim에서 Stop이 눌려 현재 작업을 취소했습니다."
                )

            if self._world.is_playing():
                if self._pause_announced:
                    print("[SIMULATION RESUMED] 중단한 단계부터 계속합니다.")
                    self._pause_announced = False
                return

            if not self._pause_announced:
                print(
                    "[SIMULATION PAUSED] Play를 누르면 현재 단계부터 계속합니다."
                )
                self._pause_announced = True

            # world.step()은 호출하지 않아 물리 시간과 모션 step을 동결한다.
            self._app.update()
            time.sleep(0.01)

        raise SimulationStopped("Isaac Sim 창이 종료되었습니다.")

    def reset_after_world_reset(self):
        """Stop 뒤 World reset으로 다시 생성된 물리 핸들에 맞춰 상태를 초기화한다."""
        self._controller.reset()
        self._pause_announced = False
        self.last_grasp_result = None
        self._held_gripper_angle_rad = None
        self._record_arm_trajectory = False
        self._arm_trajectory = []

    def _current_active_joint_positions(self):
        return np.asarray(
            self._robot.get_joint_positions(
                joint_indices=self._active_joint_indices
            ),
            dtype=float,
        )

    def apply_gripper_hold(self, full_target):
        """파지 성공 후 팔 명령이 그리퍼 닫힘 명령을 덮지 않게 한다."""
        if self._held_gripper_angle_rad is None:
            return full_target
        master_index = self._robot.get_dof_index(GRIPPER_MASTER_JOINT)
        if master_index < 0:
            raise RuntimeError(
                f"그리퍼 master joint를 찾지 못했습니다: {GRIPPER_MASTER_JOINT}"
            )
        full_target[int(master_index)] = self._held_gripper_angle_rad
        return full_target

    def reapply_gripper_hold(self):
        if self._held_gripper_angle_rad is None:
            return
        self._robot.gripper.apply_action(
            ArticulationAction(
                joint_positions=np.array(
                    [self._held_gripper_angle_rad], dtype=float
                )
            )
        )

    def start_arm_trajectory_recording(self, label):
        self._arm_trajectory = [self._current_active_joint_positions().copy()]
        self._record_arm_trajectory = True
        print(
            f"[ARM TRAJECTORY RECORD START: {label}] "
            f"joint_deg={np.round(np.degrees(self._arm_trajectory[0]), 2)}"
        )

    def clear_arm_trajectory(self):
        self._record_arm_trajectory = False
        self._arm_trajectory = []

    def has_recorded_arm_trajectory(self):
        return len(self._arm_trajectory) >= 2

    def _record_arm_joint_sample(self):
        if self._record_arm_trajectory:
            self._arm_trajectory.append(
                self._current_active_joint_positions().copy()
            )

    def reverse_recorded_arm_trajectory(
        self,
        speed_ratio=REVERSE_TRAJECTORY_SPEED_RATIO,
    ):
        """실제 관절 궤적의 시간축을 재샘플링해 지정 속도로 역재생한다."""
        self._record_arm_trajectory = False
        if len(self._arm_trajectory) < 2:
            raise RuntimeError("역재생할 관절 궤적이 없습니다")
        speed_ratio = float(np.clip(speed_ratio, 0.1, 1.0))
        reverse_path = np.vstack(
            [
                self._current_active_joint_positions(),
                np.asarray(self._arm_trajectory[-2::-1], dtype=float),
            ]
        )
        source_segments = len(reverse_path) - 1
        command_total = max(
            1,
            int(np.ceil(source_segments / speed_ratio)),
        )
        print(
            "[ARM TRAJECTORY REVERSE START]\n"
            f"  samples={len(self._arm_trajectory)}\n"
            f"  speed_ratio={speed_ratio:.2f}\n"
            f"  replay_commands={command_total}"
        )

        for command_count in range(1, command_total + 1):
            self.wait_for_timeline()
            source_position = min(
                command_count * speed_ratio,
                float(source_segments),
            )
            lower_index = int(np.floor(source_position))
            upper_index = min(lower_index + 1, source_segments)
            alpha = source_position - lower_index
            command = (
                (1.0 - alpha) * reverse_path[lower_index]
                + alpha * reverse_path[upper_index]
            )
            full_target = [None] * self._robot.num_dof
            for dof_index, joint_position in zip(
                self._active_joint_indices,
                command,
            ):
                full_target[int(dof_index)] = float(joint_position)
            self.apply_gripper_hold(full_target)
            self._robot.apply_action(
                ArticulationAction(joint_positions=full_target)
            )
            self._world.step(render=True)
            if command_count % 100 == 0:
                print(
                    "[ARM TRAJECTORY REVERSE] "
                    f"command={command_count}/{command_total}"
                )

        target_start = self._arm_trajectory[0]
        final_error_deg = float("inf")
        for settle_step in range(300):
            self.wait_for_timeline()
            full_target = [None] * self._robot.num_dof
            for dof_index, joint_position in zip(
                self._active_joint_indices,
                target_start,
            ):
                full_target[int(dof_index)] = float(joint_position)
            self.apply_gripper_hold(full_target)
            self._robot.apply_action(
                ArticulationAction(joint_positions=full_target)
            )
            self._world.step(render=True)
            actual = self._current_active_joint_positions()
            final_error_deg = float(
                np.max(np.abs(np.degrees(target_start - actual)))
            )
            if final_error_deg < np.degrees(JOINT_POSITION_TOLERANCE_RAD):
                break
        else:
            raise TimeoutError(
                "역궤적 시작 자세 재도달 실패: "
                f"max_joint_error={final_error_deg:.2f}deg"
            )
        self._controller.reset()
        self._sync_rmpflow_base_pose()
        print(
            "[ARM TRAJECTORY REVERSE COMPLETE] "
            f"max_joint_error={final_error_deg:.2f}deg"
        )
        self.clear_arm_trajectory()
        return final_error_deg

    def _sync_rmpflow_base_pose(self):
        base_position, base_orientation = self._rmpflow_base.get_world_pose()
        self._controller.rmp_flow.set_robot_base_pose(
            robot_position=np.asarray(base_position, dtype=float),
            robot_orientation=np.asarray(base_orientation, dtype=float),
        )
        return np.asarray(base_position, dtype=float)

    def _build_top_down_orientation(self, opening_axis, log_label):
        """TCP +Z를 World -Z로 두고 TCP +X를 지정한 XY 축에 맞춘다."""
        _, current_usd_orientation = self._tcp_prim.get_world_pose()
        current_usd_rotation = M0609Task.quaternion_to_rotation_matrix(
            current_usd_orientation
        )
        current_opening_xy = np.asarray(current_usd_rotation[:2, 0], dtype=float)
        current_opening_norm = float(np.linalg.norm(current_opening_xy))
        if current_opening_norm < 1.0e-6:
            raise RuntimeError(
                "현재 TCP +Y축의 XY 투영이 너무 작아 yaw를 계산할 수 없습니다"
            )
        current_opening_xy /= current_opening_norm

        desired_short_xy = np.asarray(opening_axis[:2], dtype=float)
        desired_short_xy /= np.linalg.norm(desired_short_xy)

        # 개폐축은 선이므로 +단축과 -단축 중 회전량이 작은 방향을 선택한다.
        if float(np.dot(current_opening_xy, desired_short_xy)) < 0.0:
            desired_short_xy *= -1.0

        target_x_axis = np.array(
            [desired_short_xy[0], desired_short_xy[1], 0.0],
            dtype=float,
        )
        target_z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        target_y_axis = np.cross(target_z_axis, target_x_axis)
        target_y_axis /= np.linalg.norm(target_y_axis)
        target_rotation = np.column_stack(
            (target_x_axis, target_y_axis, target_z_axis)
        )
        target_orientation = M0609Task.rotation_matrix_to_quaternion(
            target_rotation
        )

        print(
            f"[{log_label}]\n"
            f"  current_usd_tcp_x_opening_xy={np.round(current_opening_xy, 6)}\n"
            f"  selected_opening_axis_xy={np.round(desired_short_xy, 6)}\n"
            f"  target_tcp_z={np.round(target_z_axis, 6)}\n"
            f"  target_orientation_wxyz={np.round(target_orientation, 6)}"
        )
        return target_orientation, desired_short_xy

    def build_pca_short_top_down_orientation(self, pca_short_axis):
        """TCP +Z는 World -Z, 실제 개폐축인 TCP +X는 PCA 단축을 향한다."""
        # USD의 left/right inner finger 중심선은 grasp_tcp +X축과 일치한다.
        return self._build_top_down_orientation(
            pca_short_axis,
            "PCA SHORT TOP-DOWN",
        )

    def build_observation_pose(self, coarse_target_position):
        """현재 자세에서 조금만 이동하고 카메라가 물체 중심을 보게 한다."""
        tcp_position, tcp_orientation = self._tcp_prim.get_world_pose()
        camera_position, camera_orientation = self._wrist_camera_prim.get_world_pose()
        tcp_position = np.asarray(tcp_position, dtype=float)
        camera_position = np.asarray(camera_position, dtype=float)
        coarse_target_position = np.asarray(coarse_target_position, dtype=float)
        tcp_rotation = M0609Task.quaternion_to_rotation_matrix(tcp_orientation)
        camera_rotation = M0609Task.quaternion_to_rotation_matrix(camera_orientation)

        # USD Camera의 광축은 local -Z이다. 이 모델에서는 TCP +Z와 일치해야 한다.
        camera_forward_world = -camera_rotation[:, 2]
        camera_forward_tcp = tcp_rotation.T @ camera_forward_world
        if float(np.dot(camera_forward_tcp, [0.0, 0.0, 1.0])) < 0.99:
            raise RuntimeError(
                "손목 카메라 광축이 예상한 TCP +Z와 일치하지 않습니다. "
                f"camera_forward_tcp={np.round(camera_forward_tcp, 6)}"
            )

        # 정확한 수직 위를 강제하지 않는다. 수직 위 후보 방향으로 최대 20cm만
        # 이동하고, 그 위치에서 카메라 광축이 실제 점군 중심을 향하게 한다.
        overhead_camera_position = (
            coarse_target_position
            + np.array([0.0, 0.0, OBSERVATION_DISTANCE_M], dtype=float)
        )
        camera_translation = overhead_camera_position - camera_position
        requested_translation = float(np.linalg.norm(camera_translation))
        if requested_translation > OBSERVATION_MAX_CAMERA_TRANSLATION_M:
            camera_translation *= (
                OBSERVATION_MAX_CAMERA_TRANSLATION_M / requested_translation
            )
        target_camera_position = camera_position + camera_translation

        target_z_axis = coarse_target_position - target_camera_position
        target_distance = float(np.linalg.norm(target_z_axis))
        if target_distance < 1.0e-6:
            raise RuntimeError("재관측 카메라 위치와 점군 중심이 같습니다")
        target_z_axis /= target_distance

        # 현재 TCP +X를 새 광축에 수직인 평면에 투영해 불필요한 roll을 줄인다.
        current_x_axis = np.asarray(tcp_rotation[:, 0], dtype=float)
        target_x_axis = (
            current_x_axis
            - np.dot(current_x_axis, target_z_axis) * target_z_axis
        )
        x_norm = float(np.linalg.norm(target_x_axis))
        if x_norm < 1.0e-6:
            target_x_axis = np.cross([0.0, 0.0, 1.0], target_z_axis)
            x_norm = float(np.linalg.norm(target_x_axis))
        if x_norm < 1.0e-6:
            raise RuntimeError("재관측 카메라 roll 기준축을 만들 수 없습니다")
        target_x_axis /= x_norm
        target_y_axis = np.cross(target_z_axis, target_x_axis)
        target_y_axis /= np.linalg.norm(target_y_axis)
        target_rotation = np.column_stack(
            (target_x_axis, target_y_axis, target_z_axis)
        )
        target_orientation = M0609Task.rotation_matrix_to_quaternion(
            target_rotation
        )

        # 현재 자세로부터 TCP 기준 카메라 위치 오프셋을 구한다.
        camera_offset_tcp = tcp_rotation.T @ (
            camera_position - tcp_position
        )
        target_tcp_position = (
            target_camera_position - target_rotation @ camera_offset_tcp
        )

        print(
            "[OBLIQUE OBSERVATION POSE]\n"
            f"  coarse_target={np.round(coarse_target_position, 6)}\n"
            f"  current_camera_position={np.round(camera_position, 6)}\n"
            f"  overhead_candidate={np.round(overhead_camera_position, 6)}\n"
            f"  requested_translation={requested_translation:.4f}m\n"
            f"  applied_translation={np.linalg.norm(camera_translation):.4f}m\n"
            f"  camera_offset_tcp={np.round(camera_offset_tcp, 6)}\n"
            f"  target_camera_position={np.round(target_camera_position, 6)}\n"
            f"  camera_look_direction={np.round(target_z_axis, 6)}\n"
            f"  target_tcp_position={np.round(target_tcp_position, 6)}\n"
            f"  target_orientation={np.round(target_orientation, 6)}"
        )
        return target_tcp_position, target_orientation

    def verify_camera_looks_at(self, target_position, max_angle_deg=8.0):
        """관측 자세에서 실제 카메라 광축이 1차 점군 중심을 향하는지 확인한다."""
        camera_position, camera_orientation = self._wrist_camera_prim.get_world_pose()
        camera_position = np.asarray(camera_position, dtype=float)
        camera_rotation = M0609Task.quaternion_to_rotation_matrix(
            camera_orientation
        )
        camera_forward = -camera_rotation[:, 2]
        to_target = np.asarray(target_position, dtype=float) - camera_position
        target_distance = float(np.linalg.norm(to_target))
        if target_distance < 1.0e-6:
            raise RuntimeError("카메라와 관측 목표 위치가 같습니다")
        to_target /= target_distance
        look_angle_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(np.dot(camera_forward, to_target), -1.0, 1.0)
                )
            )
        )

        print(
            "[CAMERA ALIGNMENT CHECK]\n"
            f"  camera_position={np.round(camera_position, 6)}\n"
            f"  camera_forward={np.round(camera_forward, 6)}\n"
            f"  target_distance={target_distance:.4f}m\n"
            f"  look_angle_error={look_angle_deg:.3f}deg"
        )
        if look_angle_deg > max_angle_deg:
            raise RuntimeError(
                "손목 카메라가 1차 점군 중심을 충분히 바라보지 못했습니다. "
                f"look_angle_error={look_angle_deg:.3f}deg > {max_angle_deg:.3f}deg"
            )

    def report_tcp_model_alignment(self, label):
        """RMPFlow URDF TCP와 실제 USD TCP의 World pose 차이를 출력한다."""
        self._sync_rmpflow_base_pose()
        all_joint_positions = np.asarray(
            self._robot.get_joint_positions(),
            dtype=float,
        )
        active_joint_positions = all_joint_positions[self._active_joint_indices]
        rmp_position, rmp_rotation = (
            self._controller.rmp_flow.get_end_effector_pose(
                active_joint_positions
            )
        )
        usd_position, usd_orientation = self._tcp_prim.get_world_pose()
        usd_rotation = M0609Task.quaternion_to_rotation_matrix(usd_orientation)
        rmp_position = np.asarray(rmp_position, dtype=float)
        rmp_rotation = np.asarray(rmp_rotation, dtype=float)
        usd_position = np.asarray(usd_position, dtype=float)

        position_error = float(np.linalg.norm(rmp_position - usd_position))
        relative_rotation = usd_rotation.T @ rmp_rotation
        rotation_cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        rotation_error_deg = float(np.degrees(np.arccos(rotation_cosine)))
        print(
            f"[TCP MODEL ALIGNMENT] {label}\n"
            f"  usd_tcp={np.round(usd_position, 6)}\n"
            f"  rmpflow_tcp={np.round(rmp_position, 6)}\n"
            f"  position_error={position_error:.6f}m\n"
            f"  rotation_error={rotation_error_deg:.4f}deg"
        )
        return position_error, rotation_error_deg

    @staticmethod
    def _orientation_error_deg(current_orientation, target_orientation):
        current_rotation = M0609Task.quaternion_to_rotation_matrix(
            current_orientation
        )
        target_rotation = M0609Task.quaternion_to_rotation_matrix(
            target_orientation
        )
        relative_rotation = target_rotation.T @ current_rotation
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        return float(np.degrees(np.arccos(cosine)))

    def move_to_observation_pose(
        self,
        position,
        orientation,
        position_tolerance=OBSERVATION_POSITION_TOLERANCE_M,
        rotation_tolerance_deg=OBSERVATION_ROTATION_TOLERANCE_DEG,
        max_steps=MAX_OBSERVATION_MOVE_STEPS,
    ):
        """정확한 수직 정렬 없이 근접 look-at 재관측 자세로 이동한다."""
        target_position = np.asarray(position, dtype=float)
        target_orientation = np.asarray(orientation, dtype=float)
        target_orientation /= np.linalg.norm(target_orientation)
        self._controller.reset()
        self._sync_rmpflow_base_pose()

        print(
            "[MOVE OBLIQUE OBSERVATION TARGET]\n"
            f"  target_position={np.round(target_position, 4)}\n"
            f"  target_orientation={np.round(target_orientation, 4)}\n"
            f"  position_tolerance={position_tolerance:.3f}m\n"
            f"  rotation_tolerance={rotation_tolerance_deg:.2f}deg"
        )
        last_position_error = float("inf")
        last_rotation_error = float("inf")
        for step in range(max_steps):
            self.wait_for_timeline()
            self._sync_rmpflow_base_pose()
            action = self._controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            self._robot.apply_action(action)
            self._world.step(render=True)
            self._record_arm_joint_sample()

            current_position, current_orientation = self._tcp_prim.get_world_pose()
            last_position_error = float(
                np.linalg.norm(target_position - np.asarray(current_position))
            )
            last_rotation_error = self._orientation_error_deg(
                current_orientation,
                target_orientation,
            )
            if step % 10 == 0:
                print(
                    f"[MOVE OBLIQUE OBSERVATION] step={step}, "
                    f"position_error={last_position_error:.4f}m, "
                    f"rotation_error={last_rotation_error:.2f}deg"
                )
            if (
                last_position_error < position_tolerance
                and last_rotation_error < rotation_tolerance_deg
            ):
                print(
                    "[MOVE OBLIQUE OBSERVATION COMPLETE] "
                    f"position_error={last_position_error:.4f}m, "
                    f"rotation_error={last_rotation_error:.2f}deg"
                )
                return True

        raise TimeoutError(
            "비스듬한 재관측 자세 도달 실패: "
            f"position_error={last_position_error:.4f}m, "
            f"rotation_error={last_rotation_error:.2f}deg"
        )

    @staticmethod
    def _opening_axis_line_error_deg(current_usd_orientation, desired_short_xy):
        """실제 개폐축인 TCP +X의 XY 투영과 PCA 단축선의 각도 오차."""
        current_rotation = M0609Task.quaternion_to_rotation_matrix(
            current_usd_orientation
        )
        current_opening_xy = np.asarray(current_rotation[:2, 0], dtype=float)
        norm = float(np.linalg.norm(current_opening_xy))
        if norm < 1.0e-6:
            return 90.0
        current_opening_xy /= norm

        # +Y와 -Y는 같은 개폐선으로 처리한다.
        dot = abs(float(np.dot(current_opening_xy, desired_short_xy)))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    @staticmethod
    def _approach_axis_error_deg(current_usd_orientation):
        """TCP +Z와 World -Z 사이의 각도 오차."""
        current_rotation = M0609Task.quaternion_to_rotation_matrix(
            current_usd_orientation
        )
        current_z_axis = np.asarray(current_rotation[:, 2], dtype=float)
        dot = float(np.clip(np.dot(current_z_axis, [0.0, 0.0, -1.0]), -1.0, 1.0))
        return float(np.degrees(np.arccos(dot)))

    def move_to_position_with_pca_top_down(
        self,
        position,
        target_orientation,
        desired_short_xy,
        preferred_joint_positions=None,
        position_tolerance=POSITION_TOLERANCE_M,
        opening_tolerance_deg=OPENING_AXIS_TOLERANCE_DEG,
        approach_tolerance_deg=APPROACH_AXIS_TOLERANCE_DEG,
        max_steps=MAX_MOVE_STEPS,
    ):
        """위치와 PCA 단축에 맞춘 top-down 자세를 RMPflow에 전달한다."""
        target_position = np.asarray(position, dtype=float)
        target_orientation = np.asarray(target_orientation, dtype=float)
        target_orientation /= np.linalg.norm(target_orientation)

        self._controller.reset()
        if preferred_joint_positions is not None:
            preferred_joint_positions = np.asarray(
                preferred_joint_positions,
                dtype=float,
            )
            set_cspace_target = getattr(
                self._controller.rmp_flow,
                "set_cspace_target",
                None,
            )
            if callable(set_cspace_target):
                set_cspace_target(preferred_joint_positions)
                print(
                    "[RMPFLOW CSPACE PREFERENCE] "
                    f"joint_deg={np.round(np.degrees(preferred_joint_positions), 2)}"
                )
            else:
                print(
                    "[RMPFLOW CSPACE PREFERENCE WARNING] "
                    "set_cspace_target API를 찾지 못했습니다."
                )
        base_position = self._sync_rmpflow_base_pose()
        articulation_position, _ = self._robot.get_world_pose()

        print(
            "[MOVE POSITION + PCA TOP-DOWN TARGET]\n"
            f"  target_position={np.round(target_position, 4)}\n"
            f"  target_orientation={np.round(target_orientation, 4)}\n"
            f"  desired_short_xy={np.round(desired_short_xy, 4)}\n"
            f"  chassis={np.round(articulation_position, 4)}\n"
            f"  rmpflow_base={np.round(base_position, 4)}"
        )

        last_position_error = float('inf')
        last_opening_error = float('inf')
        last_approach_error = float('inf')

        for step in range(max_steps):
            self.wait_for_timeline()

            self._sync_rmpflow_base_pose()
            actions = self._controller.forward(
                target_end_effector_position=target_position,
                target_end_effector_orientation=target_orientation,
            )
            self._robot.apply_action(actions)
            self._world.step(render=True)
            self._record_arm_joint_sample()

            current_position, current_usd_orientation = self._tcp_prim.get_world_pose()
            current_position = np.asarray(current_position, dtype=float)
            position_error = float(np.linalg.norm(target_position - current_position))
            opening_error = self._opening_axis_line_error_deg(
                current_usd_orientation,
                desired_short_xy,
            )
            approach_error = self._approach_axis_error_deg(
                current_usd_orientation
            )
            last_position_error = position_error
            last_opening_error = opening_error
            last_approach_error = approach_error

            if step % 10 == 0:
                print(
                    f"[MOVE POSITION + PCA TOP-DOWN] step={step}, "
                    f"position_error={position_error:.4f}m, "
                    f"opening_axis_error={opening_error:.2f}deg, "
                    f"approach_axis_error={approach_error:.2f}deg"
                )
            if step % 50 == 0:
                self.report_tcp_model_alignment(f"move_step_{step}")

            if (
                position_error < position_tolerance
                and opening_error < opening_tolerance_deg
                and approach_error < approach_tolerance_deg
            ):
                print(
                    "[MOVE COMPLETE] "
                    f"position_error={position_error:.4f}m, "
                    f"opening_axis_error={opening_error:.2f}deg, "
                    f"approach_axis_error={approach_error:.2f}deg"
                )
                return True

        current_position, _ = self._tcp_prim.get_world_pose()
        raise TimeoutError(
            "목표 위치 또는 PCA 단축 회전에 도달하지 못했습니다. "
            f"target={target_position}, "
            f"current={np.asarray(current_position)}, "
            f"position_error={last_position_error:.4f}m, "
            f"opening_axis_error={last_opening_error:.2f}deg, "
            f"approach_axis_error={last_approach_error:.2f}deg"
        )

    def _move_gripper_smooth(self, target_angle_rad, wait_steps):
        target_angle_rad = float(
            np.clip(
                target_angle_rad,
                GRIPPER_ANGLE_RAD[0],
                GRIPPER_ANGLE_RAD[-1],
            )
        )
        command_angle = float(self._robot.gripper.get_joint_positions()[0])
        max_step = GRIPPER_MAX_SPEED_RAD_S * JOINT_MOVE_DT_S
        for _ in range(wait_steps):
            self.wait_for_timeline()
            remaining = target_angle_rad - command_angle
            command_angle += float(np.clip(remaining, -max_step, max_step))
            target_action = ArticulationAction(
                joint_positions=np.array([command_angle], dtype=float)
            )
            self._robot.gripper.apply_action(target_action)
            self._world.step(render=True)
        return target_angle_rad

    def open_gripper(self, wait_steps=120):
        self._held_gripper_angle_rad = None
        self._move_gripper_smooth(GRIPPER_OPEN[0], wait_steps)
        return True

    def close_gripper(self, target_angle_rad, wait_steps=120):
        target_angle_rad = float(
            np.clip(
                target_angle_rad,
                GRIPPER_ANGLE_RAD[0],
                GRIPPER_ANGLE_RAD[-1],
            )
        )
        current_angle_rad = float(self._robot.gripper.get_joint_positions()[0])
        print(
            "[GRIPPER CLOSE START]\n"
            f"  current_angle={current_angle_rad:.4f}rad\n"
            f"  current_gap={M0609Task.gripper_gap_from_angle(current_angle_rad):.4f}m\n"
            f"  target_angle={target_angle_rad:.4f}rad\n"
            f"  target_gap={M0609Task.gripper_gap_from_angle(target_angle_rad):.4f}m"
        )
        self._move_gripper_smooth(target_angle_rad, wait_steps)

        actual_angle_rad = float(self._robot.gripper.get_joint_positions()[0])
        actual_gap_m = M0609Task.gripper_gap_from_angle(actual_angle_rad)
        contact_likely = actual_angle_rad < target_angle_rad - 0.02
        self.last_grasp_result = {
            "target_angle_rad": target_angle_rad,
            "actual_angle_rad": actual_angle_rad,
            "actual_gap_m": actual_gap_m,
            "contact_likely": contact_likely,
        }
        self._held_gripper_angle_rad = target_angle_rad
        print(
            "[ADAPTIVE GRIPPER RESULT]\n"
            f"  target_angle={target_angle_rad:.4f}rad\n"
            f"  actual_angle={actual_angle_rad:.4f}rad\n"
            f"  estimated_actual_gap={actual_gap_m:.4f}m\n"
            f"  contact_likely={contact_likely}"
        )
        print(
            "[GRIPPER HOLD ENABLED] "
            f"target_angle={target_angle_rad:.4f}rad; "
            "PLACE 단계까지 닫힘 명령을 유지합니다."
        )
        return True


class ArmController:
    """관절 자세, 파지, 안전 자세, WasteBins 투입 기능을 제공한다."""

    def __init__(self, motion, robot, world, wastebins_prim, spin_once):
        self._motion = motion
        self._robot = robot
        self._world = world
        self._wastebins_prim = wastebins_prim
        self._spin_once = spin_once
        self._arm_joint_indices = np.array(
            [robot.get_dof_index(name) for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        if np.any(self._arm_joint_indices < 0):
            raise RuntimeError(
                f"팔 관절 인덱스를 찾지 못했습니다: {self._arm_joint_indices}"
            )
        print(f"[ARM] joint_indices={self._arm_joint_indices}")

    def _after_world_step(self):
        if self._spin_once is not None:
            self._spin_once()

    def wait_stable(self, label, steps=MOTION_STABILIZE_STEPS):
        """명령 사이에 짧게 목표를 유지해 차체와 팔 진동을 가라앉힌다."""
        seconds = int(steps) * JOINT_MOVE_DT_S
        print(
            f"[FLOW STABILIZE: {label}] "
            f"seconds={seconds:.1f}, steps={steps}"
        )
        for _ in range(int(steps)):
            self._motion.wait_for_timeline()
            self._motion.reapply_gripper_hold()
            self._world.step(render=True)
            self._after_world_step()

    def move_joint_pose(
        self,
        target_joint_deg,
        label,
        tolerance_rad=JOINT_POSITION_TOLERANCE_RAD,
        max_steps=MAX_JOINT_MOVE_STEPS,
    ):
        requested_joint_rad = np.deg2rad(
            np.asarray(target_joint_deg, dtype=float)
        )
        if requested_joint_rad.shape != (6,):
            raise ValueError(
                f"6축 목표 관절 형식 오류: shape={requested_joint_rad.shape}"
            )

        current = np.asarray(
            self._robot.get_joint_positions(
                joint_indices=self._arm_joint_indices
            ),
            dtype=float,
        )
        target_joint_rad = requested_joint_rad.copy()

        # joint_3을 제외한 회전축은 같은 자세의 ±360도 후보 중 현재 각도에서
        # 가장 가까운 값을 선택한다. 불필요하게 반대 방향으로 크게 돌지 않는다.
        for joint_index in (0, 1, 3, 4, 5):
            candidates = requested_joint_rad[joint_index] + 2.0 * np.pi * np.array(
                [-1.0, 0.0, 1.0], dtype=float
            )
            candidates = candidates[
                (candidates >= -2.0 * np.pi)
                & (candidates <= 2.0 * np.pi)
            ]
            target_joint_rad[joint_index] = candidates[
                np.argmin(np.abs(candidates - current[joint_index]))
            ]

        print(
            f"[JOINT MOVE TARGET: {label}]\n"
            f"  current_deg={np.round(np.degrees(current), 2)}\n"
            f"  requested_deg={np.round(target_joint_deg, 2)}\n"
            f"  shortest_target_deg={np.round(np.degrees(target_joint_rad), 2)}\n"
            f"  max_speed={np.degrees(JOINT_MOVE_MAX_SPEED_RAD_S):.1f}deg/s\n"
            f"  max_accel={np.degrees(JOINT_MOVE_MAX_ACCEL_RAD_S2):.1f}deg/s^2"
        )
        command_position = current.copy()
        command_velocity = np.zeros(6, dtype=float)
        last_max_error = float("inf")
        for step in range(max_steps):
            self._motion.wait_for_timeline()

            remaining = target_joint_rad - command_position
            desired_velocity = np.clip(
                remaining / JOINT_MOVE_DT_S,
                -JOINT_MOVE_MAX_SPEED_RAD_S,
                JOINT_MOVE_MAX_SPEED_RAD_S,
            )
            max_velocity_change = (
                JOINT_MOVE_MAX_ACCEL_RAD_S2 * JOINT_MOVE_DT_S
            )
            command_velocity += np.clip(
                desired_velocity - command_velocity,
                -max_velocity_change,
                max_velocity_change,
            )
            command_step = command_velocity * JOINT_MOVE_DT_S
            reached_command = np.abs(command_step) >= np.abs(remaining)
            command_position += command_step
            command_position[reached_command] = target_joint_rad[reached_command]
            command_velocity[reached_command] = 0.0

            full_target = [None] * self._robot.num_dof
            for index, target in zip(
                self._arm_joint_indices,
                command_position,
            ):
                full_target[int(index)] = float(target)
            action = ArticulationAction(joint_positions=full_target)
            self._robot.apply_action(action)
            self._world.step(render=True)
            self._after_world_step()

            current = np.asarray(
                self._robot.get_joint_positions(
                    joint_indices=self._arm_joint_indices
                ),
                dtype=float,
            )
            errors = np.abs(target_joint_rad - current)
            last_max_error = float(np.max(errors))
            if step % 25 == 0:
                print(
                    f"[JOINT MOVE: {label}] step={step}, "
                    f"max_error={np.degrees(last_max_error):.2f}deg"
                )
            if last_max_error < tolerance_rad:
                print(
                    f"[JOINT MOVE COMPLETE: {label}] "
                    f"max_error={np.degrees(last_max_error):.2f}deg"
                )
                return True

        raise TimeoutError(
            f"{label} 관절 자세 도달 실패: "
            f"max_error={np.degrees(last_max_error):.2f}deg"
        )

    def move_detection_pose(self):
        """쓰레기를 찾기 위한 고정 전방 45도 카메라 자세로 이동한다."""
        return self.move_joint_pose(
            DETECTION_JOINT_DEG,
            "DEFAULT_FORWARD_45_CAMERA_POSE",
        )

    def move_selected_joints(
        self,
        arm_joint_indices,
        target_deg,
        label,
        tolerance_rad=JOINT_POSITION_TOLERANCE_RAD,
        max_steps=MAX_JOINT_MOVE_STEPS,
        max_speed_rad_s=JOINT_MOVE_MAX_SPEED_RAD_S,
        max_accel_rad_s2=JOINT_MOVE_MAX_ACCEL_RAD_S2,
        stop_condition=None,
    ):
        """선택 관절을 움직이며 stop_condition이 참이면 즉시 탐색을 중단한다."""
        selected = np.asarray(arm_joint_indices, dtype=np.int32)
        requested = np.deg2rad(np.asarray(target_deg, dtype=float))
        if selected.ndim != 1 or requested.shape != selected.shape:
            raise ValueError(
                f"선택 관절 목표 형식 오류: indices={selected}, "
                f"target_shape={requested.shape}"
            )
        if len(selected) == 0 or np.any(selected < 0) or np.any(selected >= 6):
            raise ValueError(f"선택 관절 범위 오류: {selected}")
        max_speed_rad_s = float(max_speed_rad_s)
        max_accel_rad_s2 = float(max_accel_rad_s2)
        if max_speed_rad_s <= 0.0 or max_accel_rad_s2 <= 0.0:
            raise ValueError(
                "관절 속도와 가속도는 양수여야 합니다: "
                f"speed={max_speed_rad_s}, accel={max_accel_rad_s2}"
            )

        initial_all = np.asarray(
            self._robot.get_joint_positions(
                joint_indices=self._arm_joint_indices
            ),
            dtype=float,
        )
        current = initial_all[selected].copy()
        target = requested.copy()
        continuous_arm_indices = {0, 1, 3, 4, 5}
        for local_index, arm_index in enumerate(selected):
            if int(arm_index) not in continuous_arm_indices:
                continue
            candidates = requested[local_index] + 2.0 * np.pi * np.array(
                [-1.0, 0.0, 1.0], dtype=float
            )
            candidates = candidates[
                (candidates >= -2.0 * np.pi)
                & (candidates <= 2.0 * np.pi)
            ]
            target[local_index] = candidates[
                np.argmin(np.abs(candidates - current[local_index]))
            ]

        print(
            f"[SELECTED JOINT MOVE TARGET: {label}]\n"
            f"  joints={[int(index) + 1 for index in selected]}\n"
            f"  current_deg={np.round(np.degrees(current), 2)}\n"
            f"  requested_deg={np.round(target_deg, 2)}\n"
            f"  shortest_target_deg={np.round(np.degrees(target), 2)}\n"
            f"  max_speed={np.degrees(max_speed_rad_s):.1f}deg/s\n"
            f"  max_accel={np.degrees(max_accel_rad_s2):.1f}deg/s^2"
        )
        command_position = current.copy()
        command_velocity = np.zeros(len(selected), dtype=float)
        last_max_error = float("inf")
        for step in range(max_steps):
            self._motion.wait_for_timeline()
            remaining = target - command_position
            desired_velocity = np.clip(
                remaining / JOINT_MOVE_DT_S,
                -max_speed_rad_s,
                max_speed_rad_s,
            )
            max_velocity_change = (
                max_accel_rad_s2 * JOINT_MOVE_DT_S
            )
            command_velocity += np.clip(
                desired_velocity - command_velocity,
                -max_velocity_change,
                max_velocity_change,
            )
            command_step = command_velocity * JOINT_MOVE_DT_S
            reached_command = np.abs(command_step) >= np.abs(remaining)
            command_position += command_step
            command_position[reached_command] = target[reached_command]
            command_velocity[reached_command] = 0.0

            full_target = [None] * self._robot.num_dof
            for local_index, arm_index in enumerate(selected):
                robot_dof_index = int(self._arm_joint_indices[arm_index])
                full_target[robot_dof_index] = float(
                    command_position[local_index]
                )
            self._motion.apply_gripper_hold(full_target)
            self._robot.apply_action(
                ArticulationAction(joint_positions=full_target)
            )
            self._world.step(render=True)
            self._after_world_step()

            if stop_condition is not None and stop_condition():
                interrupted = np.asarray(
                    self._robot.get_joint_positions(
                        joint_indices=self._arm_joint_indices[selected]
                    ),
                    dtype=float,
                )
                print(
                    f"[SELECTED JOINT MOVE INTERRUPTED: {label}] "
                    "유효 포인트클라우드 감지, "
                    f"current_deg={np.round(np.degrees(interrupted), 2)}"
                )
                return False

            current = np.asarray(
                self._robot.get_joint_positions(
                    joint_indices=self._arm_joint_indices[selected]
                ),
                dtype=float,
            )
            errors = np.abs(target - current)
            last_max_error = float(np.max(errors))
            if step % 25 == 0:
                print(
                    f"[SELECTED JOINT MOVE: {label}] step={step}, "
                    f"max_error={np.degrees(last_max_error):.2f}deg"
                )
            if last_max_error < tolerance_rad:
                final_all = np.asarray(
                    self._robot.get_joint_positions(
                        joint_indices=self._arm_joint_indices
                    ),
                    dtype=float,
                )
                passive_mask = np.ones(6, dtype=bool)
                passive_mask[selected] = False
                passive_drift_deg = float(
                    np.max(
                        np.abs(
                            np.degrees(
                                final_all[passive_mask]
                                - initial_all[passive_mask]
                            )
                        )
                    )
                )
                print(
                    f"[SELECTED JOINT MOVE COMPLETE: {label}] "
                    f"max_error={np.degrees(last_max_error):.2f}deg, "
                    f"passive_joint_max_drift={passive_drift_deg:.2f}deg"
                )
                return True

        raise TimeoutError(
            f"{label} 선택 관절 이동 실패: "
            f"max_error={np.degrees(last_max_error):.2f}deg"
        )

    def move_joint_1_only(self, target_deg, label):
        return self.move_selected_joints([0], [target_deg], label)

    def move_joint_6_search(self, target_deg, label, stop_condition=None):
        """joint_6 탐색 회전 중 유효 점군이 감지되면 즉시 중단한다."""
        return self.move_selected_joints(
            [5],
            [target_deg],
            label,
            max_speed_rad_s=JOINT_6_SEARCH_MAX_SPEED_RAD_S,
            max_accel_rad_s2=JOINT_6_SEARCH_MAX_ACCEL_RAD_S2,
            stop_condition=stop_condition,
        )

    def normalize_joint_6_for_observation(self):
        """탐지 각도에서 가장 가까운 기존 관측 기준각(0/180도)으로 이동한다."""
        current_deg = float(
            np.degrees(
                self._robot.get_joint_positions(
                    joint_indices=[self._arm_joint_indices[5]]
                )[0]
            )
        ) % 360.0
        distance_to_0 = min(current_deg, 360.0 - current_deg)
        distance_to_180 = abs(current_deg - 180.0)
        target_deg = 0.0 if distance_to_0 <= distance_to_180 else 180.0
        print(
            "[JOINT 6 OBSERVATION NORMALIZE] "
            f"detected_deg={current_deg:.2f}, target_deg={target_deg:.1f}"
        )
        return self.move_joint_6_search(
            target_deg,
            f"NORMALIZE_JOINT_6_{int(target_deg)}",
        )

    def pick(self, plan):
        self._motion.open_gripper()
        self.wait_stable(
            "gripper_open_before_pick",
            GRIPPER_STABILIZE_STEPS,
        )
        safe_grasp_position = np.asarray(plan.grasp_position, dtype=float).copy()
        safe_grasp_position[2] += GRASP_Z_SAFETY_OFFSET_M
        approach_position = np.asarray(plan.approach_position, dtype=float).copy()
        if approach_position[2] <= safe_grasp_position[2]:
            raise ValueError(
                "접근점이 안전 파지점보다 높지 않습니다: "
                f"approach_z={approach_position[2]:.4f}, "
                f"grasp_z={safe_grasp_position[2]:.4f}"
            )

        print(
            "[SAFE GRASP DESCENT]\n"
            f"  pointcloud_grasp={np.round(plan.grasp_position, 4)}\n"
            f"  safe_grasp={np.round(safe_grasp_position, 4)}\n"
            f"  z_safety_offset={GRASP_Z_SAFETY_OFFSET_M:.3f}m\n"
            f"  descent_step<={GRASP_DESCENT_STEP_M:.3f}m"
        )

        # 물체 위에서 XY와 자세를 먼저 8mm/5도 이내로 맞춘다. 기존 2cm
        # 오차를 남긴 채 하강하면 RMPFlow가 대각선으로 보정하며 손가락이
        # 물체를 먼저 치는 문제가 생길 수 있다.
        self._motion.move_to_position_with_pca_top_down(
            approach_position,
            plan.target_orientation,
            plan.selected_short_xy,
            position_tolerance=PREGRASP_POSITION_TOLERANCE_M,
            opening_tolerance_deg=PREGRASP_AXIS_TOLERANCE_DEG,
            approach_tolerance_deg=PREGRASP_AXIS_TOLERANCE_DEG,
        )
        self.wait_stable("pick_approach")

        # 팔 이동 중 mimic 관절이 충분히 열리지 않았을 수 있으므로 물체 위
        # 안전 높이에서 다시 최대 개방한 뒤 실제 손가락 중심을 기준으로
        # 파지 XY를 보정한다.
        self._motion.open_gripper()
        self.wait_stable(
            "gripper_reopen_at_approach",
            GRIPPER_STABILIZE_STEPS,
        )
        finger_midpoint, _, actual_open_gap = (
            self._motion.report_gripper_geometry("pre_descent_open")
        )
        required_open_gap = plan.object_width_m + GRASP_OPEN_CLEARANCE_M
        if actual_open_gap < required_open_gap:
            print(
                "[GRIPPER REOPEN RETRY] "
                f"actual_gap={actual_open_gap:.4f}m < "
                f"required_gap={required_open_gap:.4f}m"
            )
            self._motion.open_gripper(wait_steps=240)
            self.wait_stable(
                "gripper_reopen_retry",
                GRIPPER_STABILIZE_STEPS,
            )
            finger_midpoint, _, actual_open_gap = (
                self._motion.report_gripper_geometry("pre_descent_reopened")
            )
        if actual_open_gap < required_open_gap:
            raise RuntimeError(
                "최대 개방 후에도 물체 진입 여유가 부족합니다: "
                f"actual_gap={actual_open_gap:.4f}m, "
                f"required_gap={required_open_gap:.4f}m"
            )

        finger_center_correction_xy = (
            np.asarray(plan.grasp_position[:2], dtype=float)
            - np.asarray(finger_midpoint[:2], dtype=float)
        )
        correction_distance = float(np.linalg.norm(finger_center_correction_xy))
        if correction_distance > MAX_FINGER_CENTER_CORRECTION_M:
            raise RuntimeError(
                "손가락 중심 보정량이 안전 한계를 초과했습니다: "
                f"correction={correction_distance:.4f}m > "
                f"limit={MAX_FINGER_CENTER_CORRECTION_M:.4f}m"
            )
        approach_position[:2] += finger_center_correction_xy
        safe_grasp_position[:2] += finger_center_correction_xy
        print(
            "[FINGER CENTER XY CORRECTION]\n"
            f"  object_xy={np.round(plan.grasp_position[:2], 6)}\n"
            f"  finger_midpoint_xy={np.round(finger_midpoint[:2], 6)}\n"
            f"  correction_xy={np.round(finger_center_correction_xy, 6)}\n"
            f"  correction_distance={correction_distance:.4f}m\n"
            f"  open_gap={actual_open_gap:.4f}m\n"
            f"  required_gap={required_open_gap:.4f}m"
        )
        if correction_distance > 0.001:
            self._motion.move_to_position_with_pca_top_down(
                approach_position,
                plan.target_orientation,
                plan.selected_short_xy,
                position_tolerance=PREGRASP_POSITION_TOLERANCE_M,
                opening_tolerance_deg=PREGRASP_AXIS_TOLERANCE_DEG,
                approach_tolerance_deg=PREGRASP_AXIS_TOLERANCE_DEG,
            )
            self.wait_stable("pick_approach_finger_centered")
            self._motion.report_gripper_geometry("finger_centered")

        descent_distance = approach_position[2] - safe_grasp_position[2]
        descent_steps = max(1, int(np.ceil(descent_distance / GRASP_DESCENT_STEP_M)))
        for descent_index in range(1, descent_steps + 1):
            fraction = descent_index / descent_steps
            descent_target = safe_grasp_position.copy()
            descent_target[2] = (
                approach_position[2]
                + fraction * (safe_grasp_position[2] - approach_position[2])
            )
            is_final = descent_index == descent_steps
            print(
                f"[SAFE GRASP DESCENT] step={descent_index}/{descent_steps}, "
                f"target={np.round(descent_target, 4)}"
            )
            self._motion.move_to_position_with_pca_top_down(
                descent_target,
                plan.target_orientation,
                plan.selected_short_xy,
                position_tolerance=(
                    FINAL_GRASP_POSITION_TOLERANCE_M
                    if is_final
                    else PREGRASP_POSITION_TOLERANCE_M
                ),
                opening_tolerance_deg=(
                    FINAL_GRASP_OPENING_TOLERANCE_DEG
                    if is_final
                    else PREGRASP_AXIS_TOLERANCE_DEG
                ),
                approach_tolerance_deg=(
                    FINAL_GRASP_APPROACH_TOLERANCE_DEG
                    if is_final
                    else PREGRASP_AXIS_TOLERANCE_DEG
                ),
            )
            self.wait_stable(
                f"grasp_descent_{descent_index}_{descent_steps}",
                GRASP_DESCENT_STABILIZE_STEPS,
            )

        self.wait_stable("pick_grasp_position")
        self._motion.close_gripper(plan.target_gripper_angle_rad)
        self.wait_stable("gripper_closed", GRIPPER_STABILIZE_STEPS)
        # 상승은 별도 새 경로를 만들지 않는다. 성공 시 기록 궤적 역재생이
        # 하강 경로를 그대로 거슬러 올라간다.
        return True

    def verify_grasp(self):
        result = self._motion.last_grasp_result
        if result is None:
            print("[GRASP VERIFY] 그리퍼 결과가 없습니다.")
            return False

        not_fully_closed = result["actual_gap_m"] > 0.006
        command_is_adaptive = result["target_angle_rad"] < 1.16
        # 현재 모델에는 파지력/접촉 센서가 없다. 물체를 잡아도
        # 명령 각도까지 도달하면 contact_likely=False가 되므로, 해당 값은
        # 진단용으로만 남기고 점군 폭으로 계산한 적응형 파지 명령을 승인한다.
        plausible = bool(command_is_adaptive)
        print(
            "[GRASP VERIFY]\n"
            f"  not_fully_closed={not_fully_closed}\n"
            f"  command_is_adaptive={command_is_adaptive}\n"
            f"  contact_likely_diagnostic={result['contact_likely']}\n"
            "  verification_mode=adaptive_command_no_force_sensor\n"
            f"  plausible={plausible}"
        )
        return plausible

    def _bin_local_to_world(self, local_position):
        bin_position, bin_orientation = self._wastebins_prim.get_world_pose()
        bin_rotation = M0609Task.quaternion_to_rotation_matrix(bin_orientation)
        world_position = (
            np.asarray(bin_position, dtype=float)
            + bin_rotation @ np.asarray(local_position, dtype=float)
        )
        return world_position, bin_rotation

    def rotate_joint_1_toward_bin(self, target_class):
        """45도 관측 자세를 유지하고 joint_1만 고정 통 각도로 돌린다."""
        if target_class not in BIN_JOINT_1_DEG:
            raise KeyError(f"joint_1 통 각도가 없는 클래스: {target_class}")
        bin_heading_deg = BIN_JOINT_1_DEG[target_class]
        print(
            f"[BIN JOINT_1 HEADING: {target_class}] "
            f"fixed_target={bin_heading_deg:.2f}deg"
        )
        return self.move_joint_1_only(
            bin_heading_deg,
            f"BIN_{target_class.upper()}_JOINT_1",
        )

    def place(self, target_class):
        """통 각도에서 Cartesian 이동 없이 그리퍼만 열어 투하한다."""
        if target_class not in BIN_JOINT_1_DEG:
            raise KeyError(f"joint_1 통 각도가 없는 클래스: {target_class}")
        print(
            f"[BIN PLACE: {target_class}]\n"
            f"  joint_1={BIN_JOINT_1_DEG[target_class]:.2f}deg\n"
            "  action=open_gripper_only"
        )
        self._motion.open_gripper()
        self.wait_stable(
            f"bin_{target_class}_release",
            GRIPPER_STABILIZE_STEPS,
        )
        return True


def run():
    """로봇팔 단독 워크플로를 state + if/elif 방식으로 실행한다."""
    if not rclpy.ok():
        rclpy.init()

    pedestrian_prim = loaded_stage.GetPrimAtPath(PEDESTRIAN_PRIM_PATH)
    pedestrian_enabled = pedestrian_prim.IsValid() and pedestrian_prim.IsActive()
    pedestrian_translate = None
    if pedestrian_enabled:
        with Usd.EditContext(loaded_stage, loaded_stage.GetSessionLayer()):
            pedestrian_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        pedestrian_translate = pedestrian_prim.GetAttribute("xformOp:translate")

    ros_node = rclpy.create_node("m0609_arm_workflow")
    world = World(stage_units_in_meters=1.0)
    task = M0609Task(name="m0609_task")
    world.add_task(task)
    world.reset()

    robot = world.scene.get_object("m0609_robot")
    if robot is None:
        raise RuntimeError("로봇을 Scene에서 찾지 못했습니다")
    initialize_robot(robot, world)

    controller = RMPFlowController(
        name="controller",
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=RMPFLOW_EE_LINK_NAME,
    )
    rmpflow_base_prim = SingleXFormPrim(
        prim_path=RMPFLOW_BASE_PRIM_PATH,
        name="m0609_rmpflow_base",
        reset_xform_properties=False,
    )
    tcp_prim = SingleXFormPrim(
        prim_path=TCP_PRIM_PATH,
        name="m0609_grasp_tcp",
        reset_xform_properties=False,
    )
    wrist_camera_prim = SingleXFormPrim(
        prim_path=WRIST_DEPTH_CAMERA_PRIM_PATH,
        name="m0609_wrist_depth_camera",
        reset_xform_properties=False,
    )
    left_inner_finger_prim = SingleXFormPrim(
        prim_path=LEFT_INNER_FINGER_PRIM_PATH,
        name="m0609_left_inner_finger",
        reset_xform_properties=False,
    )
    right_inner_finger_prim = SingleXFormPrim(
        prim_path=RIGHT_INNER_FINGER_PRIM_PATH,
        name="m0609_right_inner_finger",
        reset_xform_properties=False,
    )
    wastebins_prim = SingleXFormPrim(
        prim_path=WASTEBINS_PRIM_PATH,
        name="cleanrobot_wastebins",
        reset_xform_properties=False,
    )
    if pedestrian_enabled:
        print(
            "[PEDESTRIAN MOTION] USD native S-curve + walking clip, "
            "speed=0.30m/s, collider=static"
        )
    else:
        print("[PEDESTRIAN DISABLED] Nathan_DynamicRoot is inactive")
    motion = M0609Motion(
        app=simulation_app,
        world=world,
        robot=robot,
        controller=controller,
        rmpflow_base=rmpflow_base_prim,
        tcp_prim=tcp_prim,
        wrist_camera_prim=wrist_camera_prim,
        left_inner_finger_prim=left_inner_finger_prim,
        right_inner_finger_prim=right_inner_finger_prim,
    )

    signals = AMRSignalInterface(ros_node, arm_only_mode=ARM_ONLY_MODE)
    perception = PerceptionManager(ros_node, task)
    workflow = ArmWorkflow()

    def spin_once():
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        time.sleep(0.01)

    arm = ArmController(
        motion=motion,
        robot=robot,
        world=world,
        wastebins_prim=wastebins_prim,
        spin_once=spin_once,
    )

    def step_and_spin():
        motion.wait_for_timeline()
        world.step(render=True)
        spin_once()

    def settle_without_capture(step_count, label):
        perception.disable_capture(clear=True)
        print(
            f"[POINTCLOUD GATE] {label}: "
            f"capture disabled, steps={step_count}"
        )
        for _ in range(step_count):
            step_and_spin()

    def wait_for_pointcloud(label, timeout_s, target_class=None):
        perception.begin_capture(label, target_class=target_class)
        max_steps = max(1, int(float(timeout_s) / 0.01))
        for _ in range(max_steps):
            step_and_spin()
            if perception.has_target():
                return perception.take_target()

        perception.disable_capture(clear=True)
        print(
            f"[POINTCLOUD TIMEOUT: {label}] "
            f"class={target_class or 'can|plastic'}, "
            f"timeout={timeout_s:.1f}s"
        )
        return None, None, None

    def make_grasp_plan(
        target_class,
        refined_points,
        initial_frame,
        refined_frame,
    ):
        (
            grasp_position,
            _pca_major_axis,
            pca_short_axis,
        ) = task.estimate_target_position_and_pca(refined_points)
        (
            object_width_m,
            target_finger_gap_m,
            target_gripper_angle_rad,
        ) = task.estimate_adaptive_grasp(
            refined_points,
            pca_short_axis,
        )
        (
            target_orientation,
            selected_short_xy,
        ) = motion.build_pca_short_top_down_orientation(pca_short_axis)
        approach_position = grasp_position + np.array(
            [0.0, 0.0, APPROACH_HEIGHT_M], dtype=float
        )
        refinement_delta = (
            grasp_position - workflow.coarse_target_position
        )
        print(
            "[REFINED TARGET CHECK]\n"
            f"  initial_frame={initial_frame!r}\n"
            f"  refined_frame={refined_frame!r}\n"
            f"  coarse_target={np.round(workflow.coarse_target_position, 4)}\n"
            f"  refined_target={np.round(grasp_position, 4)}\n"
            f"  refinement_delta={np.round(refinement_delta, 4)}\n"
            f"  refinement_distance={np.linalg.norm(refinement_delta):.4f}m"
        )
        return GraspPlan(
            target_class=target_class,
            grasp_position=grasp_position,
            approach_position=approach_position,
            target_orientation=target_orientation,
            selected_short_xy=selected_short_xy,
            object_width_m=object_width_m,
            target_finger_gap_m=target_finger_gap_m,
            target_gripper_angle_rad=target_gripper_angle_rad,
        )

    restart_required = False

    try:
        world.play()
        try:
            settle_without_capture(STARTUP_SETTLE_STEPS, "startup")
            if pedestrian_enabled:
                timeline_time = omni.timeline.get_timeline_interface().get_current_time()
                time_code = timeline_time * loaded_stage.GetTimeCodesPerSecond()
                print(
                    "[PEDESTRIAN ANIMATION CHECK] "
                    f"time={timeline_time:.3f}s, "
                    f"position={pedestrian_translate.Get(time_code)}"
                )
            motion.report_tcp_model_alignment("workflow_start")
        except SimulationStopped as error:
            if simulation_app.is_running():
                print(f"[TASK CANCELLED] {error}")
                restart_required = True

        while simulation_app.is_running():
            if restart_required:
                print("[RESTART WAIT] Isaac Sim에서 Play를 누르세요.")
                while (
                    simulation_app.is_running()
                    and not world.is_playing()
                ):
                    simulation_app.update()
                    spin_once()

                if not simulation_app.is_running():
                    break

                print("[RESTART] World와 로봇팔 워크플로를 초기화합니다.")
                world.reset()
                initialize_robot(robot, world)
                motion.reset_after_world_reset()
                signals.reset_cycle()
                perception.disable_capture(clear=True)
                workflow.reset()
                settle_without_capture(STARTUP_SETTLE_STEPS, "restart")
                restart_required = False

            try:
                state = workflow.state

                if state == ArmState.WAIT_AMR:
                    if signals.is_aligned():
                        workflow.transition(
                            ArmState.MOVE_DETECTION_POSE,
                            "AMR 정렬 완료",
                        )
                    else:
                        step_and_spin()

                elif state == ArmState.MOVE_DETECTION_POSE:
                    perception.disable_capture(clear=True)
                    arm.move_detection_pose()
                    arm.wait_stable("detection_pose")
                    motion.clear_arm_trajectory()
                    workflow.transition(ArmState.WAIT_TARGET)

                elif state == ArmState.WAIT_TARGET:
                    perception.begin_capture("joint_6_continuous_sweep")
                    for search_target_deg, search_label in (
                        (180.0, "SEARCH_SWEEP_0_TO_180"),
                        (0.0, "SEARCH_SWEEP_180_TO_0"),
                    ):
                        arm.move_joint_6_search(
                            search_target_deg,
                            search_label,
                            stop_condition=perception.has_target,
                        )
                        if perception.has_target():
                            break

                    if not perception.has_target():
                        perception.disable_capture(clear=True)
                        motion.clear_arm_trajectory()
                        workflow.transition(
                            ArmState.COMPLETE,
                            "joint_6 왕복 연속 탐색 중 남은 쓰레기 없음",
                        )
                        continue

                    points, frame, target_class = perception.take_target()
                    arm.wait_stable("continuous_search_detection_stop")
                    arm.normalize_joint_6_for_observation()
                    arm.wait_stable("joint_6_observation_normalized")
                    motion.start_arm_trajectory_recording(
                        "continuous_sweep_detection_to_pick"
                    )
                    workflow.initial_points = points
                    workflow.initial_frame = frame
                    workflow.target_class = target_class
                    workflow.coarse_target_position = (
                        task.estimate_target_position(points)
                    )
                    workflow.observation_pose = motion.build_observation_pose(
                        workflow.coarse_target_position
                    )
                    workflow.transition(ArmState.ALIGN_CAMERA)

                elif state == ArmState.ALIGN_CAMERA:
                    perception.disable_capture(clear=True)
                    (
                        observation_position,
                        observation_orientation,
                    ) = workflow.observation_pose
                    motion.open_gripper()
                    motion.move_to_observation_pose(
                        observation_position,
                        observation_orientation,
                    )
                    motion.verify_camera_looks_at(
                        workflow.coarse_target_position
                    )
                    workflow.transition(ArmState.REFINE_POINTCLOUD)

                elif state == ArmState.REFINE_POINTCLOUD:
                    settle_without_capture(
                        OBSERVATION_SETTLE_STEPS,
                        "observation_pose",
                    )
                    (
                        refined_points,
                        refined_frame,
                        refined_class,
                    ) = wait_for_pointcloud(
                        "refined",
                        REFINED_POINTCLOUD_TIMEOUT_S,
                        target_class=workflow.target_class,
                    )
                    if refined_points is None:
                        if motion.has_recorded_arm_trajectory():
                            motion.reverse_recorded_arm_trajectory()
                            arm.wait_stable("refine_timeout_reverse")
                        workflow.transition(
                            ArmState.MOVE_DETECTION_POSE,
                            "재관측 점군 timeout",
                        )
                        continue

                    workflow.grasp_plan = make_grasp_plan(
                        refined_class,
                        refined_points,
                        workflow.initial_frame,
                        refined_frame,
                    )
                    workflow.transition(ArmState.PICK)

                elif state == ArmState.PICK:
                    arm.pick(workflow.grasp_plan)
                    arm.wait_stable("pick_complete")
                    workflow.transition(ArmState.VERIFY_GRASP)

                elif state == ArmState.VERIFY_GRASP:
                    if arm.verify_grasp():
                        workflow.transition(ArmState.REVERSE_TRAJECTORY)
                    else:
                        workflow.grasp_attempts += 1
                        motion.open_gripper()
                        if motion.has_recorded_arm_trajectory():
                            motion.reverse_recorded_arm_trajectory()
                            arm.wait_stable("grasp_failure_reverse")
                        if workflow.grasp_attempts <= MAX_GRASP_RETRIES:
                            workflow.transition(
                                ArmState.MOVE_DETECTION_POSE,
                                f"파지 재시도 {workflow.grasp_attempts}",
                            )
                        else:
                            workflow.failure_reason = "파지 재시도 초과"
                            workflow.transition(ArmState.ERROR)

                elif state == ArmState.REVERSE_TRAJECTORY:
                    perception.disable_capture(clear=True)
                    motion.reverse_recorded_arm_trajectory()
                    arm.wait_stable("reverse_to_detection_complete")
                    workflow.transition(ArmState.ROTATE_TO_BIN)

                elif state == ArmState.ROTATE_TO_BIN:
                    perception.disable_capture(clear=True)
                    arm.rotate_joint_1_toward_bin(
                        workflow.grasp_plan.target_class
                    )
                    arm.wait_stable("joint_1_bin_rotation")
                    workflow.transition(ArmState.PLACE)

                elif state == ArmState.PLACE:
                    arm.place(workflow.grasp_plan.target_class)
                    workflow.collected_count += 1
                    workflow.grasp_attempts = 0
                    workflow.transition(ArmState.RETURN_DETECTION)

                elif state == ArmState.RETURN_DETECTION:
                    perception.disable_capture(clear=True)
                    arm.move_joint_1_only(0.0, "RETURN_DETECTION_JOINT_1")
                    arm.wait_stable("return_detection_pose")
                    arm.move_joint_6_search(
                        0.0,
                        "RETURN_DETECTION_JOINT_6_0",
                    )
                    arm.wait_stable("return_detection_joint_6_0")
                    motion.clear_arm_trajectory()
                    workflow.grasp_plan = None
                    workflow.transition(ArmState.WAIT_TARGET)

                elif state == ArmState.COMPLETE:
                    if not workflow._completion_announced:
                        signals.publish_complete()
                        workflow._completion_announced = True
                        print(
                            "[ARM WORKFLOW COMPLETE] "
                            f"collected_count={workflow.collected_count}"
                        )
                    elif not signals.is_aligned():
                        perception.disable_capture(clear=True)
                        motion.clear_arm_trajectory()
                        signals.reset_cycle()
                        workflow.reset()
                        print(
                            "[ARM WORKFLOW RESET] "
                            "다음 쓰레기 지점의 AMR 신호를 기다립니다"
                        )
                        continue
                    step_and_spin()

                elif state == ArmState.ERROR:
                    if not workflow._error_recovery_done:
                        perception.disable_capture(clear=True)
                        print(
                            "[ARM WORKFLOW ERROR] "
                            f"reason={workflow.failure_reason}"
                        )
                        try:
                            if motion.has_recorded_arm_trajectory():
                                motion.reverse_recorded_arm_trajectory()
                                arm.wait_stable("error_reverse_complete")
                                arm.move_joint_6_search(
                                    0.0,
                                    "ERROR_RETURN_JOINT_6_0",
                                )
                                arm.wait_stable("error_joint_6_0")
                            else:
                                arm.move_detection_pose()
                                arm.wait_stable("error_detection_pose")
                        except Exception as recovery_error:
                            print(
                                "[DETECTION POSE RECOVERY ERROR] "
                                f"{type(recovery_error).__name__}: "
                                f"{recovery_error}"
                            )
                        workflow._error_recovery_done = True
                    step_and_spin()

            except SimulationStopped as error:
                if not simulation_app.is_running():
                    break
                perception.disable_capture(clear=True)
                print(f"[TASK CANCELLED] {error}")
                restart_required = True

            except Exception as error:
                workflow.failure_reason = (
                    f"{type(error).__name__}: {error}"
                )
                workflow.transition(
                    ArmState.ERROR,
                    workflow.failure_reason,
                )

    finally:
        perception.disable_capture(clear=True)
        if world.is_playing():
            world.pause()
        ros_node.destroy_node()


def main():
    try:
        run()

    except Exception as error:
        print(f"{type(error).__name__}: {error}")
        raise

    finally:
        if rclpy.ok():
            rclpy.shutdown()

        simulation_app.close()


if __name__ == "__main__":
    main()
