#!/usr/bin/env python3
"""기존 WP2 수거 -> 새 WP1~6 순찰/WP6 수거 -> 전체 역순 -> Home."""

import importlib.util
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

os.environ["ROS_DOMAIN_ID"] = "108"
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")

import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


INTEGRATION_ROOT = Path(
    os.environ.get("CLEANROBOT_INTEGRATION_ROOT", str(Path(__file__).resolve().parent))
).expanduser().resolve()
PROJECT_ROOT = INTEGRATION_ROOT.parents[2]
COBOT3_WS_ROOT = INTEGRATION_ROOT / "cobot3_ws"
PNPTEST = COBOT3_WS_ROOT / "rokey_cobot3" / "서영채" / "pnptest1.py"
AMR_SOURCE = (
    PROJECT_ROOT
    / "ros2_ws"
    / "src"
    / "cleanrobot_amr_navigation"
    / "scripts"
    / "cleanrobot_patrol_trash_scenario.py"
)
ARM_READY_TIMEOUT_S = 120.0
FIRST_TRASH_STOP_DISTANCE_M = 0.30
SECOND_TRASH_STOP_DISTANCE_M = 0.27
INTEGRATION_ALIGN_ANGULAR_KP = 0.0045
INTEGRATION_MAX_ANGULAR_RAD_S = 0.35
BACKUP_SPEED_MPS = 0.10
BACKUP_SEGMENT_M = 0.10
BACKUP_COSTMAP_SETTLE_S = 0.5
BACKUP_RETRY_COUNT = 1
INTEGRATION_CMD_VEL_TOPIC = "/cmd_vel_nav"
NEW_WAYPOINTS = (
    ("웨이포인트1", -3.426, -6.987, 75.4),
    ("웨이포인트2", -0.270, -1.289, 57.0),
    ("웨이포인트3", 0.042, 5.121, 94.5),
    ("웨이포인트4", -0.328, 12.477, 94.6),
    ("웨이포인트5", -0.627, 19.483, 170.8),
    ("웨이포인트6", -2.048, 19.783, 174.3),
)


def load_existing_amr_code():
    spec = importlib.util.spec_from_file_location(
        "cleanrobot_existing_amr_flow", AMR_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"기존 AMR 코드를 불러올 수 없습니다: {AMR_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArmHandshake(Node):
    def __init__(self):
        super().__init__("wp2_arm_handshake")
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.arm_ready = False
        self.arm_complete = False
        self._aligned_pub = self.create_publisher(
            Bool, "/amr/arm_alignment_complete", qos
        )
        self._complete_sub = self.create_subscription(
            Bool, "/robot_arm/task_complete", self._on_complete, qos
        )

    def _on_complete(self, message):
        if message.data:
            self.arm_complete = True
            self.get_logger().info("매니퓰레이터 작업 완료 신호 수신")
        else:
            self.arm_ready = True
            self.arm_complete = False
            self.get_logger().info("매니퓰레이터 준비 신호(False) 수신")

    def publish_arrived(self, arrived):
        self._aligned_pub.publish(Bool(data=bool(arrived)))
        self.get_logger().info(
            f"/amr/arm_alignment_complete={bool(arrived)} 발행"
        )

    def prepare_next_cycle(self):
        self.arm_ready = False
        self.arm_complete = False
        self.publish_arrived(False)


def validate_paths(require_isaac):
    isaac_root_value = os.environ.get("ISAAC_SIM_ROOT")
    if require_isaac and not isaac_root_value:
        raise RuntimeError(
            "Isaac Sim release 경로를 ISAAC_SIM_ROOT로 지정하세요."
        )
    isaac_python = (
        Path(isaac_root_value).expanduser().resolve() / "python.sh"
        if isaac_root_value
        else None
    )
    required = [
        INTEGRATION_ROOT
        / "isaac_sim/Integrated_CleanRobot_Personal_D112/CleanRobot_Park.usda",
        PNPTEST,
        AMR_SOURCE,
    ]
    if require_isaac:
        required.append(isaac_python)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("필수 파일 누락:\n  " + "\n  ".join(missing))
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError(
            "ros2를 찾지 못했습니다. ROS 2 Humble과 cobot3_ws를 source한 뒤 실행하세요."
        )
    return isaac_python, ros2


def start_processes(isaac_python, ros2):
    ros_env = os.environ.copy()
    ros_env.update(
        {
            "CLEANROBOT_INTEGRATION_ROOT": str(INTEGRATION_ROOT),
            "COBOT3_WS_ROOT": str(COBOT3_WS_ROOT),
            "ROS_DOMAIN_ID": "108",
            "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        }
    )
    isaac_env = ros_env.copy()
    # Isaac Python은 3.11이므로 source된 ROS Humble Python 3.10 경로를
    # 상속하면 rclpy C 확장을 불러올 수 없다. Isaac 내장 rclpy만 사용한다.
    for name in (
        "PYTHONPATH",
        "OLD_PYTHONPATH",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
    ):
        isaac_env.pop(name, None)
    remaining_library_paths = [
        path
        for path in ros_env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path and "/opt/ros/" not in path and "/cobot3_ws/" not in path
    ]
    internal_ros_lib = (
        Path(isaac_python).parent
        / "exts"
        / "isaacsim.ros2.bridge"
        / "humble"
        / "lib"
    )
    isaac_env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(internal_ros_lib), *remaining_library_paths]
    )
    print("[START] Isaac Sim + pnptest1.py", flush=True)
    isaac_process = subprocess.Popen(
        [str(isaac_python), str(PNPTEST)],
        env=isaac_env,
        start_new_session=True,
    )
    print("[START] 기존 cleanrobot_amr_bringup.launch.py", flush=True)
    try:
        nav2_process = subprocess.Popen(
            [
                ros2,
                "launch",
                "cleanrobot_amr_navigation",
                "cleanrobot_amr_bringup.launch.py",
            ],
            env=ros_env,
            start_new_session=True,
        )
    except Exception:
        stop_process(isaac_process)
        raise
    return isaac_process, nav2_process


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_for_arm_ready(handshake, isaac_process):
    deadline = time.monotonic() + ARM_READY_TIMEOUT_S
    while rclpy.ok() and time.monotonic() < deadline:
        if isaac_process is not None and isaac_process.poll() is not None:
            raise RuntimeError(
                f"pnptest1.py가 준비 전에 종료되었습니다: code={isaac_process.returncode}"
            )
        rclpy.spin_once(handshake, timeout_sec=0.1)
        if handshake.arm_ready:
            return
    raise TimeoutError("매니퓰레이터 준비 신호(False)를 받지 못했습니다")


def go_to_waypoint(nav, amr, waypoint):
    label, x, y, yaw_deg = waypoint
    print(f"\n🚀 [{label}]로 이동 → ({x:.3f}, {y:.3f}, {yaw_deg:.1f}°)")
    nav.goToPose(amr.create_pose(nav, x, y, yaw_deg))
    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f"  [{label}] 남은 거리: {feedback.distance_remaining:.2f} m")
        time.sleep(1.0)
    if nav.getResult() != TaskResult.SUCCEEDED:
        print(f"❌ [{label}] 이동 실패")
        return None
    print(f"🎉 [{label}] 도착")
    return True


def opposite_yaw(yaw_deg):
    return math.degrees(
        math.atan2(
            math.sin(math.radians(yaw_deg + 180.0)),
            math.cos(math.radians(yaw_deg + 180.0)),
        )
    )


def return_waypoints(amr):
    outbound = (*amr.WAYPOINTS[:2], *NEW_WAYPOINTS)
    return tuple(
        (f"{label} 복귀", x, y, opposite_yaw(yaw_deg))
        for label, x, y, yaw_deg in reversed(outbound[:-1])
    )


def wait_for_arm_complete(handshake, approach_node, isaac_process):
    while rclpy.ok():
        if isaac_process is not None and isaac_process.poll() is not None:
            raise RuntimeError(
                f"pnptest1.py가 작업 완료 전에 종료되었습니다: code={isaac_process.returncode}"
            )
        rclpy.spin_once(handshake, timeout_sec=0.1)
        approach_node.stop()
        if handshake.arm_complete:
            return


def position_distance(start, end):
    if start is None or end is None:
        raise RuntimeError("후진 거리 계산에 필요한 odom 위치가 없습니다")
    return math.hypot(end[0] - start[0], end[1] - start[1])


def backup_approach_distance(nav, distance):
    if distance <= 0.01:
        print("[BACKUP] 전진 거리가 없어 후진을 생략합니다", flush=True)
        return True

    segment_count = math.ceil(distance / BACKUP_SEGMENT_M)
    print(
        f"[BACKUP] 접근 거리 {distance:.3f}m 후진 "
        f"(speed={BACKUP_SPEED_MPS:.2f}m/s, "
        f"segment<={BACKUP_SEGMENT_M:.2f}m, count={segment_count})",
        flush=True,
    )

    print("[BACKUP COSTMAP] local costmap 초기화 및 재갱신", flush=True)
    nav.clearLocalCostmap()
    time.sleep(BACKUP_COSTMAP_SETTLE_S)

    completed_distance = 0.0
    for segment_index in range(1, segment_count + 1):
        segment_distance = min(
            BACKUP_SEGMENT_M,
            distance - completed_distance,
        )
        segment_succeeded = False
        for attempt in range(BACKUP_RETRY_COUNT + 1):
            if attempt:
                print(
                    f"[BACKUP RETRY] segment={segment_index}/{segment_count}, "
                    "local costmap 재초기화",
                    flush=True,
                )
                nav.clearLocalCostmap()
                time.sleep(BACKUP_COSTMAP_SETTLE_S)

            time_allowance = max(
                5,
                math.ceil(segment_distance / BACKUP_SPEED_MPS) + 3,
            )
            print(
                f"[BACKUP SEGMENT] {segment_index}/{segment_count}, "
                f"distance={segment_distance:.3f}m, attempt={attempt + 1}",
                flush=True,
            )
            if not nav.backup(
                backup_dist=segment_distance,
                backup_speed=BACKUP_SPEED_MPS,
                time_allowance=time_allowance,
            ):
                print("[BACKUP] Nav2가 후진 명령을 거부했습니다", flush=True)
                continue

            while not nav.isTaskComplete():
                feedback = nav.getFeedback()
                if feedback is not None and hasattr(feedback, "distance_traveled"):
                    print(
                        f"  구간 후진: {feedback.distance_traveled:.3f}m / "
                        f"{segment_distance:.3f}m",
                        flush=True,
                    )
                time.sleep(0.2)

            if nav.getResult() == TaskResult.SUCCEEDED:
                segment_succeeded = True
                break
            print(
                f"[BACKUP COLLISION] segment={segment_index}/{segment_count}, "
                f"attempt={attempt + 1} 중단",
                flush=True,
            )

        if not segment_succeeded:
            print(
                "[BACKUP FAILED] 후방 충돌 판정이 계속되어 "
                "Home으로 이동하지 않습니다",
                flush=True,
            )
            return False

        completed_distance += segment_distance
        print(
            f"[BACKUP PROGRESS] {completed_distance:.3f}m / {distance:.3f}m",
            flush=True,
        )

    print("[BACKUP COMPLETE] 접근 전 위치까지 후진 완료", flush=True)
    return True


def collect_trash(
    nav, amr, approach_node, handshake, isaac_process, label, stop_distance_m
):
    amr.STOP_DISTANCE = stop_distance_m
    print(f"[TRASH APPROACH] {label} 정지거리={stop_distance_m:.2f}m")
    wait_for_arm_ready(handshake, isaac_process)
    approach_node.clear_lock()
    if not amr.wait_for_detection_at_trash_waypoint(approach_node, label):
        print(f"❌ [{label}]에서 쓰레기를 찾지 못했습니다")
        return False

    print(f"\n🗑️ [{label}] 기존 AMR 정렬·접근 로직 시작")
    approach_start = approach_node.get_position()
    arrived_at_threshold = amr.align_and_approach(approach_node)
    approach_node.stop()
    approach_end = approach_node.get_position()
    if not arrived_at_threshold:
        print("❌ AMR이 임계거리에 도착하지 못해 팔을 실행하지 않습니다")
        return False

    print("\n🅿️ AMR 임계거리 도착 → 매니퓰레이터 시작 신호 전송")
    handshake.publish_arrived(True)
    wait_for_arm_complete(handshake, approach_node, isaac_process)
    approached_distance = position_distance(approach_start, approach_end)
    print(
        "\n✅ 매니퓰레이터 완료 → "
        f"접근 거리 {approached_distance:.3f}m 후진",
        flush=True,
    )
    if not backup_approach_distance(nav, approached_distance):
        approach_node.stop()
        return False

    handshake.prepare_next_cycle()
    print(f"✅ [{label}] 수거·후진 완료 → 순찰 재개", flush=True)
    return True


def run_test(amr, isaac_process):
    rclpy.init()
    # Nav2 velocity_smoother 입력으로 보내 /cmd_vel 다중 발행 충돌을 피한다.
    amr.CMD_VEL_TOPIC = INTEGRATION_CMD_VEL_TOPIC
    nav = BasicNavigator()
    approach_node = amr.TrashApproachNode()
    handshake = ArmHandshake()
    try:
        # 원본 AMR 파일은 바꾸지 않고 이 통합 시험에서만 팔 작업 거리로 적용한다.
        amr.STOP_DISTANCE = FIRST_TRASH_STOP_DISTANCE_M
        amr.ANGULAR_KP = INTEGRATION_ALIGN_ANGULAR_KP
        amr.MAX_ANGULAR = INTEGRATION_MAX_ANGULAR_RAD_S
        print(
            f"[INTEGRATION] AMR 정지거리={amr.STOP_DISTANCE:.2f}m, "
            f"정렬 P게인={amr.ANGULAR_KP:.4f}, "
            f"정렬 최대 각속도={amr.MAX_ANGULAR:.2f}rad/s, "
            f"접근 속도={amr.FORWARD_SPEED:.2f}m/s(원본 유지), "
            f"접근 명령={amr.CMD_VEL_TOPIC}",
            flush=True,
        )
        handshake.publish_arrived(False)
        nav.setInitialPose(amr.create_pose(nav, amr.HOME_X, amr.HOME_Y, amr.HOME_YAW))
        print("[WAIT] Nav2 활성화", flush=True)
        nav.waitUntilNav2Active()
        print("[WAIT] 매니퓰레이터 준비", flush=True)
        wait_for_arm_ready(handshake, isaac_process)

        print("\n[1차 경로] 기존 WP1 → 기존 WP2(첫 번째 쓰레기)", flush=True)
        for waypoint_number, waypoint in enumerate(amr.WAYPOINTS[:2], start=1):
            if not go_to_waypoint(nav, amr, waypoint):
                return False
            if waypoint_number == 1:
                print("➡️  기존 일반 경유지 — 기존 WP2로 이동")
            elif not collect_trash(
                    nav, amr, approach_node, handshake, isaac_process,
                    waypoint[0], FIRST_TRASH_STOP_DISTANCE_M,
                ):
                    return False

        print("\n[2차 경로] 새 WP1 → 새 WP6(두 번째 쓰레기)", flush=True)
        for waypoint_number, waypoint in enumerate(NEW_WAYPOINTS, start=1):
            if not go_to_waypoint(nav, amr, waypoint):
                return False
            if waypoint_number < 6:
                print("➡️  새 일반 순회 지점 — 바로 다음 웨이포인트로 이동")
            elif not collect_trash(
                    nav, amr, approach_node, handshake, isaac_process,
                    waypoint[0], SECOND_TRASH_STOP_DISTANCE_M,
                ):
                    return False

        print("\n🔄 새 WP6 수거 완료 → 전체 경로를 반대 방향 자세로 역순 복귀", flush=True)
        for waypoint in return_waypoints(amr):
            if not go_to_waypoint(nav, amr, waypoint):
                return False
            print("⬅️  복귀 경유지 — 쓰레기 로직 없이 통과")

        print("\n🏠 역순 경로 완료 → AMR Home 복귀", flush=True)
        return amr.return_home(nav)
    finally:
        if rclpy.ok():
            approach_node.stop()
            nav.lifecycleShutdown()
        handshake.destroy_node()
        approach_node.destroy_node()
        nav.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    control_only = sys.argv[1:] == ["--control-only"]
    check_only = sys.argv[1:] == ["--check"]
    isaac_python, ros2 = validate_paths(
        require_isaac=not (control_only or check_only)
    )
    amr = load_existing_amr_code()
    if check_only:
        assert len(NEW_WAYPOINTS) == 6
        assert amr.WAYPOINTS[:2] == [
            ("웨이포인트1", 1.014, -16.684, 103.4),
            ("웨이포인트2", -1.176, -12.486, -143.922),
        ]
        assert NEW_WAYPOINTS[1] == (
            "웨이포인트2", -0.270, -1.289, 57.0
        )
        assert NEW_WAYPOINTS[5] == (
            "웨이포인트6", -2.048, 19.783, 174.3
        )
        assert [round(wp[3], 1) for wp in return_waypoints(amr)] == [
            -9.2, -85.4, -85.5, -123.0, -104.6, 36.1, -76.6
        ]
        assert position_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
        assert FIRST_TRASH_STOP_DISTANCE_M == 0.30
        assert SECOND_TRASH_STOP_DISTANCE_M == 0.27
        assert INTEGRATION_ALIGN_ANGULAR_KP == 0.0045
        assert INTEGRATION_MAX_ANGULAR_RAD_S == 0.35
        assert BACKUP_SEGMENT_M == 0.10
        assert BACKUP_COSTMAP_SETTLE_S == 0.5
        assert BACKUP_RETRY_COUNT == 1
        assert INTEGRATION_CMD_VEL_TOPIC == "/cmd_vel_nav"
        print(f"CHECK_OK: 기존 정렬 함수 재사용={amr.align_and_approach.__code__.co_filename}")
        print(f"CHECK_OK: pnptest={PNPTEST}")
        print("CHECK_OK: 기존 WP1→WP2 수거 후 새 WP1→WP6 수거")
        print("CHECK_OK: 새 WP5→WP1→기존 WP2→WP1 역순 후 Home")
        print("CHECK_OK: 첫 쓰레기=0.30m, 두 번째 쓰레기=0.27m")
        print("CHECK_OK: local costmap 초기화 + 0.10m 구간 후진 + 1회 재시도")
        print("CHECK_OK: 접근 명령=/cmd_vel_nav -> velocity_smoother -> /cmd_vel")
        return 0
    if control_only:
        print("[MODE] 외부에서 실행한 Isaac과 Nav2에 연결합니다", flush=True)
        try:
            return 0 if run_test(amr, None) else 1
        except KeyboardInterrupt:
            print("\n사용자 중단 — AMR 제어를 안전하게 종료합니다")
            return 130

    isaac_process = None
    nav2_process = None
    try:
        isaac_process, nav2_process = start_processes(isaac_python, ros2)
        return 0 if run_test(amr, isaac_process) else 1
    except KeyboardInterrupt:
        print("\n사용자 중단 — AMR 정지 후 실행 프로세스를 종료합니다")
        return 130
    finally:
        stop_process(nav2_process)
        stop_process(isaac_process)


if __name__ == "__main__":
    raise SystemExit(main())
