#!/usr/bin/env python3
# cleanrobot_patrol_trash_scenario.py (CleanRobot 최종 시나리오 수동 연동 시험)
# 16개 웨이포인트 순찰 → 2/10번만 정면 탐지 대기 → 쓰레기 접근
# → Enter 2회로 매니퓰레이터 시작/완료 신호 모의 → 순찰 계속
#
# 비전 파이프라인 연동:
#   - 로봇이 발행: /front_stereo_camera/amr/rgb, /front_stereo_camera/amr/depth
#   - 비전 노드가 발행(우리가 구독): /amr/trash_target (std_msgs/String, JSON)
#       {"detected": true, "bbox_center_x": 324.5, "bbox_center_y": 210.0, "distance_m": 2.1834}
#
# 동작:
#   1) 모든 웨이포인트를 순서대로 이동하고 2/10번에서만 정면 탐지 대기
#   2) 2/10번에서 쓰레기 미탐지 시 다음 웨이포인트로 이동
#   3) 탐지 시 bbox 중심이 이미지 가운데 오도록 제자리 회전 정렬
#   4) 정렬되면 STOP_DISTANCE까지 접근, Enter 2회로 매니퓰레이터 동작을 모의한 뒤 순찰 재개

import json
import math
import select
import sys
import time

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# ===================== 조정 파라미터 =====================
IMAGE_WIDTH = 640          # RGB 이미지 가로 해상도(px). 카메라 설정에 맞게 수정
IMAGE_CENTER_X = IMAGE_WIDTH / 2.0

CENTER_TOLERANCE_PX = 25   # bbox 중심이 이 픽셀 이내면 "정렬됨"으로 간주
STOP_DISTANCE = 0.5       # 이 거리(m)까지 접근하면 정지 (로봇팔 집기 가능 거리)
DISTANCE_TOLERANCE = 0.05  # 목표 거리 허용 오차(m)

ANGULAR_KP = 0.003         # 정렬 회전 P 게인 (픽셀 오차 -> 각속도)
ALIGN_ANGULAR_SIGN = -1.0  # 화면 오른쪽 대상 -> 음의 yaw (현재 AMR 카메라 기준)
MAX_ANGULAR = 0.12         # 정렬 최대 각속도(rad/s)
ALIGN_STABLE_FRAMES = 3    # 중앙 판정이 연속으로 이 횟수 확인되어야 정렬 완료
ALIGN_TIMEOUT = 25.0       # 무한 정렬 방지 시간(s)
FORWARD_SPEED = 0.15       # 접근 직진 속도(m/s)

# 접근 중 방향 유지/미세 조향
APPROACH_ANGULAR_SIGN = -1.0  # 화면 오른쪽 대상은 음의 yaw (통합 AMR 카메라 기준)
APPROACH_VISUAL_KP = 0.0020   # bbox 픽셀 오차 -> 접근 중 각속도
HEADING_HOLD_KP = 1.5         # yaw 오차(rad) -> 각속도
MAX_APPROACH_ANGULAR = 0.12   # 접근 중 최대 각속도(rad/s)
APPROACH_VISION_MAX_AGE = 0.5 # 이 시간 이내 detection만 실시간 조향에 사용
APPROACH_RELOCK_PX = 60.0     # 이보다 벗어나면 전진을 멈추고 먼저 재정렬

DETECTION_TIMEOUT = 5.0    # 이 시간(s) 동안 detection 없으면 경고
CMD_VEL_TOPIC = '/cmd_vel'
TRASH_TOPIC = '/amr/trash_target'

# 기존 좌우 스캔 설정(최종 시나리오 main에서는 호출하지 않음)
SCAN_OFFSETS_DEG = (0.0, -20.0, 0.0, 20.0, 0.0)
SCAN_DWELL_TIME = 1.0      # 각 스캔 자세에서 탐지 대기 시간(s)
TRASH_WAYPOINT_WAIT = 5.0  # 2/10번 도착 자세에서 먼저 탐지할 시간(s)
TRASH_WAYPOINT_NUMBERS = {2, 10}
# =======================================================


# 기지(홈) 좌표 — Isaac Sim 실제 배치 좌표
HOME_X, HOME_Y, HOME_YAW = 2.0, -19.0, 90.0

# 공원 순찰 웨이포인트 — 현장에서 저장한 원본 좌표
WAYPOINTS = [
    ("웨이포인트1", 1.014, -16.684, 103.4),
    ("웨이포인트2", -1.176, -12.486, -143.922),
    ("웨이포인트3", -2.935, -6.758, 84.0),
    ("웨이포인트4", -0.400, -1.367, 60.0),
    ("웨이포인트5", 0.866, -1.65, 64.3),
    ("웨이포인트6", 0.427, 12.519, 82.0),
    ("웨이포인트7", 6.575, 12.230, -7.9),
    ("웨이포인트8", 8.496, 15.300, 69.8),
    ("웨이포인트9", 5.519, 18.945, 150.1),
    ("웨이포인트10", -1.585, 20.059, 169.4),
    ("웨이포인트11", -2.267, 18.250, -113.0),
    ("웨이포인트12", -6.177, 17.901, -176.7),
    ("웨이포인트13", -6.855, 13.431, -85.6),
    ("웨이포인트14", -1.005, 12.060, -42.8),
    ("웨이포인트15", -1.106, -4.976, -114.7),
    ("웨이포인트16", 0.103, -14.679, -75.8),
]


def get_quaternion_from_euler(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def create_pose(navigator, x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    q = get_quaternion_from_euler(0, 0, math.radians(yaw_deg))
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


class TrashApproachNode(Node):
    """/amr/trash_target 구독 + /cmd_vel 발행으로 쓰레기 정렬/접근을 담당"""

    def __init__(self):
        super().__init__('trash_approach')
        self.latest = None            # 최신 detection dict
        self.last_stamp = 0.0
        self._was_detected = False    # 직전 상태 (탐지 전이 로그용)

        # === Lock 관련 ===
        # 한 번 탐지되면 그 값을 스냅샷으로 잠그고, 이후 갱신 무시.
        # 재탐색이 필요하면 clear_lock() 호출.
        self.locked = None            # 잠긴 detection dict (None이면 아직 lock 안 됨)

        self.sub = self.create_subscription(
            String, TRASH_TOPIC, self._on_trash, 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        # 오도메트리 구독 — 실제 이동 거리 측정용
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self._on_odom, 10)
        self.robot_x = None
        self.robot_y = None

    def _on_odom(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        # yaw 추출 (odom quaternion -> yaw)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def get_position(self):
        if self.robot_x is None:
            return None
        return (self.robot_x, self.robot_y)

    def get_yaw(self):
        return getattr(self, 'robot_yaw', None)

    def _on_trash(self, msg):
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, TypeError):
            self.get_logger().warn(f"trash_target JSON 파싱 실패: {msg.data!r}")
            return
        self.latest = data
        self.last_stamp = time.time()

        # 아직 lock 안 됐고 이번 detection이 유효하면 → lock (스냅샷)
        now_detected = bool(data.get("detected", False))
        if self.locked is None and now_detected:
            # dict를 그대로 저장 — 이후 들어오는 값은 무시
            self.locked = dict(data)
            cx = data.get("bbox_center_x")
            cy = data.get("bbox_center_y")
            dist = data.get("distance_m")
            self.get_logger().info(
                f"🔒 좌표 LOCK - bbox=({cx}, {cy}), distance={dist} m "
                f"(이후 갱신은 무시하고 이 값으로 접근)")
            self._was_detected = True
            return

        # (참고 로그) — lock 걸린 뒤에는 그냥 스킵. 필요하면 상태 전이 로그 유지
        if self.locked is None:
            if now_detected and not self._was_detected:
                self.get_logger().info("👀 쓰레기 탐지됨 (lock 대기)")
            elif (not now_detected) and self._was_detected:
                self.get_logger().info("👀 쓰레기 탐지 끊김")
            self._was_detected = now_detected

    def clear_lock(self):
        """다시 새로운 detection을 받고 싶을 때 호출"""
        self.locked = None
        self.latest = None
        self.last_stamp = 0.0
        self._was_detected = False
        self.get_logger().info("🔓 lock 및 이전 detection 초기화")

    def stop(self):
        self.cmd_pub.publish(Twist())

    def send_cmd(self, linear_x=0.0, angular_z=0.0):
        t = Twist()
        t.linear.x = float(linear_x)
        t.angular.z = float(angular_z)
        self.cmd_pub.publish(t)

    def get_detection(self):
        """
        접근 로직이 사용할 detection 값.
        - lock이 걸려 있으면 그 스냅샷을 계속 반환 (갱신 무시)
        - 아직 lock 안 됐으면 최신 latest에서 유효한 값을 반환
        """
        if self.locked is not None:
            return self.locked

        if self.latest is None:
            return None
        if time.time() - self.last_stamp > DETECTION_TIMEOUT:
            return None
        if not self.latest.get("detected", False):
            return None
        return self.latest


def align_and_approach(node: TrashApproachNode):
    """
    정렬 단계: 실시간 bbox 수평 오차의 부호와 크기로 P 제어.
      - 검출 노이즈에 의한 회전 방향 반전을 없앰
      - 중앙 범위를 연속 프레임으로 확인하고 timeout으로 무한 회전 방지
    접근 단계: 정렬이 끝난 시점의 값을 lock 하고, 그 이후는 갱신 무시 + 오도메트리로 직진
      (여기는 원래 detection 깜빡임 때문에 문제가 있었던 구간이라 lock 유지)
    """
    node.get_logger().info("쓰레기 정렬 시작 (실시간 폐루프) — 첫 탐지 대기 중")

    # --- 첫 탐지 대기 ---
    wait_start = time.time()
    while rclpy.ok() and node.latest is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - wait_start > DETECTION_TIMEOUT * 3:
            node.get_logger().warn("첫 탐지 안 옴 — 접근 중단")
            node.stop()
            return False

    # --- bbox 오차 기반 연속 P 제어 정렬 ---
    align_deadline = time.monotonic() + ALIGN_TIMEOUT
    centered_frames = 0
    last_processed_stamp = None
    last_log_time = 0.0
    aligned = False

    while rclpy.ok() and time.monotonic() < align_deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

        det = node.latest
        detection_fresh = (time.time() - node.last_stamp) <= DETECTION_TIMEOUT
        if det is None or not det.get("detected", False) or not detection_fresh:
            centered_frames = 0
            node.stop()
            if time.monotonic() - last_log_time >= 0.5:
                node.get_logger().warn("정렬 중 탐지 끊김 — 새 detection 대기")
                last_log_time = time.monotonic()
            continue

        # 같은 메시지를 여러 번 spin해도 연속 프레임으로 세지 않음
        is_new_frame = node.last_stamp != last_processed_stamp
        err_px = float(det["bbox_center_x"]) - IMAGE_CENTER_X

        if abs(err_px) <= CENTER_TOLERANCE_PX:
            node.stop()
            if is_new_frame:
                centered_frames += 1
                last_processed_stamp = node.last_stamp
                node.get_logger().info(
                    f"  중앙 확인 {centered_frames}/{ALIGN_STABLE_FRAMES}: "
                    f"bbox_x={float(det['bbox_center_x']):.1f}, 오차={err_px:+.1f}px")
            if centered_frames >= ALIGN_STABLE_FRAMES:
                aligned = True
                break
            continue

        centered_frames = 0
        angular_z = clamp(
            ALIGN_ANGULAR_SIGN * ANGULAR_KP * err_px,
            -MAX_ANGULAR,
            MAX_ANGULAR)
        node.send_cmd(0.0, angular_z)
        if is_new_frame:
            last_processed_stamp = node.last_stamp
        if time.monotonic() - last_log_time >= 0.5:
            node.get_logger().info(
                f"  정렬 중: bbox_x={float(det['bbox_center_x']):.1f}, "
                f"오차={err_px:+.1f}px, angular_z={angular_z:+.3f}rad/s")
            last_log_time = time.monotonic()

    node.stop()
    if not aligned:
        node.get_logger().warn(
            f"정렬 {ALIGN_TIMEOUT:.0f}초 timeout — 무한 회전을 막고 접근을 취소합니다")
        return False

    node.get_logger().info(
        f"정렬 완료 (연속 {ALIGN_STABLE_FRAMES}프레임, "
        f"최대 각속도 {MAX_ANGULAR:.2f}rad/s)")

    # --- 정렬 완료 시점 값을 lock (여기서부터 접근 단계는 갱신 무시) ---
    final_det = node.latest
    node.locked = dict(final_det)
    target_dist = float(node.locked.get("distance_m", 0.0))
    node.get_logger().info(
        f"🔒 정렬 완료 시점 값 LOCK - distance={target_dist:.3f} m "
        f"(이후 접근 단계에서는 갱신 무시)")

    # --- 2단계: 직진 접근 (오도메트리 기반 실측, 락된 target_dist에서 실제 이동거리를 뺌) ---
    move_dist = max(0.0, target_dist - STOP_DISTANCE)  # 실제로 전진해야 할 거리
    node.get_logger().info(
        f"직진 접근 시작: {move_dist:.3f} m 전진 예정 "
        f"(쓰레기까지 {target_dist:.2f}m, 목표 앞 {STOP_DISTANCE:.2f}m에서 정지)")

    # 시작 위치 저장 (오도메트리가 아직 안 왔으면 기다림)
    wait_odom = time.time()
    while rclpy.ok() and node.get_position() is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - wait_odom > 3.0:
            node.get_logger().warn("오도메트리 안 옴 — 접근 중단")
            node.stop()
            return False

    start_x, start_y = node.get_position()
    target_heading = node.get_yaw()
    if target_heading is None:
        node.get_logger().warn("정렬 완료 yaw를 읽지 못함 — 접근 중 각속도 0으로 진행")
    else:
        node.get_logger().info(
            f"🧭 쓰레기 방향 yaw 저장: {math.degrees(target_heading):.1f}° "
            "(접근 중 bbox 우선, 탐지 끊김 시 이 yaw 유지)"
        )
    node.get_logger().info(f"시작 위치: ({start_x:.3f}, {start_y:.3f})")

    last_log = 0.0
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)

        cur = node.get_position()
        if cur is None:
            node.send_cmd(FORWARD_SPEED, 0.0)
            continue

        traveled = math.hypot(cur[0] - start_x, cur[1] - start_y)
        remaining_to_target = target_dist - traveled  # 락된 쓰레기까지 남은 예상 거리

        # 접근 중에는 최신 bbox로 쓰레기를 계속 화면 중앙에 유지한다.
        # detection이 순간적으로 끊기면 정렬 완료 때 저장한 yaw를 유지한다.
        current_heading = node.get_yaw()
        linear_cmd = FORWARD_SPEED
        angular_cmd = 0.0
        steering_mode = "직진"
        bbox_error = None
        heading_error = None

        latest = node.latest
        vision_fresh = (
            latest is not None
            and bool(latest.get("detected", False))
            and time.time() - node.last_stamp <= APPROACH_VISION_MAX_AGE
            and "bbox_center_x" in latest
        )

        if vision_fresh:
            bbox_error = float(latest["bbox_center_x"]) - IMAGE_CENTER_X
            angular_cmd = clamp(
                APPROACH_ANGULAR_SIGN * APPROACH_VISUAL_KP * bbox_error,
                -MAX_APPROACH_ANGULAR,
                MAX_APPROACH_ANGULAR,
            )
            steering_mode = "bbox 추종"

            # 대상이 화면 중앙에서 크게 벗어나면 더 멀어지기 전에 제자리 재정렬한다.
            if abs(bbox_error) > APPROACH_RELOCK_PX:
                linear_cmd = 0.0
                steering_mode = "bbox 재정렬(전진 정지)"

            # 중앙에 들어온 순간의 실제 yaw를 새 쓰레기 방향으로 계속 갱신한다.
            if abs(bbox_error) <= CENTER_TOLERANCE_PX and current_heading is not None:
                target_heading = current_heading
        elif target_heading is not None and current_heading is not None:
            heading_error = normalize_angle(target_heading - current_heading)
            angular_cmd = clamp(
                HEADING_HOLD_KP * heading_error,
                -MAX_APPROACH_ANGULAR,
                MAX_APPROACH_ANGULAR,
            )
            steering_mode = "저장 yaw 유지"

        # 0.5초마다 실시간 로그
        now = time.time()
        if now - last_log > 0.5:
            direction_log = (
                f"조향={steering_mode}, linear={linear_cmd:.2f} m/s, "
                f"angular={angular_cmd:+.3f} rad/s"
            )
            if bbox_error is not None:
                direction_log += f", bbox 오차={bbox_error:+.1f}px"
            elif heading_error is not None:
                direction_log += f", yaw 오차={math.degrees(heading_error):+.1f}°"
            node.get_logger().info(
                f"  이동 {traveled:.3f} m / {move_dist:.3f} m  "
                f"(쓰레기까지 예상 거리: {remaining_to_target:.3f} m)  "
                f"{direction_log}")
            last_log = now

        # 목표 이동 거리 도달 → 정지
        if traveled >= move_dist:
            node.stop()
            node.get_logger().info(
                f"🅿️  목표 지점 도달 (이동 {traveled:.3f} m, "
                f"쓰레기까지 예상 {remaining_to_target:.3f} m)")
            break

        node.send_cmd(linear_cmd, angular_cmd)

    node.stop()
    return True


def scan_waypoint(nav, node, label, x, y, base_yaw_deg):
    """기존 좌우 스캔 함수. 최종 시나리오에서는 호출하지 않는다."""
    for index, offset_deg in enumerate(SCAN_OFFSETS_DEG):
        scan_yaw = base_yaw_deg + offset_deg

        # 첫 자세(0도)는 웨이포인트 도착 자세이므로 중복 목표를 보내지 않는다.
        if index > 0:
            print(f"  🔎 [{label}] 스캔 회전 → {scan_yaw:.1f}°")
            nav.goToPose(create_pose(nav, x, y, scan_yaw))

            while not nav.isTaskComplete():
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.get_detection() is not None:
                    nav.cancelTask()
                    while not nav.isTaskComplete():
                        time.sleep(0.05)
                    node.stop()
                    print(f"  🗑️ [{label}] 회전 중 쓰레기 탐지")
                    return True

            if nav.getResult() != TaskResult.SUCCEEDED:
                print(f"  ⚠️ [{label}] {scan_yaw:.1f}° 스캔 자세 회전 실패 — 다음 자세 계속")

        print(f"  👀 [{label}] {scan_yaw:.1f}° 방향 탐지 중")
        scan_start = time.time()
        while rclpy.ok() and time.time() - scan_start < SCAN_DWELL_TIME:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.get_detection() is not None:
                node.stop()
                print(f"  🗑️ [{label}] 쓰레기 탐지")
                return True

    print(f"  ✅ [{label}] 스캔 완료 — 쓰레기 미탐지")
    return False


def wait_for_detection_at_trash_waypoint(node, label):
    """2/10번 도착 자세를 유지하며 일정 시간 새 쓰레기 탐지를 기다린다."""
    print(
        f"  ⏳ [{label}] 쓰레기 지정 지점 — "
        f"도착 자세에서 {TRASH_WAYPOINT_WAIT:.0f}초 탐지 대기"
    )
    start = time.time()
    last_second = None

    while rclpy.ok() and time.time() - start < TRASH_WAYPOINT_WAIT:
        rclpy.spin_once(node, timeout_sec=0.1)
        node.stop()

        if node.get_detection() is not None:
            elapsed = time.time() - start
            print(f"  🗑️ [{label}] {elapsed:.1f}초 후 쓰레기 탐지")
            return True

        remaining = max(0, math.ceil(TRASH_WAYPOINT_WAIT - (time.time() - start)))
        if remaining != last_second:
            print(f"     탐지 대기 중... {remaining}초")
            last_second = remaining

    print(f"  ℹ️ [{label}] 정면 탐지 대기 종료 — 쓰레기 미탐지")
    return False


def wait_for_operator_enter(node, message):
    """Enter 입력을 기다리는 동안 ROS callback을 처리하고 정지 명령을 유지한다."""
    print(f"\n{message}")
    print("  ▶ Enter 키를 누르세요. (Ctrl+C: 전체 종료)")

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        node.stop()

        readable, _, _ = select.select([sys.stdin], [], [], 0.1)
        if readable:
            line = sys.stdin.readline()
            if line == "":
                node.get_logger().error("터미널 입력이 닫혀 수동 시험을 계속할 수 없습니다.")
                return False
            return True

    return False


def simulate_manipulator_sequence(node, label):
    """실제 매니퓰레이터 토픽 대신 Enter 2회로 시작/완료 handshake를 모의한다."""
    node.stop()
    node.get_logger().info(
        f"[{label}] AMR 정지 완료 — 매니퓰레이터 시작 신호 수동 모의 대기"
    )

    if not wait_for_operator_enter(
        node,
        "🤖 [1/2] 매니퓰레이터에 동작 시작 신호를 보낼 시점입니다.",
    ):
        return False

    node.get_logger().info(
        f"[{label}] [가상 신호] 매니퓰레이터 동작 시작: "
        "카메라 탐지 → 파지 → 쓰레기통 투입"
    )
    print("  🤖 매니퓰레이터가 쓰레기를 탐지하고 집어 쓰레기통에 넣는 것으로 가정합니다.")

    if not wait_for_operator_enter(
        node,
        "✅ [2/2] 투입 완료 후 남은 쓰레기가 없다고 가정하려면",
    ):
        return False

    node.get_logger().info(
        f"[{label}] [가상 신호] 매니퓰레이터 작업 완료 + 남은 쓰레기 없음"
    )
    print("  📡 AMR이 '쓰레기 없음/작업 완료' 신호를 받은 것으로 가정합니다.")
    return True


def return_home(nav):
    """동적 TF가 아닌 기존 map 좌표계의 고정 홈 자세로 복귀한다."""
    print("\n🏠 초기 좌표로 복귀")
    nav.goToPose(create_pose(nav, HOME_X, HOME_Y, HOME_YAW))

    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            print(f"  [초기 좌표 복귀] 남은 거리: {fb.distance_remaining:.2f} m")
        time.sleep(1.0)

    home_result = nav.getResult()
    if home_result == TaskResult.SUCCEEDED:
        print("🏠 초기 좌표 복귀 완료")
        return True

    print(f"❌ 초기 좌표 복귀 실패 (result={home_result})")
    return False


def main():
    rclpy.init()

    nav = BasicNavigator()

    # 초기 위치 설정 및 Nav2 활성화
    init_pose = create_pose(nav, HOME_X, HOME_Y, HOME_YAW)
    nav.setInitialPose(init_pose)
    nav.waitUntilNav2Active()
    print("✅ Nav2 활성화 완료")

    approach_node = TrashApproachNode()
    try:
        approach_failed = False

        # --- 1) 16개 웨이포인트 순찰 (2/10번만 도착 자세에서 탐지 대기) ---
        for waypoint_number, (label, x, y, yaw_deg) in enumerate(WAYPOINTS, start=1):
            print(f"\n🚀 [{label}] 로 이동 → ({x}, {y}, {yaw_deg}°)")
            nav.goToPose(create_pose(nav, x, y, yaw_deg))

            while not nav.isTaskComplete():
                fb = nav.getFeedback()
                if fb:
                    print(f"  [{label}] 남은 거리: {fb.distance_remaining:.2f} m")
                time.sleep(1.0)

            result = nav.getResult()
            if result != TaskResult.SUCCEEDED:
                print(f"❌ [{label}] 이동 실패 (result={result}) — 종료")
                nav.lifecycleShutdown()
                return

            print(f"🎉 [{label}] 도착")

            if waypoint_number not in TRASH_WAYPOINT_NUMBERS:
                print(f"➡️  [{label}] 일반 통과 지점 — 바로 다음 웨이포인트로 이동")
                continue

            # 이동 중 잡힌 오래된 detection lock은 버리고 이 지점에서 새로 탐지한다.
            approach_node.clear_lock()
            detected = wait_for_detection_at_trash_waypoint(approach_node, label)

            if detected:
                print("\n🗑️ 쓰레기 탐지 — 정렬/접근 단계로 전환")
                ok = align_and_approach(approach_node)
                approach_node.stop()

                if not ok:
                    approach_failed = True
                    print("\n⚠️  쓰레기 접근 실패 — 대기 상태")
                    break

                print(
                    f"\n🅿️  쓰레기 앞 {STOP_DISTANCE:.2f}m 임계거리 도달 — "
                    "AMR 동작 정지"
                )
                if not simulate_manipulator_sequence(approach_node, label):
                    approach_failed = True
                    print("\n⚠️  매니퓰레이터 수동 모의 입력 실패 — 순찰 중단")
                    break

                approach_node.clear_lock()
                print("➡️  매니퓰레이터 완료 신호 확인 — 다음 웨이포인트 순찰 재개")
            else:
                approach_node.clear_lock()
                print(f"➡️  [{label}] 쓰레기 미탐지 — 다음 웨이포인트로 이동")

        # --- 전체 웨이포인트 처리 후 고정 홈 좌표로 복귀 ---
        if approach_failed:
            print("\n⚠️  쓰레기 접근 실패로 순찰 중단 — 대기 상태")
        else:
            print("\n✅ 전체 웨이포인트 순찰 완료")
            return_home(nav)

        # --- 5) 기존 대기 상태 유지 ---
        print("대기 중... (Ctrl+C로 종료)")
        while rclpy.ok():
            rclpy.spin_once(approach_node, timeout_sec=0.5)
            approach_node.stop()  # 안전: 대기 중 정지 유지
    except KeyboardInterrupt:
        pass
    finally:
        approach_node.stop()
        approach_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
