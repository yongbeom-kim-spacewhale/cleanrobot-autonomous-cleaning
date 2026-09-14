# ROS 2 토픽 흐름

모든 장치는 `ROS_DOMAIN_ID=108`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`를 사용한다.

## 외부 비전 계약

| 방향 | 토픽 | 타입 | 필수 조건 |
|---|---|---|---|
| YOLO → AMR | `/amr/trash_target` | `std_msgs/String` | JSON에 `detected`, `bbox_center_x`, `bbox_center_y`, `distance_m` 포함 |
| YOLO → 팔 | `/robot_arm/trash_points/class_0` | `sensor_msgs/PointCloud2` | 캔, `frame_id=map` |
| YOLO → 팔 | `/robot_arm/trash_points/class_2` | `sensor_msgs/PointCloud2` | 플라스틱, `frame_id=map` |
| Isaac → YOLO | `/front_stereo_camera/amr/rgb` | 이미지 토픽 | AMR RGB |
| Isaac → YOLO | `/front_stereo_camera/amr/depth` | 깊이 토픽 | AMR Depth |
| Isaac → YOLO | `/robot_arm/rgb` | 이미지 토픽 | 손목 RGB |
| Isaac → YOLO | `/robot_arm/depth` | 깊이 토픽 | 손목 Depth |

## 내부 제어 계약

| 토픽 | 타입 | 역할 |
|---|---|---|
| `/cmd_vel_nav` | `geometry_msgs/Twist` | Nav2 및 AMR 직접 접근 명령 |
| `/cmd_vel` | `geometry_msgs/Twist` | velocity smoother 최종 출력, Isaac 입력 |
| `/amr/arm_alignment_complete` | `std_msgs/Bool` | AMR 접근 완료 신호 |
| `/robot_arm/task_complete` | `std_msgs/Bool` | 팔 준비(False) 및 작업 완료(True) |
| `/cobot3/front/scan` | `sensor_msgs/LaserScan` | 전방 self-filter LiDAR |
| `/cobot3/rear/scan` | `sensor_msgs/LaserScan` | 후방 self-filter LiDAR |

## 신호 순서

```text
팔 시작 → /robot_arm/task_complete=False
AMR 접근 완료 → /amr/arm_alignment_complete=True
팔 수거 완료 → /robot_arm/task_complete=True
AMR 다음 사이클 준비 → /amr/arm_alignment_complete=False
```
