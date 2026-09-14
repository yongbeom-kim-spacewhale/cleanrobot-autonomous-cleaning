# cleanrobot_nav2_core.launch.py
# CleanRobot용 Nav2 launch (Nav2 base=tf_amr, odom=/odom, front/rear 2D lidar)
# 새 Cobot3 로봇 (front/rear 2D LiDAR: /cobot3/front, /cobot3/rear)
# 기존 패키지와 충돌하지 않는 CleanRobot 전용 실행 진입점

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("cleanrobot_amr_navigation")
    nav2_bringup_share = get_package_share_directory("nav2_bringup")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_file = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")

    keepout_mask_yaml = os.path.join(
        pkg_share, "maps", "cleanrobot_keepout_mask.yaml"
    )

    rviz_config_dir = os.path.join(
        pkg_share, "rviz2", "cleanrobot_amr_navigation.rviz"
    )
    nav2_bringup_launch_dir = os.path.join(nav2_bringup_share, "launch")

    lifecycle_nodes_filters = ["filter_mask_server", "costmap_filter_info_server"]

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            default_value=os.path.join(pkg_share, "maps", "cleanrobot_park_map.yaml"),
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(pkg_share, "params", "cleanrobot_amr_nav2.yaml"),
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        # RViz2 자동 실행
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_launch_dir, "rviz_launch.py")
            ),
            launch_arguments={
                "namespace": "",
                "use_namespace": "False",
                "rviz_config": rviz_config_dir,
            }.items(),
        ),

        # Nav2 전체 스택
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_launch_dir, "bringup_launch.py")
            ),
            launch_arguments={
                "map": map_file,
                "use_sim_time": use_sim_time,
                "params_file": params_file,
            }.items(),
        ),

        # LiDAR static TF는 여기서 발행하지 않는다.
        # Isaac USD가 odom -> tf_amr -> chassis_link -> lidar_link 트리를 소유한다.
        # base_link는 매니퓰레이터 프레임으로만 사용한다.

        # ===== Keepout Filter =====
        Node(
            package="nav2_map_server",
            executable="map_server",
            name="filter_mask_server",
            output="screen",
            emulate_tty=True,
            parameters=[
                params_file,
                {"yaml_filename": keepout_mask_yaml},
            ],
        ),
        Node(
            package="nav2_map_server",
            executable="costmap_filter_info_server",
            name="costmap_filter_info_server",
            output="screen",
            emulate_tty=True,
            parameters=[params_file],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_costmap_filters",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "use_sim_time": True,
                "autostart": True,
                "node_names": lifecycle_nodes_filters,
            }],
        ),
    ])
