import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def filter_process(name, input_topic, output_topic):
    return Node(
        package="cleanrobot_amr_navigation",
        executable="cleanrobot_laser_self_filter.py",
        name=name,
        parameters=[{
            "input_topic": input_topic,
            "output_topic": output_topic,
            "self_frame": "tf_amr",
            "min_x": -0.97,
            "max_x": 0.67,
            "min_y": -0.53,
            "max_y": 0.53,
            "use_sim_time": True,
        }],
        output="screen",
    )


def generate_launch_description():
    package_share = get_package_share_directory("cleanrobot_amr_navigation")
    original_launch = os.path.join(
        package_share, "launch", "cleanrobot_nav2_core.launch.py"
    )
    corrected_params = os.path.join(
        package_share, "params", "cleanrobot_amr_nav2.yaml"
    )

    return LaunchDescription([
        # Bundled USD가 AMCL 추정이 아닌 ground-truth map->odom을 사용한다.
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom_ground_truth",
            output="screen",
            arguments=[
                "--x", "1.6175898173585577",
                "--y", "-19.045652432754707",
                "--z", "0.0",
                "--yaw", "1.5707963267948966",
                "--pitch", "0.0",
                "--roll", "0.0",
                "--frame-id", "map",
                "--child-frame-id", "odom",
            ],
        ),
        filter_process(
            "front_scan_self_filter",
            "/cobot3/front/scan_unfiltered",
            "/cobot3/front/scan",
        ),
        filter_process(
            "rear_scan_self_filter",
            "/cobot3/rear/scan_unfiltered",
            "/cobot3/rear/scan",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(original_launch),
            launch_arguments={"params_file": corrected_params}.items(),
        ),
    ])
