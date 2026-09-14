#!/usr/bin/env python3
import math

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class PersonalLaserSelfFilter(Node):
    def __init__(self):
        super().__init__("personal_laser_self_filter")
        input_topic = self.declare_parameter("input_topic", "scan_unfiltered").value
        output_topic = self.declare_parameter("output_topic", "scan").value
        self.self_frame = self.declare_parameter("self_frame", "tf_amr").value
        self.min_x = float(self.declare_parameter("min_x", -0.97).value)
        self.max_x = float(self.declare_parameter("max_x", 0.67).value)
        self.min_y = float(self.declare_parameter("min_y", -0.53).value)
        self.max_y = float(self.declare_parameter("max_y", 0.53).value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(
            LaserScan, output_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            LaserScan, input_topic, self.filter_scan, qos_profile_sensor_data
        )

    @staticmethod
    def rotate(q, x, y, z=0.0):
        # Quaternion-vector rotation without external Python dependencies.
        tx = 2.0 * (q.y * z - q.z * y)
        ty = 2.0 * (q.z * x - q.x * z)
        tz = 2.0 * (q.x * y - q.y * x)
        return (
            x + q.w * tx + q.y * tz - q.z * ty,
            y + q.w * ty + q.z * tx - q.x * tz,
            z + q.w * tz + q.x * ty - q.y * tx,
        )

    def filter_scan(self, scan):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.self_frame,
                scan.header.frame_id,
                Time.from_msg(scan.header.stamp),
                timeout=Duration(seconds=0.05),
            ).transform
        except TransformException as error:
            self.get_logger().warning(
                f"Waiting for {scan.header.frame_id} -> {self.self_frame} TF: {error}",
                throttle_duration_sec=2.0,
            )
            return

        output = LaserScan()
        output.header = scan.header
        output.angle_min = scan.angle_min
        output.angle_max = scan.angle_max
        output.angle_increment = scan.angle_increment
        output.time_increment = scan.time_increment
        output.scan_time = scan.scan_time
        output.range_min = scan.range_min
        output.range_max = scan.range_max
        output.ranges = list(scan.ranges)
        output.intensities = list(scan.intensities)

        angle = scan.angle_min
        for index, distance in enumerate(output.ranges):
            if math.isfinite(distance) and scan.range_min <= distance <= scan.range_max:
                px = distance * math.cos(angle)
                py = distance * math.sin(angle)
                rx, ry, _ = self.rotate(transform.rotation, px, py)
                x = rx + transform.translation.x
                y = ry + transform.translation.y
                if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y:
                    output.ranges[index] = math.inf
            angle += scan.angle_increment

        self.publisher.publish(output)


def main():
    rclpy.init()
    node = PersonalLaserSelfFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
