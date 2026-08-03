"""Subscribe to an RGB topic and display YOLO instance-segmentation results."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from ultralytics import YOLO


class YoloViewer(Node):
    def __init__(self) -> None:
        super().__init__('yolo_viewer')
        default_model = Path(get_package_share_directory('yolo_pub')) / 'resource' / 'yolo-v8n-best.pt'
        self.declare_parameter('model_path', str(default_model))
        self.declare_parameter('image_topic', '/patrol_camera/color/image_raw')
        self.declare_parameter('depth_topic', '/patrol_camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/patrol_camera/color/camera_info')
        self.declare_parameter('confidence', 0.25)
        self.declare_parameter('imgsz', 640)

        model_path = Path(self.get_parameter('model_path').value)
        if not model_path.is_file():
            raise FileNotFoundError(f'YOLO weights not found: {model_path}')
        self.model = YOLO(str(model_path))
        self.confidence = float(self.get_parameter('confidence').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.window_name = 'YOLO-seg: patrol camera  |  q = quit'
        self.frame_count = 0
        self.rgb_by_stamp, self.depth_by_stamp = {}, {}
        self.fx, self.fy, self.cx, self.cy = 554.3307, 569.7543, 320.0, 240.0
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 960, 720)
        topic = str(self.get_parameter('image_topic').value)
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(Image, topic, self.rgb_callback, qos)
        self.depth_subscription = self.create_subscription(Image, str(self.get_parameter('depth_topic').value), self.depth_callback, qos)
        self.info_subscription = self.create_subscription(CameraInfo, str(self.get_parameter('camera_info_topic').value), self.info_callback, qos)
        self.points_pub = self.create_publisher(PointCloud2, '/trash/object_points', qos)
        self.get_logger().info(f'Loaded {model_path.name}; subscribing to RGB-D from {topic}')

    @staticmethod
    def to_bgr(message: Image) -> np.ndarray:
        if message.encoding not in ('rgb8', 'bgr8', 'rgba8', 'bgra8'):
            raise ValueError(f'Unsupported image encoding: {message.encoding}')
        channels = 4 if message.encoding in ('rgba8', 'bgra8') else 3
        row = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        image = row[:, : message.width * channels].reshape(message.height, message.width, channels)
        if message.encoding == 'rgb8':
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if message.encoding == 'rgba8':
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if message.encoding == 'bgra8':
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    @staticmethod
    def stamp(message: Image) -> tuple[int, int]:
        return message.header.stamp.sec, message.header.stamp.nanosec

    def info_callback(self, message: CameraInfo) -> None:
        if len(message.k) == 9 and message.k[0] > 0 and message.k[4] > 0:
            self.fx, self.fy, self.cx, self.cy = message.k[0], message.k[4], message.k[2], message.k[5]

    def rgb_callback(self, message: Image) -> None:
        self.rgb_by_stamp[self.stamp(message)] = message
        self.process_pair(message)

    def depth_callback(self, message: Image) -> None:
        self.depth_by_stamp[self.stamp(message)] = message
        rgb = self.rgb_by_stamp.get(self.stamp(message))
        if rgb is not None:
            self.process_pair(rgb)

    def process_pair(self, rgb_message: Image) -> None:
        key = self.stamp(rgb_message)
        depth_message = self.depth_by_stamp.pop(key, None)
        if depth_message is None:
            return
        self.rgb_by_stamp.pop(key, None)
        try:
            bgr = self.to_bgr(rgb_message)
            depth = np.frombuffer(depth_message.data, dtype=np.float32).reshape(depth_message.height, depth_message.step // 4)[:, :depth_message.width]
            result = self.model.predict(bgr, imgsz=self.imgsz, conf=self.confidence, retina_masks=True, verbose=False)[0]
            overlay = result.plot()
            points = []
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.cpu().numpy()
                for index, mask in enumerate(masks):
                    mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    valid = mask & np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
                    rows, cols = np.where(valid)
                    if len(rows) > 4000:
                        choose = np.linspace(0, len(rows) - 1, 4000, dtype=int); rows, cols = rows[choose], cols[choose]
                    class_id = int(result.boxes.cls[index].item()); confidence = float(result.boxes.conf[index].item())
                    z = depth[rows, cols]
                    for u, v, value in zip(cols, rows, z):
                        points.append(((u - self.cx) * value / self.fx, (v - self.cy) * value / self.fy, float(value), class_id, confidence))
            fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1), PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1), PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1), PointField(name='class_id', offset=12, datatype=PointField.UINT32, count=1), PointField(name='confidence', offset=16, datatype=PointField.FLOAT32, count=1)]
            cloud = point_cloud2.create_cloud(rgb_message.header, fields, points)
            self.points_pub.publish(cloud)
            cv2.imshow(self.window_name, overlay)
            self.frame_count += 1
            if self.frame_count == 1:
                self.get_logger().info(f'First RGB-D pair received: {rgb_message.width}x{rgb_message.height}; published {len(points)} points')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rclpy.shutdown()
        except Exception as error:
            self.get_logger().error(f'Inference failed: {error}')

    def destroy_node(self) -> bool:
        cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
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
