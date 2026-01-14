#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np


class RealSenseCameraNode(Node):
    def __init__(self):
        super().__init__("realsense_camera_node")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/camera/raw", 10)

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipe.start(cfg)

        self.timer = self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info("RealSense RGB started (BGR8 -> /camera/raw)")

    def _tick(self):
        try:
            frames = self.pipe.wait_for_frames(timeout_ms=1000)
            frame = frames.get_color_frame()
            if not frame:
                return

            img = np.asanyarray(frame.get_data())  # BGR uint8
            msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "realsense_rgb"

            self.pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")

    def destroy_node(self):
        self.pipe.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = RealSenseCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
