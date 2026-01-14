#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPreprocessNode(Node):
    def __init__(self):
        super().__init__("camera_preprocess_node")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "/camera", 10)
        self.sub = self.create_subscription(
            Image, "/camera/raw", self._cb, 10
        )

        self.get_logger().info(
            "Camera preprocess: resize 256x256, BGR->RGB, uint8 -> /camera"
        )

    def _cb(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img = cv2.resize(img, (256, 256))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            out = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")
            out.header = msg.header
            self.pub.publish(out)

        except Exception as e:
            self.get_logger().warn(f"Preprocess error: {e}")

    def destroy_node(self):
        super().destroy_node()

def main():
    rclpy.init()
    node = CameraPreprocessNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
