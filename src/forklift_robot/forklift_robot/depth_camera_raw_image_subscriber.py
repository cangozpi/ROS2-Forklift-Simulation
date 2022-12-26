import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class DepthCameraRawImageSubscriber(Node):

    def __init__(self):
        super().__init__('depth_camera_raw_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        depth_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_camera_img = cv2.normalize(depth_camera_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize the depth_camera_image to range [0,1]
        cv2.imshow('Forklift depth_camera_raw_image message', depth_camera_img)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    depth_camera_raw_image_subscriber = DepthCameraRawImageSubscriber()

    rclpy.spin(depth_camera_raw_image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depth_camera_raw_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()