import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
import numpy as np
import cv2

class CameraRawImageSubscriber(Node):

    def __init__(self):
        super().__init__('camera_raw_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        camera_img = np.asarray(msg.data).reshape((msg.height, msg.width, 3)) # camera image of shape [H, W, 3]
        cv2.imshow('Forklift camera_raw_image message', camera_img)
        cv2.waitKey(1)



def main(args=None):
    rclpy.init(args=args)

    camera_raw_image_subscriber = CameraRawImageSubscriber()

    rclpy.spin(camera_raw_image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    camera_raw_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()