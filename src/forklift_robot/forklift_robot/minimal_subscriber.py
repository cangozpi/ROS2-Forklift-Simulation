import rclpy
from rclpy.node import Node

from std_msgs.msg import String
# from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('depth_camera_raw_image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info('HEYOOOO')

    def listener_callback(self, msg):
        # self.get_logger().info(f'I heard:  {np.asarray(msg.data).shape}, {msg.height}, {msg.width}, {np.asarray(msg.data).shape[0] /(int(msg.height) * int(msg.width))}')
        # self.get_logger().info(f'I heard:  {np.asarray(msg.data).reshape((3, msg.height, msg.width)).shape}')
        # plt.imshow(np.asarray(msg.data).reshape((msg.height, msg.width, 3)))
        # plt.show()
        cv2.imshow('Forklift camera_raw_image message', np.asarray(msg.data).reshape((msg.height, msg.width, 3)))
        cv2.waitKey(3)
        self.get_logger().info('MAUW')



def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()