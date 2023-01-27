import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class DepthCameraRawImageSubscriber(Node):

    def __init__(self, cb):
        """
        cb is the callback function passed as subscriber callback.
        """
        super().__init__('depth_camera_raw_image_subscriber')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])


        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            cb,
            10)
        self.subscription  # prevent unused variable warning

    # def listener_callback(self, msg):
    #     """
    #     Not used by default. Change cb with listener_callback to debug.
    #     """
    #     depth_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    #     depth_camera_img = cv2.normalize(depth_camera_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize the depth_camera_image to range [0,1]
    #     cv2.imshow('Forklift depth_camera_raw_image message', depth_camera_img)
    #     cv2.waitKey(1)



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