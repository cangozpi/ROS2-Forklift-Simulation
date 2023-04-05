import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
import cv2
from cv_bridge import CvBridge

class DepthCameraRawImageSubscriber(Node):

    def __init__(self, env, normalize_img = False):
        """
        Inputs:
            env (class ForkliftEnv): Reference to ForkliftEnv in which the agent will receive rewards which are implemented here.
            normalize_img (bool) = if True, img is normalized into range [0, 1]. Defaults to False.
        """
        super().__init__('depth_camera_raw_image_subscriber')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])

        self.bridge = CvBridge()
        # Define callback function for subscribing
        env.depth_camera_img_observation = None # callback will record coming information to this variable of the env
        def depth_camera_raw_image_subscriber_cb(msg):
            depth_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if normalize_img:
                depth_camera_img = cv2.normalize(depth_camera_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize the depth_camera_image to range [0,1]

            env.depth_camera_img_observation = { # record to env's variable
                'header': msg.header,
                'image': depth_camera_img
            }

        self.subscription = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            # '/camera/image_raw',
            depth_camera_raw_image_subscriber_cb, # callback function
            qos_profile = qos_profile_sensor_data
            )
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