import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from rclpy.qos import qos_profile_sensor_data

class ForkliftRobotTfSubscriber(Node):

    def __init__(self, cb):
        """
        cb is the callback function passed as subscriber callback.
        """
        super().__init__('forklift_robot_tf_subscriber')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")


        self.subscription = self.create_subscription(
            TFMessage,
            '/tf',
            cb,
            qos_profile = qos_profile_sensor_data
            )
        self.subscription  # prevent unused variable warning



def main(args=None):
    rclpy.init(args=args)

    depth_camera_raw_image_subscriber = ForkliftRobotTfSubscriber()

    rclpy.spin(depth_camera_raw_image_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    depth_camera_raw_image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()