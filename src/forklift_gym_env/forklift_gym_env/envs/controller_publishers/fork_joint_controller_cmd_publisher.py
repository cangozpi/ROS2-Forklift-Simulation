import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray

class ForkJointContCmdPublisher(Node):

    def __init__(self):
        super().__init__('fork_joint_cont_cmd_publisher')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        self.publisher_ = self.create_publisher(
            Float64MultiArray,
            '/fork_joint_controller/commands',
            10
            )

    
    def publish_cmd(self, msg):
        """
        Input:
            msg (std_msgs.msg.Float64MultiArray): message to publish.
        """
        self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg)



def main(args=None):
    rclpy.init(args=args)

    diff_cont_cmd_vel_unstamped_publisher = ForkJointContCmdPublisher()

    rclpy.spin(diff_cont_cmd_vel_unstamped_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    diff_cont_cmd_vel_unstamped_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()