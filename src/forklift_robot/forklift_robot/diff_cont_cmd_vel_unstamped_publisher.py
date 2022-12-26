import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Vector3


class DiffContCmdVelUnstampedPublisher(Node):

    def __init__(self):
        super().__init__('diff_cont_cmd_vel_unstamped_publisher')
        self.publisher_ = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        # msg.linear = Vector3()
        msg.linear.x = 1.0 # use this one
        msg.linear.y = 0.0
        msg.linear.z = 0.0

        # msg.angular = Vector3()
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 2.0 # use this one

        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)


def main(args=None):
    rclpy.init(args=args)

    diff_cont_cmd_vel_unstamped_publisher = DiffContCmdVelUnstampedPublisher()

    rclpy.spin(diff_cont_cmd_vel_unstamped_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    diff_cont_cmd_vel_unstamped_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()