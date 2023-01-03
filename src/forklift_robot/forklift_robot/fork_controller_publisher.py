import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray


class ForkControllerPublisher(Node):

    def __init__(self):
        super().__init__('fork_controller_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/fork_joint_controller/commands', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float64MultiArray()
        # msg.layout = None
        msg.data = [2.0]
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)


def main(args=None):
    rclpy.init(args=args)

    fork_controller_publisher = ForkControllerPublisher()

    rclpy.spin(fork_controller_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    fork_controller_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()