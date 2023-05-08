import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data
from threading import Thread

class ForkliftRobotRosClockSubscriber(Node):

    def __init__(self, env):
        """
        cb is the callback function passed as subscriber callback.
        """
        super().__init__('forklift_robot_tf_subscriber')
        # Set ros node's clock to use simulation time (gazebo time)
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        def cb(msg):
            env.ros_clock = msg.clock

        self.subscription = self.create_subscription(
            Clock,
            '/clock',
            cb,
            qos_profile = qos_profile_sensor_data
            )
        self.subscription  # prevent unused variable warning

        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(self)
        executor_thread = Thread(target=executor.spin, daemon=True)
        executor_thread.start()
        # executor_thread.join()
    




def main(args=None):
    rclpy.init(args=args)

    ros_clock_subscriber = ForkliftRobotRosClockSubscriber()

    rclpy.spin(ros_clock_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ros_clock_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()