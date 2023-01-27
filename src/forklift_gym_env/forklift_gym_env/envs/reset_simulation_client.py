from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node


class ResetSimulationClientAsync(Node):

    def __init__(self):
        super().__init__('reset_simulation_client_async')
        # Set ros node's clock to use simulation time (gazebo time)
        # use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        # self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # self.cli = self.create_client(Empty, '/reset_simulation')
        # self.cli = self.create_client(Empty, '/reset_world')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('service not available, waiting again...')
        
        self.cur_node = rclpy.create_node('some_node3')        

        self.req = Empty.Request()

    def send_request(self):
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.cur_node.set_parameters([use_sim_time_parameter])
        cli = self.cur_node.create_client(Empty, '/reset_world')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/unpause_physics service not available, waiting again...')

        self.future = cli.call_async(self.req)
        rclpy.spin_until_future_complete(self.cur_node, self.future)

        self.cur_node.destroy_client(cli)

        # Check that Service is available
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('/reset_world service not available, waiting again...')

        # Call Service
        # future = self.cli.call_async(self.req)
        # rclpy.spin_until_future_complete(self, future)

        # return future.result()


def main(args=None):
    rclpy.init(args=args)

    minimal_client = ResetSimulationClientAsync()
    response = minimal_client.send_request()
    minimal_client.get_logger().info(response)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()