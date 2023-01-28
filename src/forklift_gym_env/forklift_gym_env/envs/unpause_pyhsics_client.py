from std_srvs.srv import Empty
import rclpy
from rclpy.node import Node


class UnpausePhysicsClient(Node):

    def __init__(self):
        super().__init__('unpause_physics_client_async')
        # Set ros node's clock to use simulation time (gazebo time)
        # use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        # self.set_parameters([use_sim_time_parameter])
        # print("KK", self.get_parameter('use_sim_time').get_parameter_value().bool_value, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

        # self.cli = self.create_client(Empty, '/unpause_physics')
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('/unpause_physics service not available, waiting again...')

        self.cur_node = rclpy.create_node('some_node')        

        self.req = Empty.Request()


    def send_request(self):
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        self.cur_node.set_parameters([use_sim_time_parameter])
        cli = self.cur_node.create_client(Empty, '/unpause_physics')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('/unpause_physics service not available, waiting again...')
        
        print("KOOOO")
        self.future = cli.call_async(self.req)
        rclpy.spin_until_future_complete(self.cur_node, self.future)
        print("MUQ")
        while not self.cur_node.destroy_client(cli):
            continue

        # Check that Service is available
        # while not self.cli.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().info('/unpause_physics service not available, waiting again...')

        # Call Service
        # print("KOOOO")
        # self.future = self.cli.call_async(self.req)
        # rclpy.spin_until_future_complete(self, self.future)
        # print("MUQ")
        # return self.future.result()
        # while not future.done():
        #     # future = self.cli.call_async(Empty.Request())
        #     rclpy.spin_until_future_complete(self, future, timeout_sec=3)
        #     print(future.done(), future.result(), " NOOOOOOOOOOOOO")
        # print(future.done(), future.result(), " heyooJJ:JJ")

        # while rclpy.ok():
        #     print("a")
        #     rclpy.spin_once(self)
        #     print("b")
        #     if future.done():
        #         print(future.done(), future.result(), " NOOOOOOOOOOOOO")
        #         break
        #     print(future.cancelled(), future.exception(), self.cli.service_is_ready(), rclpy.ok(), future.done(), future.result(), " FUUUUUUUUUUUKKKKKKKKK")
        # return self.future.result()
        # print("KOOOO")
        # response = self.cli.call(Empty.Request())
        # print(response," heyooJJ:JJ")
        # return response


def main(args=None):
    rclpy.init(args=args)

    minimal_client = UnpausePhysicsClient()
    response = minimal_client.send_request()
    minimal_client.get_logger().info(response)

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()