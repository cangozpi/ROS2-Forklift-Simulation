import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty


class SimulationController():
    def __init__(self):
        self.req = Empty.Request()


    def send_reset_simulation_request(self):
        cur_node = rclpy.create_node('reset_simulation_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/reset_world')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/reset_world service not available, waiting again...')

        self.future = cli.call_async(self.req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()


    def send_pause_physics_client_request(self):
        cur_node = rclpy.create_node('pause_physics_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/pause_physics')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/pause_physics service not available, waiting again...')
        
        self.future = cli.call_async(self.req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()

    def send_unpause_physics_client_request(self):
        cur_node = rclpy.create_node('unpause_physics_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/unpause_physics')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/unpause_physics service not available, waiting again...')
        
        self.future = cli.call_async(self.req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()