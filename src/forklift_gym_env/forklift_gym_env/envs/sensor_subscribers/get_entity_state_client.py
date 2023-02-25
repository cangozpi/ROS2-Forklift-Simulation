import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import GetEntityState


class GetEntityStateClient():

    def __init__(self):
        pass


    def get_entity_state(self, name, reference_frame):
        cur_node = rclpy.create_node('get_entity_state_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(GetEntityState, '/gazebo/get_entity_state')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/gazebo/get_entity_state service not available, waiting again...')
        
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = reference_frame
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(cur_node, future)

        cur_node.destroy_node()
        return future.result()


def main():
    rclpy.init()
    try:
        for i in range(5): 
            node = GetEntityStateClient()
            result = node.get_entity_state("fork_base_link", "world")
            print(result, "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == "__main__":
    main()