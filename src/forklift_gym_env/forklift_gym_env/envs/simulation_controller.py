import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose
from controller_manager.controller_manager_services import LoadController, ConfigureController, SwitchController
from .utils import get_robot_description_raw


class SimulationController():
    def __init__(self):
        self.empty_req = Empty.Request()


    def send_reset_simulation_request(self):
        # Request /reset_simulation
        # Note that this resets the simulation_time (self.ros_clock)
        cur_node = rclpy.create_node('reset_simulation_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/reset_simulation')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/reset_simulation service not available, waiting again...')

        self.future = cli.call_async(self.empty_req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()

        # Request /reset_world
        cur_node = rclpy.create_node('reset_world_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/reset_world')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/reset_world service not available, waiting again...')

        self.future = cli.call_async(self.empty_req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()


    def send_pause_physics_client_request(self):
        cur_node = rclpy.create_node('pause_physics_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/pause_physics')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/pause_physics service not available, waiting again...')
        
        self.future = cli.call_async(self.empty_req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()

    def send_unpause_physics_client_request(self):
        cur_node = rclpy.create_node('unpause_physics_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(Empty, '/unpause_physics')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/unpause_physics service not available, waiting again...')
        
        self.future = cli.call_async(self.empty_req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()
    

    def activate_ros_controllers(self, controller_names: dict): 
        """
        Loads, Configures, and Activates the given ros2 controllers with the names specified in the controller_names (list).
        """
        # Load controllers ------------------
        for controller_name in controller_names:
            cur_node = rclpy.create_node('load_controller_client')        
            use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
            cur_node.set_parameters([use_sim_time_parameter])

            cli = cur_node.create_client(LoadController, '/controller_manager/load_controller')
            while not cli.wait_for_service(timeout_sec=1.0):
                cur_node.get_logger().info('/controller_manager/load_controller service not available, waiting again...')
        
            req = LoadController.Request()
            req.name = controller_name
            self.future = cli.call_async(req)
            rclpy.spin_until_future_complete(cur_node, self.future)

            cur_node.destroy_node()


        # Configure controllers ------------------
        for controller_name in controller_names:
            cur_node = rclpy.create_node('configure_controller_client')        
            use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
            cur_node.set_parameters([use_sim_time_parameter])

            cli = cur_node.create_client(ConfigureController, '/controller_manager/configure_controller')
            while not cli.wait_for_service(timeout_sec=1.0):
                cur_node.get_logger().info('/controller_manager/configure_controller service not available, waiting again...')
        
            req = ConfigureController.Request()
            req.name = controller_name
            self.future = cli.call_async(req)
            rclpy.spin_until_future_complete(cur_node, self.future)

            cur_node.destroy_node()

        # Activate controllers ------------------
        cur_node = rclpy.create_node('switch_controller_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(SwitchController, '/controller_manager/switch_controller')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/controller_manager/switch_controller service not available, waiting again...')
        
        req = SwitchController.Request()
        req.activate_controllers = controller_names
        req.deactivate_controllers = []
        req.strictness = 2
        self.future = cli.call_async(req)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()
    


    def change_agent_location(self, entity_name, agent_location, ros_controller_names: list, agent_pose_position_z):
        """
        Delete agent and respawn it at the self._agent_location coordinates in the simulation. 
        Loads, Configures, and Activates the given ros2 controllers with the names specified in the controller_names (list).
        """
        # Delete agent from simulation ------------------
        cur_node = rclpy.create_node('delete_agent_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(DeleteEntity, '/delete_entity')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/delete_entity service not available, waiting again...')
        
        delete_request = DeleteEntity.Request()
        delete_request.name = entity_name
        
        self.future = cli.call_async(delete_request)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()

        # Spawn agent ------------------
        cur_node = rclpy.create_node('spawn_agent_client')        
        use_sim_time_parameter = rclpy.parameter.Parameter('use_sim_time', rclpy.parameter.Parameter.Type.BOOL, True)
        cur_node.set_parameters([use_sim_time_parameter])

        cli = cur_node.create_client(SpawnEntity, '/spawn_entity')
        while not cli.wait_for_service(timeout_sec=1.0):
            cur_node.get_logger().info('/spawn_entity service not available, waiting again...')
        
        spawn_request = SpawnEntity.Request()
        spawn_request.name = entity_name
        spawn_request.xml = get_robot_description_raw()
        spawn_request.robot_namespace = ""

        agent_pose = Pose()
        agent_pose.position.x = agent_location[0]
        agent_pose.position.y = agent_location[1]
        agent_pose.position.z = agent_pose_position_z

        agent_pose.orientation.x = 0.0
        agent_pose.orientation.y = 0.0
        agent_pose.orientation.z = 0.0
        agent_pose.orientation.w = 1.0

        spawn_request.initial_pose = agent_pose
        spawn_request.reference_frame = ""
        
        self.future = cli.call_async(spawn_request)
        rclpy.spin_until_future_complete(cur_node, self.future)

        cur_node.destroy_node()

        # Configure, Load, and Activate ROS Controllers ------------------
        self.activate_ros_controllers(ros_controller_names)