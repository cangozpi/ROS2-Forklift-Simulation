import gym
from gym import spaces
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from forklift_gym_env.envs.utils import generate_and_launch_ros_description_as_new_process, read_yaml_config, \
    ObservationType, RewardType, ActionType
from forklift_gym_env.envs.simulation_controller import SimulationController

# Sensor Subscribers
from forklift_gym_env.envs.sensor_subscribers.forklift_robot_ros_clock_subscriber import ForkliftRobotRosClockSubscriber
from forklift_gym_env.envs.sensor_subscribers.forklift_robot_tf_subscriber import ForkliftRobotTfSubscriber
# from forklift_gym_env.envs.sensor_subscribers.forklift_robot_frame_listener import FrameListener
from forklift_gym_env.envs.sensor_subscribers.get_entity_state_client import GetEntityStateClient
from forklift_gym_env.envs.sensor_subscribers.depth_camera_raw_image_subscriber import DepthCameraRawImageSubscriber
from forklift_gym_env.envs.sensor_subscribers.collision_detection_subscriber import CollisionDetectionSubscriber

# Controller Publishers
from forklift_gym_env.envs.controller_publishers.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher
from forklift_gym_env.envs.controller_publishers.fork_joint_controller_cmd_publisher import ForkJointContCmdPublisher

# Observation, Action, Reward function Factories
from forklift_gym_env.envs.forklift_env_observations_utils import observation_space_factory, init_check_goal_achieved, flatten_and_concatenate_observation
from forklift_gym_env.envs.forklift_env_Actions_utils import action_space_factory
from forklift_gym_env.envs.forklift_env_Rewards_utils import calculate_reward_factory

from forklift_gym_env.logging.logger import Logger



class ForkliftEnv(gym.GoalEnv): # Note that gym.GoalEnv is a subclass of gym.Env
    metadata = {
        "render_modes": ["no_render", "show_depth_camera_img_raw", "draw_coordinates"], 
        # "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None, config_path = None, use_GoalEnv = False):
        self.use_GoalEnv = use_GoalEnv # if True, ForkliftEnv behaves like a gym.GoalEnv. If False, behaves like a regular gym.Env
        # Read in parameters from config.yaml
        self.config = read_yaml_config(config_path)
        self.logger = Logger("ForkliftEnv_logger", ['FileHandler'], 'log_ForkliftEnv.log')

        # Set observation_space, _get_obs method, and action_space
        self.obs_types = [ObservationType(obs_type) for obs_type in self.config["observation_types"]]
        self.observation_space, self._get_obs = observation_space_factory(self, obs_types = self.obs_types)
        self.act_types = [ActionType(act_type) for act_type in self.config["action_types"]]
        self.action_space = action_space_factory(self, act_types = self.act_types)
        self.reward_types = [RewardType(reward_type) for reward_type in self.config["reward_types"]]
        self.calc_reward = calculate_reward_factory(self, reward_types = self.reward_types)
        self.check_goal_achieved = init_check_goal_achieved(self)

        # Set render_mode
        for x in self.config["render_mode"]:
            assert x in ForkliftEnv.metadata['render_modes'] 
        self.render_mode = self.config['render_mode']

        # -------------------- 
        # Start gazebo simulation, robot_state_publisher
        self.gazebo_launch_subp = generate_and_launch_ros_description_as_new_process(self.config) # Reference to subprocess that runs these things

        # -------------------- 
        # Listen to sensors: ============================== 
        rclpy.init()

        # -------------------- /clock
        self.ros_clock = None # self.ros_clock will be used to check that observations are obtained after the actions are taken
        self.get_ros_clock_client = ForkliftRobotRosClockSubscriber(self) # automatically updates self.ros_clock with the latest published msg
        # --------------------
        # -------------------- /gazebo/get_entity_state
        self.get_entity_state_client = GetEntityStateClient()
        # --------------------
        # -------------------- /camera/depth/image_raw
        self.depth_camera_raw_image_subscriber = DepthCameraRawImageSubscriber(self, normalize_img=False)
        # -------------------- 
        # -------------------- /collision_detections/*
        self.collision_detection_states = {}
        self.collision_detection_subscribers = []
        for link_name in self.config['collision_detection_link_names']:
            self.collision_detection_subscribers.append(CollisionDetectionSubscriber.initialize_collision_detection_subscriber(self, link_name))
        # ====================================================

        # Create publisher for controlling forklift robot's joints: ============================== 
        # --------------------  /cmd_vel
        self.diff_cont_cmd_vel_unstamped_publisher = DiffContCmdVelUnstampedPublisher()
        # -------------------- 
        # --------------------  /fork_joint_controller/commands
        self.fork_joint_cont_cmd_publisher = ForkJointContCmdPublisher()
        # -------------------- 
        # ====================================================

        # Create Clients for Simulation Services: ============================== 
        # --------------------  /reset_simulation, /pause_physics, /unpause_physics, /controller_manager/*
        self.simulation_controller_node = SimulationController()
        # -------------------- 
        # ====================================================
       
        # Parameters: -------------------- 
        self.cur_iteration = 0
        self.max_episode_length = self.config['max_episode_length']
        self._agent_entity_name = self.config['entity_name'] # entity name of the forklift robot in the simulation.
        self._pallet_entity_name = "pallet"
        self._ros_controller_names = self.config['ros_controller_names'] # ROS2 controllers that are activated.
        self._tolerance_x = self.config['tolerance_x'] # chassis_bottom_transform's x location's tolerable distance to target_location's x coordinate for agent to achieve the goal state
        self._tolerance_y = self.config['tolerance_y'] # chassis_bottom_transform's y location's tolerable distance to target_location's y coordinate for agent to achieve the goal state
        # -------------------------------------

        self._target_transform = None # agent's goal state



    def reset(self):
        seed = self.config["seed"]
        # super().reset()

        self.cur_iteration = 0

        # Choose the agent's location uniformly at random
        # self._agent_location = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]
        # self._agent_location *= np.array([0, 1]) # make it only change in y axis
        self._agent_location = np.array([0.0, 0.0]) # fix forklift start state to origin

        # Sample the target's location randomly until it does not coincide with the agent's location
        # self._target_transform = np.array([0.0, 0.0]) 
        # self._target_transform[0] = np.random.random(size=(1,)) * 10 #+ (-10) # x in the range [-10, 10]
        # self._target_transform[1] = np.random.random(size=(1,)) * 4 - 2 # x in the range [-2, 2]
        # while np.array_equal(self._target_transform, self._agent_location):
            # self._target_transform = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]
            # self._target_transform *= np.array([0, 1]) # make it only change in y axis
            # self._target_transform[0] = np.random.random(size=(1,)) * 4 - 2 # x in the range [-2, 2]
        #     self._target_transform[0] = np.random.random(size=(1,)) * 10 #+ (-10) # y in the range [-10, 10]
        # if self._target_transform[0] > -4 and self._target_transform[0] < 4:
        #     self._target_transform[0] = 5 * np.sign(np.random.random(size=(1,)))

        self._target_transform = np.array([6.0, 2.0]) # fixed at location [6, 2]
        # In the range [6, (-4,4)]
        # self._target_transform = np.array([6.0, 0.0]) # in the range [6.0, 0.0]
        # self._target_transform[1] = np.random.random() * 8 - 4 # in the range [-4, 4]
 

        # Unpause sim so that simulation can be reset
        self.simulation_controller_node.send_unpause_physics_client_request()

        self.action = np.zeros(self.action_space.shape, dtype=np.float32) # diff_cont_msg

        # Send 'no action' action to diff_cmd_cont of forklift robot
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = 0.0 # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = 0.0 # use this one
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)
        rclpy.spin_once(self.diff_cont_cmd_vel_unstamped_publisher)


        # send 'no action' action to fork_joint_cont of forklift robot
        fork_joint_cont_msg = Float64MultiArray()
        fork_joint_cont_msg.data = [0.0]
        # Take fork_joint_cont action
        self.fork_joint_cont_cmd_publisher.publish_cmd(fork_joint_cont_msg)
        rclpy.spin_once(self.fork_joint_cont_cmd_publisher)

        # Reset the simulation & world (gazebo)
        self.simulation_controller_node.send_reset_simulation_request()

        # Change agent location in the simulation, activate ros_controllers
        self.simulation_controller_node.change_entity_location(self._agent_entity_name, self._agent_location, self._ros_controller_names, self.config['agent_pose_position_z'])
        
        # Change pallet model's location in the simulation
        self.simulation_controller_node.change_entity_location(self._pallet_entity_name, self._target_transform, [], \
            0.0, self.config, spawn_pallet=True)

        self.ros_clock_ckpt = self.ros_clock

        # Get observation
        observation = self._get_obs()

        # Convert nested Dict obs to the required format (gym.Env | gym.GoalEnv)
        processed_observations = flatten_and_concatenate_observation(self, observation)

        # Pause simuation so that obseration does not change until another action is taken
        self.simulation_controller_node.send_pause_physics_client_request()

        # Render
        self.draw_coordinates_dict = {
            'forklift_coordinates': [],
            'target_coordinate': None
        }
        self.draw_coordinates_dict['forklift_coordinates'].append(self._agent_location) # Mark forklift location
        self.draw_coordinates_dict['target_coordinate'] = self._target_transform # Mark target location
        # Initialize and render plot
        plt.close()
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.autoscale()
        self.render(observation)


        # return observation_flat, self._get_info(None, diff_cont_msg, observation)
        return processed_observations

    
    def step(self, action):
        self.cur_iteration += 1

        self.logger.log_tabular(key="step() > ros_clock before unpause simulation", value=self.ros_clock)

        # Unpause simulation so that action can be taken
        self.simulation_controller_node.send_unpause_physics_client_request()
        self.logger.log_tabular(key="step() > ros_clock after unpause simulation", value=self.ros_clock)

        # Set diff_cont action
        # diff_cont_action = action['diff_cont_action'] 
        # action[0] = (float(action[0]) + 1) / 2 # modify action from [-1, ,1] to [0, 1] so that it only moves forward
        self.action = action
        diff_cont_action = action
        # convert diff_cont_action to Twist message
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = float(diff_cont_action[0]) # use this one
        # diff_cont_msg.linear.x = (float(diff_cont_action[0]) + 1) / 2 # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = float(diff_cont_action[1]) # use this one
        # diff_cont_msg.angular.z = float(diff_cont_action[0]) # use this one
        # Take diff_cont action
        self.logger.log_tabular(key="step() > ros_clock before diff_cont_cmd_vel_unstamped_publisher.publish_cmd()", value=self.ros_clock)
        if not self.config['manual_control']:
            self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)
            rclpy.spin_once(self.diff_cont_cmd_vel_unstamped_publisher)
        self.logger.log_tabular(key="step() > ros_clock after rclpy.spin_once(diff_cont_cmd_vel_unstamped_publisher)", value=self.ros_clock)

        # set fork_joint_cont action
        # fork_joint_cont_action = action['fork_joint_cont_action']
        # convert fork_joint_cont_action to Float64MultiArray message
        fork_joint_cont_msg = Float64MultiArray()
        # fork_joint_cont_msg.data = fork_joint_cont_action.tolist()
        fork_joint_cont_msg.data = [0.0]
        # Take fork_joint_cont action
        if not self.config['manual_control']:
            self.fork_joint_cont_cmd_publisher.publish_cmd(fork_joint_cont_msg)
            rclpy.spin_once(self.fork_joint_cont_cmd_publisher)

        # Get observation after taking the action
        self.ros_clock_ckpt = self.ros_clock # will be used to make sure observation is coming from after the action was taken

        self.logger.log_tabular(key="step() > ros_clock before self._get_obs()", value=self.ros_clock)
        observation = self._get_obs()
        self.logger.log_tabular(key="step() > ros_clock after self._get_obs()", value=self.ros_clock)
        self.logger.log_tabular(key=f"step() > time btw action taken and observation captured", value=f'sec: {(self.ros_clock.sec - self.ros_clock_ckpt.sec)} nanosec:{(self.ros_clock.nanosec - self.ros_clock_ckpt.nanosec)}')

        # Pause simulation so that obseration does not change until another action is taken
        self.simulation_controller_node.send_pause_physics_client_request()
        self.logger.log_tabular(key="step() > ros_clock after pause simulation", value=self.ros_clock)

        # Convert nested Dict obs to the required format (gym.Env | gym.GoalEnv)
        processed_observations = flatten_and_concatenate_observation(self, observation)
        self.logger.log_tabular(key="step() > ros_clock after flatten_and_concatenate_observation()", value=self.ros_clock)

        # Get info
        info = self._get_info(observation) 

        # Calculate reward
        if self.use_GoalEnv: # Support gym.GoalEnv(HER) and return reward
            reward = self.compute_reward(processed_observations['achieved_goal'], processed_observations['desired_goal'], info)
        else: # Support gym.Env and return reward
            reward = self.calc_reward(observation)
        print(f'reward: {reward}')
        print('action:', action)


        # Check if episode should terminate 
        if self.use_GoalEnv:
            achieved_goal = processed_observations['achieved_goal']
        else:
            achieved_goal = np.array([observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].x, \
                observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].y])
        done = bool(self.cur_iteration >= (self.max_episode_length)) or (self.check_goal_achieved(achieved_goal, full_obs=False))

        # Render
        self.draw_coordinates_dict['forklift_coordinates'].append(achieved_goal) # Mark forklift location
        self.render(observation)

        self.logger.log_tabular(key="step() > ros_clock before returning from step()", value=self.ros_clock)
        self.logger.log_tabular(key="======================== end of step() ====================", value='')
        return processed_observations, reward, done, info # (observation, reward, done, truncated, info)



    def compute_reward(self, achieved_goal, desired_goal, info): # Required for supporting gym.GoalEnv
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in the 'info' parameter and compute it accordingly.

        *Note that compute reward internally calls the self.calc_reward() function. Only functionality it
        adds on top of it is accumulating the rewards returned by calling it for the batch input cases.
        In other words it makes self.calc_reward() process batch inputs.
        

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information.

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = self.step()
                assert reward == self.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        # Handle batch and single processing cases
        dims = len(achieved_goal.shape)

        if dims == 1: # Process single data coming from step() function
            reward = self.calc_reward(achieved_goal, desired_goal, info) 
            return reward

        elif dims > 1: # Process batch data coming from HER implementation.
            rewards = []
            for cur_a_goal, cur_d_goal, cur_info in zip(achieved_goal, desired_goal, info):
                cur_reward = self.calc_reward(cur_a_goal, cur_d_goal, cur_info) 
                rewards.append(cur_reward) 
            return np.array(rewards)

        else: # Invalid inputs
            raise Exception("compute_reward got inputs with invalid shape")
    


    def _get_info(self, observation):
        info = {
            "iteration": self.cur_iteration,
            "max_episode_length": self.max_episode_length,
            "agent_location": [observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x,\
                    observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y], # [translation_x, translation_y]
            "target_location": self._target_transform, # can be changed with: observation['pallet_position']['pallet_model']['pose']['position']
            "verbose": self.config["verbose"],
            "observation": observation,
        }
        return info
    


    def render(self, observation): 
        if self.render_mode is not None:
            self._render_frame(observation)
    
    def _render_frame(self, observation):
        if "draw_coordinates" in self.render_mode:
            self.ax.clear()
            # Draw Forklift coordinates
            fork_coordinates = np.array(self.draw_coordinates_dict['forklift_coordinates'])
            x = fork_coordinates[:,0]
            y = fork_coordinates[:,1]
            self.ax.plot(x, y, 'b-', label="forklift coordinates")
            # Draw Target coordinates
            self.ax.plot(self.draw_coordinates_dict['target_coordinate'][0], self.draw_coordinates_dict['target_coordinate'][1], 'r*', label='pallet coordinates')
            # Label the plot
            self.ax.set_title('Rendering Coordinates')
            self.ax.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        if "show_depth_camera_img_raw" in self.render_mode:
            depth_camera_img = self.depth_camera_img_observation['image']
            cv2.imshow('Forklift depth_camera_raw_image message', depth_camera_img)
            cv2.waitKey(1)



    def close(self): 
        # delete ros nodes
        self.depth_camera_raw_image_subscriber.destroy_node()
        self.diff_cont_cmd_vel_unstamped_publisher.destroy_node()
        self.fork_joint_cont_cmd_publisher.destroy_node()
        for subscriber in self.collision_detection_subscribers:
            subscriber.destroy_node()
        rclpy.shutdown()

        # Stop gazebo launch process
        self.gazebo_launch_subp.terminate() 
        self.gazebo_launch_subp.join() 
        self.gazebo_launch_subp.close()     

        # Release used CV2 resources
        cv2.destroyAllWindows()