import gym
from gym import spaces
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
import rclpy
from geometry_msgs.msg import Twist

from forklift_gym_env.envs.utils import generate_and_launch_ros_description_as_new_process, read_yaml_config, \
    ObservationType, RewardType, ActionType
from forklift_gym_env.envs.simulation_controller import SimulationController

# Sensor Subscribers
from forklift_gym_env.envs.sensor_subscribers.forklift_robot_tf_subscriber import ForkliftRobotTfSubscriber
from forklift_gym_env.envs.sensor_subscribers.depth_camera_raw_image_subscriber import DepthCameraRawImageSubscriber

# Controller Publishers
from forklift_gym_env.envs.controller_publishers.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher



class ForkliftEnv(gym.Env):
    metadata = {
        "render_modes": ["no_render", "show_depth_camera_img_raw"], 
        # "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None):
        # Read in parameters from config.yaml
        config_path = 'build/forklift_gym_env/forklift_gym_env/config/config.yaml'
        self.config = read_yaml_config(config_path)

        # Set observation_space, _get_obs method, and action_space
        self.observation_space, self._get_obs = self.observation_space_factory(obs_type = ObservationType(self.config["observation_type"]))
        self.action_space = self.action_space_factory(act_type = ActionType(self.config["action_type"]))
        self.calc_reward = self.calculate_reward_factory(reward_type = RewardType(self.config["reward_type"]))

        # Set render_mode
        for x in self.config["render_mode"]:
            assert x in ForkliftEnv.metadata['render_modes'] 
        self.render_mode = self.config['render_mode']

        # self.ros_clock will be used to check that observations are obtained after the actions are taken
        self.ros_clock = None

        # -------------------- 
        # Start gazebo simulation, robot_state_publisher
        self.gazebo_launch_subp = generate_and_launch_ros_description_as_new_process(self.config) # Reference to subprocess that runs these things

        # -------------------- 
        # Subscribe to sensors: ============================== 
        rclpy.init()

        # -------------------- /tf
        self.forklift_robot_tf_subscriber = self.initialize_forklift_robot_tf_subscriber()
        # -------------------- 
        # -------------------- /camera/depth/image/raw
        self.depth_camera_raw_image_subscriber = self.initialize_depth_camera_raw_image_subscriber(normalize_img = True)
        # -------------------- 
        # ====================================================

        # Create publisher for controlling forklift robot's joints: ============================== 
        # --------------------  /diff_cont/cmd_vel_unstamped
        self.diff_cont_cmd_vel_unstamped_publisher = DiffContCmdVelUnstampedPublisher()
        # -------------------- 
        # ====================================================

        # Create Clients for Services: ============================== 
        # --------------------  /reset_simulation, /pause_physics, /unpause_physics, /controller_manager/*
        self.simulation_controller_node = SimulationController()
        # -------------------- 
        # ====================================================
       
        # Parameters: -------------------- 
        self.cur_iteration = 0
        self.max_episode_length = self.config['max_episode_length']
        self._entity_name = self.config['entity_name']
        self._ros_controller_names = self.config['ros_controller_names']
        self._tolerance_x = self.config['tolerance_x'] # chassis_bottom_transform's x location's tolerable distance to target_location's x coordinate for agent to achieve the goal state
        self._tolerance_y = self.config['tolerance_y']  # chassis_bottom_transform's y location's tolerable distance to target_location's y coordinate for agent to achieve the goal state
        # -------------------------------------

        self._target_transform = None 
        


    def _get_obs_tf_only(self):
        # Obtain a tf observation which belongs to a time after the action was taken
        current_forklift_robot_tf_obs = {}
        flag = True
        while current_forklift_robot_tf_obs == {} or flag:
            # Obtain forklift_robot_tf observation -----
            rclpy.spin_once(self.forklift_robot_tf_subscriber)
            current_forklift_robot_tf_obs = self.forklift_robot_tf_state

            flag = False
            for k, v in current_forklift_robot_tf_obs.items():
                if v['time'] < (self.ros_clock.nanoseconds + self.config['step_duration']): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.
                    flag = True
                    break
            if "chassis_bottom_link" not in current_forklift_robot_tf_obs:
                flag = True
        # --------------------------------------------

        # reset observations for next iteration
        self.forklift_robot_tf_state = {}

        return {
            'forklift_robot_tf_observation': {
                'chassis_bottom_link': current_forklift_robot_tf_obs["chassis_bottom_link"]
                },
        }


    def _get_obs_depth_camera_raw_image_and_tf(self):
        # Depth_camera_raw_image observation -----
        # Check that the observation is from after the action was taken
        current_depth_camera_raw_image_obs = None
        while (current_depth_camera_raw_image_obs is None) or \
            (int(str(current_depth_camera_raw_image_obs["header"].stamp.sec) \
                + (str(current_depth_camera_raw_image_obs["header"].stamp.nanosec))) \
                    < (self.ros_clock.nanoseconds + self.config['step_duration'])): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.  
            # Obtain new depth_camera_raw_image_observation: -----
            rclpy.spin_once(self.depth_camera_raw_image_subscriber)
            current_depth_camera_raw_image_obs = self.depth_camera_img_observation

        depth_camera_raw_image_observation = current_depth_camera_raw_image_obs["image"] # get image
        # --------------------------------------------


        # Forklift_robot_tf observation -----
        # Obtain a tf observation which belongs to a time after the action was taken
        current_forklift_robot_tf_obs = {}
        flag = True
        while current_forklift_robot_tf_obs == {} or flag:
            # Obtain forklift_robot_tf observation -----
            rclpy.spin_once(self.forklift_robot_tf_subscriber)
            current_forklift_robot_tf_obs = self.forklift_robot_tf_state

            flag = False
            for k, v in current_forklift_robot_tf_obs.items():
                if v['time'] < (self.ros_clock.nanoseconds + self.config['step_duration']): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.
                    flag = True
                    break
            if "chassis_bottom_link" not in current_forklift_robot_tf_obs:
                flag = True
        # --------------------------------------------

        # reset observations for next iteration
        self.forklift_robot_tf_state = {}
        self.depth_camera_img_observation = None 

        return {
            'depth_camera_raw_image_observation': depth_camera_raw_image_observation,
            'forklift_robot_tf_observation': {
                'chassis_bottom_link': current_forklift_robot_tf_obs["chassis_bottom_link"]
                },
        }


    def _get_info(self, reward, diff_cont_msg):
        info = {
            "iteration": self.cur_iteration,
            "max_episode_length": self.max_episode_length,
            "reward": reward,
            "agent_location": self._agent_location,
            "target_location": self._target_transform,
            "verbose": self.config["verbose"]
        }
        return info
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cur_iteration = 0

        # Choose the agent's location uniformly at random
        self._agent_location = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]

        # Sample the target's location randomly until it does not coincide with the agent's location
        self._target_transform = self._agent_location
        while np.array_equal(self._target_transform, self._agent_location):
            self._target_transform = np.random.random(size=2) * 40 - 20 # in the range [-20, 20]


        # Unpause sim so that simulation can be reset
        self.simulation_controller_node.send_unpause_physics_client_request()

        # Send 'no action' action to forklift robot
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = 0.0 # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = 0.0 # use this one
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        # Reset the simulation & world (gazebo)
        self.simulation_controller_node.send_reset_simulation_request()

        # Change agent location in the simulation, activate ros_controllers
        self.simulation_controller_node.change_agent_location(self._entity_name, self._agent_location, self._ros_controller_names, self.config['agent_pose_position_z'])

        self.ros_clock = self.forklift_robot_tf_subscriber.get_clock().now()

        # Get observation
        observation = self._get_obs()

        # Render
        self.render(observation)

        return observation, self._get_info(None, diff_cont_msg)

    
    def step(self, action):
        self.cur_iteration += 1

        # Unpause simulation so that action can be taken
        self.simulation_controller_node.send_unpause_physics_client_request()

        # Set action
        diff_cont_action = action['diff_cont_action'] 
        # convert diff_cont_action to Twist message
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = float(diff_cont_action[0]) # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = float(diff_cont_action[1]) # use this one
        # Take action
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        # Get observation after taking the action
        self.ros_clock = self.forklift_robot_tf_subscriber.get_clock().now() # will be used to make sure observation is coming from after the action was taken

        observation = self._get_obs()

        # Pause simuation so that obseration does not change until another action is taken
        self.simulation_controller_node.send_pause_physics_client_request()

        # Calculate reward
        reward = self.calc_reward(observation['forklift_robot_tf_observation'], self._target_transform) 

        # Check if episode should terminate 
        done = bool(self.cur_iteration >= (self.max_episode_length)) or (self.check_goal_achieved(observation))

        # Get info
        info = self._get_info(reward, diff_cont_msg) 

        # Render
        self.render(observation)

        return observation, reward, done, False, info # (observation, reward, done, truncated, info)


    def render(self, observation): 
        if self.render_mode is not None:
            self._render_frame(observation)
    
    def _render_frame(self, observation):
        if "show_depth_camera_img_raw" in self.render_mode:
            depth_camera_img = observation['depth_camera_raw_image_observation']
            cv2.imshow('Forklift depth_camera_raw_image message', depth_camera_img)
            cv2.waitKey(1)


    def close(self): 
        # delete ros nodes
        self.depth_camera_raw_image_subscriber.destroy_node()
        self.diff_cont_cmd_vel_unstamped_publisher.destroy_node()
        self.forklift_robot_tf_subscriber.destroy_node()
        rclpy.shutdown()

        # Stop gazebo launch process
        self.gazebo_launch_subp.terminate() 
        self.gazebo_launch_subp.join() 
        self.gazebo_launch_subp.close()     

        # Release used CV2 resources
        cv2.destroyAllWindows()


    # ============================================= Helper functions
    def initialize_depth_camera_raw_image_subscriber(self, normalize_img = False):
        """
        Input:
            normalize_img (bool) = if True, img is normalized into range [0, 1]. Defaults to False.
        """
        self.depth_camera_img_observation = None
        def depth_camera_raw_image_subscriber_cb(msg):
            try:
                self.bridge
            except:
                self.bridge = CvBridge()

            depth_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if normalize_img:
                depth_camera_img = cv2.normalize(depth_camera_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize the depth_camera_image to range [0,1]

            self.depth_camera_img_observation = {
                'header': msg.header,
                'image': depth_camera_img
            }

        return DepthCameraRawImageSubscriber(depth_camera_raw_image_subscriber_cb)
    


    def initialize_forklift_robot_tf_subscriber(self):
       self.forklift_robot_tf_state = {}
       def forklift_robot_tf_cb(msg):
        for cur_msg in msg.transforms:
            cur_msg_time = int(str(cur_msg.header.stamp.sec) + str(cur_msg.header.stamp.nanosec))
            cur_msg_child_frame_id = cur_msg.child_frame_id
            cur_msg_transform = cur_msg.transform
            if cur_msg_child_frame_id not in self.forklift_robot_tf_state:
                self.forklift_robot_tf_state[cur_msg_child_frame_id] = {
                    'time': cur_msg_time,
                    'transform': np.asarray([cur_msg_transform.translation.x, cur_msg_transform.translation.y, cur_msg_transform.translation.z, \
                        cur_msg_transform.rotation.x, cur_msg_transform.rotation.y, cur_msg_transform.rotation.z, cur_msg_transform.rotation.w])
                }
            else:
                if self.forklift_robot_tf_state[cur_msg_child_frame_id]['time'] < cur_msg_time: # newer information came (update)
                    self.forklift_robot_tf_state[cur_msg_child_frame_id] = {
                        'time': cur_msg_time,
                        'transform': np.asarray([cur_msg_transform.translation.x, cur_msg_transform.translation.y, cur_msg_transform.translation.z, \
                            cur_msg_transform.rotation.x, cur_msg_transform.rotation.y, cur_msg_transform.rotation.z, cur_msg_transform.rotation.w])
                    }

       return ForkliftRobotTfSubscriber(forklift_robot_tf_cb)
    

    def observation_space_factory(self, obs_type: ObservationType):
        """
        Returns observation space and corresponding _get_obs method that corresponds to the given obs_type
        Inputs:
            obs_type (ObservationType): specificies which observation is being used.
        """
        assert obs_type in ObservationType

        # Set observation_space according to obs_type 
        if obs_type == ObservationType.TF_ONLY:
            return spaces.Dict({ 
                "forklift_robot_tf_observation": spaces.Dict({
                    "chassis_bottom_link": spaces.Dict({
                        "time": spaces.Box(low = 0.0, high = float("inf"), dtype = int),
                        "transform": spaces.Box(low = -float("inf") * np.ones((7,)), high = float("inf") * np.ones((7,)), dtype = float) # TODO: set these values to min and max from ros diff_controller
                        })
                    }),
            }), self._get_obs_tf_only

        elif obs_type == ObservationType.TF_AND_DEPTH_CAMERA_RAW:
            return spaces.Dict({ 
                "depth_camera_raw_image_observation": spaces.Box(low = -float("inf") * \
                    np.ones(tuple(self.config['depth_camera_raw_image_dimensions'])), \
                        high = float("inf") * np.ones(tuple(self.config['depth_camera_raw_image_dimensions']), dtype = np.float32)),

                "forklift_robot_tf_observation": spaces.Dict({
                    "chassis_bottom_link": spaces.Dict({
                        "time": spaces.Box(low = 0.0, high = float("inf"), dtype = int),
                        "transform": spaces.Box(low = -float("inf") * np.ones((7,)), high = float("inf") * np.ones((7,)), dtype = float)
                        })
                    }),
            }), self._get_obs_depth_camera_raw_image_and_tf
    
    
    def calculate_reward_factory(self, reward_type: RewardType):
        """
        Returns a function that calculates reward which corresponds to the given reward_type
        Inputs:
            reward_type (RewardType): specifies which reward function is used.
        """
        assert reward_type in RewardType

        # return corresponding reward calculation funciton 
        if reward_type == RewardType.L2_DIST:
            def calc_reward_L2_dist(forklift_robot_transform, target_transform):
                """
                Returns negative of L2 distance between the (translation_x, translation_y) coordinates 
                of forklift_robot_transform and target_transform.
                Inputs:
                    forklift_robot_transform (dict): has a key "chassis_bottom_link" which holds (translation_x, translation_y) 
                        in its first two indices. This cooresponds to the current location of the forklift robot
                    target_transform (list): has (tranlation_x, translation_y) coordinates in its first two indices of the 
                        target location
                Returns:
                    reward: -L2 distance

                """
                # Return negative L2 distance btw chassis_bottom_link and the target location as reward
                # Note that we are only using translation here, NOT using rotation information
                robot_transform_translation = forklift_robot_transform['chassis_bottom_link']['transform'][:2] # [translation_x, translation_y]
                l2_dist = np.linalg.norm(robot_transform_translation - target_transform)
                return - l2_dist
            return calc_reward_L2_dist
    

    def action_space_factory(self, act_type: ActionType):
        """
        Returns observation space that corresponds to obs_type
        Inputs:
            act_type (ActionType): specifies which actions can be taken by the agent.
        """
        assert act_type in ActionType

        # Set action space according to act_type 
        if act_type == ActionType.DIFF_CONT:
            return spaces.Dict({
                "diff_cont_action": spaces.Box(low = -10 * np.ones((2)), high = 10 * np.ones((2)), dtype=np.float32) #TODO: set this to limits from config file
            })


    def check_goal_achieved(self, observation):
        """
        Returns True if "chassis_bottom_link" is withing the (self._tolerance_x, self._tolerance_y) 
        distance from the self_target_transform. In other words, checks if goal state is reached by the agent In other words, checks 
        if goal state is reached by the agent.
        """
        # Get (translation_x, translation_y) of forklift robotA
        agent_location = observation["forklift_robot_tf_observation"]["chassis_bottom_link"]["transform"][:2] 

        # Check if agent is withing the tolerance of target_location
        if (abs(agent_location[0] - self._target_transform[0]) <= self._tolerance_x) and \
            (abs(agent_location[1] - self._target_transform[1]) <= self._tolerance_y):
            return True
        else:
            return False

    