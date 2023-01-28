import gym
import time
from gym import spaces
import numpy as np
from forklift_gym_env.envs.utils import generate_and_launch_ros_description_as_new_process
from forklift_gym_env.envs.depth_camera_raw_image_subscriber import DepthCameraRawImageSubscriber
import rclpy
import cv2
from cv_bridge import CvBridge
from forklift_gym_env.envs.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher
from forklift_gym_env.envs.forklift_robot_tf_subscriber import ForkliftRobotTfSubscriber
from forklift_gym_env.envs.reset_simulation_client import ResetSimulationClientAsync
from geometry_msgs.msg import Twist
from forklift_gym_env.envs.pause_pyhsics_client import PausePhysicsClient
from forklift_gym_env.envs.unpause_pyhsics_client import UnpausePhysicsClient
from forklift_gym_env.envs.simulation_controller import SimulationController
import numpy as np


class ForkliftEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "no_render"], # TODO: set this to supported types
        # "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None):
       # Set observation_space, _get_obs method, and action_space
       self.observation_space, self._get_obs = self.observation_space_factory(obs_type="tf_only")
       self.action_space = self.action_space_factory(act_type="diff_cont")
       self.calc_reward = self.calculate_reward_factory(reward_type="L2_dist")

       # Set render_mode
       assert render_mode is None or render_mode in self.metadata["render_modes"]
       self.render_mode = render_mode

       # self.ros_clock will be used to check that observations are obtained after the actions are taken
       self.ros_clock = None

       # -------------------- 
       # Start gazebo simulation, spawn forklift model, load ros controllers
       self.launch_subp = generate_and_launch_ros_description_as_new_process() # Reference to subprocess that runs these things

       # -------------------- 
       # Subscribe to sensors: ============================== 
       rclpy.init()
       # -------------------- /camera/depth/image/raw
       self.depth_camera_raw_image_subscriber = self.initialize_depth_camera_raw_image_subscriber()

       # -------------------- /tf
       self.forklift_robot_tf_subscriber = self.initialize_forklift_robot_tf_subscriber()
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
       
       # HYPERPARAMETERS: --------------------  # TODO: get these from config
       self.cur_iteration = 0
       self.max_episode_length = 50
       self._entity_name = 'forklift_bot'
       self._ros_controller_names = ['joint_broad', 'fork_joint_controller', 'diff_cont']
       self._tolerance_x = 1.115 # chassis_bottom_transform's x location's tolerable distance to target_location's x coordinate for agent to achieve the goal state
       self._tolerance_y = 1.59  # chassis_bottom_transform's y location's tolerable distance to target_location's y coordinate for agent to achieve the goal state
       # -------------------------------------

       self._target_transform = { # TODO: take this as a parameter and set it randomly, This is the Goal State
        'transform': np.asarray([0.0, 0.0]) # [translation_x, translation_y]
       }


    def _get_obs_tf_only(self):
        # Obtain an observation which belongs to a time after the action was taken
        current_forklift_robot_tf_obs = {}
        flag = True
        while current_forklift_robot_tf_obs == {} or flag:
            # Obtain forklift_robot_tf observation -----
            rclpy.spin_once(self.forklift_robot_tf_subscriber)
            current_forklift_robot_tf_obs = self.forklift_robot_tf_state

            flag = False
            for k, v in current_forklift_robot_tf_obs.items():
                if v['time'] < self.ros_clock.nanoseconds: # make sure that observation was obtained after the action was taken
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


    def _get_obs_camera(self):
        # Check that the observation is from after the action was taken
        while (current_depth_camera_raw_image_obs is None) or (int(str(current_depth_camera_raw_image_obs["header"].stamp.sec) + (str(current_depth_camera_raw_image_obs["header"].stamp.nanosec))) < self.ros_clock.nanoseconds):
            # Obtain new depth_camera_raw_image_observation: -----
            rclpy.spin_once(self.depth_camera_raw_image_subscriber)
            current_depth_camera_raw_image_obs = self.depth_camera_img_observation
            rclpy.spin_once(self.forklift_robot_tf_subscriber)

        depth_camera_raw_image_observation = current_depth_camera_raw_image_obs["image"] # get image
        # --------------------------------------------


        # Forklift_robot_tf observation -----
        rclpy.spin_once(self.forklift_robot_tf_subscriber)
        current_forklift_robot_tf_obs = self.forklift_robot_tf_state

        # Check that the observation is from after the action was taken
        flag = True
        while current_forklift_robot_tf_obs == {} or flag:
            rclpy.spin_once(self.forklift_robot_tf_subscriber)
            current_forklift_robot_tf_obs = self.forklift_robot_tf_state

            flag = False
            for k, v in current_forklift_robot_tf_obs.items():
                if v['time'] < self.ros_clock.nanoseconds:
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
            "target_location": self._target_transform
        }
        return info
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cur_iteration = 0

        # Choose the agent's location uniformly at random
        self._agent_location = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]
        # Change agent location in the simulation
        self.simulation_controller_node.change_agent_location(self._entity_name, self._agent_location, self._ros_controller_names)

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

        time.sleep(2)
        # Reset the simulation (gazebo)
        self.simulation_controller_node.send_reset_simulation_request()

        self.ros_clock = self.depth_camera_raw_image_subscriber.get_clock().now()

        return self._get_obs(), self._get_info(None, diff_cont_msg)

    
    def step(self, action):
        # if self.render_mode == "human": # TODO: handle this once the simulation is figured out with gazebo
        #     self._render_frame()

        # -------------------- 
        self.cur_iteration += 1

        # Unpause simulation so that action can be taken
        time.sleep(0.05)
        print("line 272 next step is to call unpause_sim")
        self.simulation_controller_node.send_unpause_physics_client_request()

        # Take action
        diff_cont_action = action['diff_cont_action'] 
        # convert diff_cont_action to Twist message
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = float(diff_cont_action[0]) # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = float(diff_cont_action[1]) # use this one
        print("line 286 next step is to publish_cmd")
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        # Get observation after taking the action
        self.ros_clock = self.depth_camera_raw_image_subscriber.get_clock().now() # will be used to make sure observation is coming from after the action was taken
        print("line 289 getting observations 1")
        observation = self._get_obs()

        # Pause simuation so that obseration does not change until another action is taken
        print("line 302 observations arrived next step is to pause the simulation 2")
        self.simulation_controller_node.send_pause_physics_client_request()

        # Calculate reward
        reward = self.calc_reward(observation['forklift_robot_tf_observation'], self._target_transform) 

        # Check if episode should terminate #TODO: check also if goal state is reached
        done = bool(self.cur_iteration >= (self.max_episode_length - 1)) or (self.check_goal_achieved(observation))

        # Get info
        info = self._get_info(reward, diff_cont_msg) 

        # -------------------- 

        return observation, reward, done, False, info # (observation, reward, done, truncated, info)


    def render(self): # TODO: rewrite for sensory information such as camera
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()
        # TODO: render sensory information such as camera mounted on the forklift
        raise NotImplementedError("render not supported")
    
    def _render_frame(self):
        raise NotImplementedError("_render_frame not supported")


    def close(self): # TODO: close any resources that are open (e.g. ros2 nodes, gazebo, rviz e.t.c)
        # TODO: Close cv2 window if render is used

        # delete ros nodes
        self.depth_camera_raw_image_subscriber.destroy_node()
        self.diff_cont_cmd_vel_unstamped_publisher.destroy_node()
        # self.forklift_robot_tf_subscriber.destroy_node()
        rclpy.shutdown()

        self.launch_subp.join()
    

    # ============================================= Helper functions
    def initialize_depth_camera_raw_image_subscriber(self):
       self.depth_camera_img_observation = None
       def depth_camera_raw_image_subscriber_cb(msg):
        try:
            self.bridge
        except:
            self.bridge = CvBridge()
        depth_camera_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # depth_camera_img = cv2.normalize(depth_camera_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # Normalize the depth_camera_image to range [0,1]
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
    

    def observation_space_factory(self, obs_type = "tf_only"):
        """
        Returns observation space and corresponding _get_obs method that corresponds to the given obs_type
        Inputs:
            obs_type: supports "tf_only", "tf and depth_camera_raw".
        """
        assert obs_type in ["tf_only", "tf and depth_camera_raw"]

        # Set observation_space according to obs_type 
        if obs_type == "tf_only":
            return spaces.Dict({ 
                "forklift_robot_tf_observation": spaces.Dict({
                    "chassis_bottom_link": spaces.Dict({
                        "time": spaces.Box(low = 0.0, high = float("inf"), dtype = int),
                        "transform": spaces.Box(low = -float("inf") * np.ones((7,)), high = float("inf") * np.ones((7,)), dtype = float) # TODO: set these values to min and max from ros diff_controller
                        })
                    }),
            }), self._get_obs_tf_only

        elif obs_type == "tf and depth_camera_raw":
            return spaces.Dict({ 
                "depth_camera_raw_image_observation": spaces.Box(low = -float("inf") * np.ones((480, 640)), high = float("inf") * np.ones((480, 640), dtype = np.float32)),
                "forklift_robot_tf_observation": spaces.Dict({
                    "chassis_bottom_link": spaces.Dict({
                        "time": spaces.Box(low = 0.0, high = float("inf"), dtype = int),
                        "transform": spaces.Box(low = -float("inf") * np.ones((7,)), high = float("inf") * np.ones((7,)), dtype = float)
                        })
                    }),
            }), self._get_obs_camera
    
    
    def calculate_reward_factory(self, reward_type = "L2_dist"):
        """
        Returns a function that calculates reward which corresponds to the given reward_type
        Inputs:
            reward_type: supports "L2_dist".
        """
        assert reward_type in ["L2_dist"]

        # return corresponding reward calculation funciton 
        if reward_type == "L2_dist":
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
    

    def action_space_factory(self, act_type = "diff_cont"):
        """
        Returns observation space that corresponds to obs_type
        Inputs:
            act_type: supports "diff_cont".
        """
        assert act_type in ["diff_cont"]

        # Set action space according to act_type 
        if act_type == "diff_cont":
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

    