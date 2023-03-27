import gym
from gym import spaces
import numpy as np
import cv2
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import time
import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from forklift_gym_env.envs.utils import generate_and_launch_ros_description_as_new_process, read_yaml_config, \
    ObservationType, RewardType, ActionType
from forklift_gym_env.envs.simulation_controller import SimulationController

# Sensor Subscribers
from forklift_gym_env.envs.sensor_subscribers.forklift_robot_tf_subscriber import ForkliftRobotTfSubscriber
from forklift_gym_env.envs.sensor_subscribers.forklift_robot_frame_listener import FrameListener
from forklift_gym_env.envs.sensor_subscribers.get_entity_state_client import GetEntityStateClient
from forklift_gym_env.envs.sensor_subscribers.depth_camera_raw_image_subscriber import DepthCameraRawImageSubscriber
from forklift_gym_env.envs.sensor_subscribers.collision_detection_subscriber import CollisionDetectionSubscriber

# Controller Publishers
from forklift_gym_env.envs.controller_publishers.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher
from forklift_gym_env.envs.controller_publishers.fork_joint_controller_cmd_publisher import ForkJointContCmdPublisher


from forklift_gym_env.rl.sb3_HER.utils import flatten_and_concatenate_observation

class ForkliftEnvSb3HER(gym.GoalEnv):
    metadata = {
        "render_modes": ["no_render", "show_depth_camera_img_raw", "draw_coordinates"], 
        # "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None):
        # Read in parameters from config.yaml
        config_path = 'build/forklift_gym_env/forklift_gym_env/config/config.yaml'
        self.config = read_yaml_config(config_path)

        # Set observation_space, _get_obs method, and action_space
        self.obs_types = [ObservationType(obs_type) for obs_type in self.config["observation_types"]]
        self.observation_space, self._get_obs = self.observation_space_factory(obs_types = self.obs_types)
        self.act_types = [ActionType(act_type) for act_type in self.config["action_types"]]
        self.action_space = self.action_space_factory(act_types = self.act_types)
        self.reward_types = [RewardType(reward_type) for reward_type in self.config["reward_types"]]
        self.calc_reward = self.calculate_reward_factory(reward_types = self.reward_types)

        # Set render_mode
        for x in self.config["render_mode"]:
            assert x in ForkliftEnvSb3HER.metadata['render_modes'] 
        self.render_mode = self.config['render_mode']

        # self.ros_clock will be used to check that observations are obtained after the actions are taken
        self.ros_clock = None

        # -------------------- 
        # Start gazebo simulation, robot_state_publisher
        self.gazebo_launch_subp = generate_and_launch_ros_description_as_new_process(self.config) # Reference to subprocess that runs these things

        # -------------------- 
        # Listen to sensors: ============================== 
        rclpy.init()

        # -------------------- /gazebo/get_entity_state
        self.get_entity_state_client = GetEntityStateClient()
        # -------------------- 
        # -------------------- /camera/depth/image/raw
        self.depth_camera_raw_image_subscriber = self.initialize_depth_camera_raw_image_subscriber(normalize_img = True)
        # -------------------- 
        # -------------------- /collision_detections/*
        self.collision_detection_states = {}
        self.collision_detection_subscribers = []
        for link_name in self.config['collision_detection_link_names']:
            self.collision_detection_subscribers.append(self.initialize_collision_detection_subscriber(link_name))
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
        

    def _get_obs_target_transform_decorator(self, func):
        def _get_obs_target_transform(self):
            return {
                'target_transform_observation': self._target_transform,
            }
        

        def f():
            return {
                **(func()),
                **(_get_obs_target_transform(self))
            }
            # do NOT reset observations for next iteration because target_transform_observation does not change through an episode

        return f


    def _get_obs_forklift_position_decorator(self, func):
        def _get_obs_position(self):
            # Obtain a tf observation which belongs to a time after the action was taken
            current_forklift_robot_position_obs = None
            flag = True
            while (current_forklift_robot_position_obs is None) or flag:
                # Obtain forklift_robot frame observation -----
                t = self.get_entity_state_client.get_entity_state(\
                    "forklift_bot::base_link::base_link_fixed_joint_lump__chassis_bottom_link_collision_collision", \
                        "world")
                if t is not None and (t.success == True):
                    current_forklift_robot_position_obs = t
                    flag = False

                    # Make sure that observation was obtained after the action was taken at least 'step_duration' time later
                    if (int(str(t.header.stamp.sec) \
                        + (str(t.header.stamp.nanosec))) \
                            < (self.ros_clock.nanoseconds + self.config['step_duration'])):
                            flag = True
                            break

            # --------------------------------------------
            return {
                'forklift_position_observation': {
                    'chassis_bottom_link': {
                        'pose': {
                            'position': current_forklift_robot_position_obs.state.pose.position,
                            'orientation': current_forklift_robot_position_obs.state.pose.orientation
                        },
                        'twist': {
                            'linear': current_forklift_robot_position_obs.state.twist.linear,
                            'angular': current_forklift_robot_position_obs.state.twist.angular
                        }
                    } 
                },
            }
        

        def f():
            return {
                **(func()),
                **(_get_obs_position(self))
            }

        return f


    def _get_obs_pallet_position_decorator(self, func):
        def _get_obs_position(self):
            # Obtain a tf observation which belongs to a time after the action was taken
            current_pallet_position_obs = None
            flag = True
            while (current_pallet_position_obs is None) or flag:
                # Obtain forklift_robot frame observation -----
                t = self.get_entity_state_client.get_entity_state("pallet", "world")
                if t is not None and (t.success == True):
                    current_pallet_position_obs = t
                    flag = False

                    # Make sure that observation was obtained after the action was taken at least 'step_duration' time later
                    if (int(str(t.header.stamp.sec) \
                        + (str(t.header.stamp.nanosec))) \
                            < (self.ros_clock.nanoseconds + self.config['step_duration'])):
                            flag = True
                            break

            # --------------------------------------------
            return {
                'pallet_position_observation': {
                    'pallet_model': {
                        'pose': {
                            'position': current_pallet_position_obs.state.pose.position,
                            'orientation': current_pallet_position_obs.state.pose.orientation
                        },
                        'twist': {
                            'linear': current_pallet_position_obs.state.twist.linear,
                            'angular': current_pallet_position_obs.state.twist.angular
                        }
                    } 
                },
            }
        

        def f():
            return {
                **(func()),
                **(_get_obs_position(self))
            }

        return f


    def _get_obs_depth_camera_raw_image_decorator(self, func):
        def _get_obs_depth_camera_raw_image(self):
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
            # reset observations for next iteration
            self.depth_camera_img_observation = None 

            return {
                'depth_camera_raw_image_observation': depth_camera_raw_image_observation,
            }


        def f():
            return {
                **(func()),
                **(_get_obs_depth_camera_raw_image(self))
            }

        return f


    def _get_obs_collision_detection_decorator(self, func):
        def _get_obs_collision_detection(self):
            """
            Returns collision detection observations that are being published by ros_gazebo_collision_detection_plugin.
            """
            collision_detection_observations = {} # holds finalized up to date collision detection information from all contact sensors
            for subscriber in self.collision_detection_subscribers:
                # Check that the observation is from after the action was taken
                cur_collision_msg = None
                while (cur_collision_msg is None) or \
                    (int(str(cur_collision_msg.header.stamp.sec) \
                        + (str(cur_collision_msg.header.stamp.nanosec))) \
                            < (self.ros_clock.nanoseconds + self.config['step_duration'])): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.  
                    # Obtain new collision_detection_observation: -----
                    rclpy.spin_once(subscriber)
                    if subscriber.link_name in self.collision_detection_states:
                        cur_collision_msg = self.collision_detection_states[subscriber.link_name]

                collision_detection_observations[subscriber.link_name] = cur_collision_msg # record obs

                # Get obs by converting ROS msg to dict (This conversion is required for observation_spaces of gym)
                collision_detection_observations[subscriber.link_name] = {
                    "header": {
                        "stamp": {
                            "sec": cur_collision_msg.header.stamp.sec,
                            "nanosec": cur_collision_msg.header.stamp.nanosec
                        },
                        "frame_id": cur_collision_msg.header.frame_id
                    },
                    "states": [{
                        "info": contactState.info,
                        "collision1_name": contactState.collision1_name,
                        "collision2_name": contactState.collision2_name,
                        "wrenches": [{
                            "force": {
                                "x": wrench.force.x,
                                "y": wrench.force.y,
                                "z": wrench.force.z,
                            },
                            "torque": {
                                "x": wrench.torque.x,
                                "y": wrench.torque.y,
                                "z": wrench.torque.z,
                            },

                        } for wrench in contactState.wrenches],
                        "total_wrench": {
                            "force": {
                                "x": contactState.total_wrench.force.x,
                                "y": contactState.total_wrench.force.y,
                                "z": contactState.total_wrench.force.z,
                            },
                            "torque": {
                                "x": contactState.total_wrench.torque.x,
                                "y": contactState.total_wrench.torque.y,
                                "z": contactState.total_wrench.torque.z,
                            },
                        },
                        "contact_positions": [{
                                "x": contact_position.x,
                                "y": contact_position.y,
                                "z": contact_position.z,

                        } for contact_position in contactState.contact_positions],

                        "contact_normals": [{
                                "x": contact_normal.x,
                                "y": contact_normal.y,
                                "z": contact_normal.z,

                        } for contact_normal in contactState.contact_normals],
                        "depths": contactState.depths
                    } for contactState in cur_collision_msg.states]
                }

            # reset observations for next iteration
            self.collision_detection_states = {} 

            return {
                'collision_detection_observations': collision_detection_observations,
            }


        def f():
            return {
                **(func()),
                **(_get_obs_collision_detection(self))
            }

        return f


    def _get_info(self, reward, diff_cont_msg, observation):
        info = {
            "iteration": self.cur_iteration,
            "max_episode_length": self.max_episode_length,
            "reward": reward,
            "agent_location": [observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x,\
                 observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y], # [translation_x, translation_y]
            "target_location": self._target_transform, # can be changed with: observation['pallet_position']['pallet_model']['pose']['position']
            "verbose": self.config["verbose"]
        }
        return info
    

    def reset(self):
        seed = self.config["seed"]
        # super().reset()

        self.cur_iteration = 0

        # Choose the agent's location uniformly at random
        # self._agent_location = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]
        # self._agent_location *= np.array([0, 1]) # make it only change in y axis
        self._agent_location = np.array([0.0, 0.0]) # fix forklift start state to origin

        # Sample the target's location randomly until it does not coincide with the agent's location
        self._target_transform = np.array([0.0, 0.0]) # in the range [-10, 10]
        self._target_transform[0] = np.random.random(size=(1,)) * 20 + (-10) # x in the range [-10, 10]
        self._target_transform[1] = np.random.random(size=(1,)) * 4 - 2 # x in the range [-2, 2]
        while np.array_equal(self._target_transform, self._agent_location):
            # self._target_transform = np.random.random(size=2) * 20 - 10 # in the range [-10, 10]
            # self._target_transform *= np.array([0, 1]) # make it only change in y axis
            # self._target_transform[0] = np.random.random(size=(1,)) * 4 - 2 # x in the range [-2, 2]
            self._target_transform[0] = np.random.random(size=(1,)) * 20 + (-10) # y in the range [-10, 10]
        if self._target_transform[0] > -4 and self._target_transform[0] < 4:
            self._target_transform[0] = 5 * np.sign(np.random.random(size=(1,)))

 

        # Unpause sim so that simulation can be reset
        self.simulation_controller_node.send_unpause_physics_client_request()

        # Send 'no action' action to diff_cmd_cont of forklift robot
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = 0.0 # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = 0.0 # use this one
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)


        # send 'no action' action to fork_joint_cont of forklift robot
        # convert fork_joint_cont_action to Float64MultiArray message
        fork_joint_cont_msg = Float64MultiArray()
        fork_joint_cont_msg.data = [0.0]
        # Take fork_joint_cont action
        self.fork_joint_cont_cmd_publisher.publish_cmd(fork_joint_cont_msg)


        # Reset the simulation & world (gazebo)
        self.simulation_controller_node.send_reset_simulation_request()

        # Change agent location in the simulation, activate ros_controllers
        self.simulation_controller_node.change_entity_location(self._agent_entity_name, self._agent_location, self._ros_controller_names, self.config['agent_pose_position_z'])
        
        # Change pallet model's location in the simulation
        self.simulation_controller_node.change_entity_location(self._pallet_entity_name, self._target_transform, [], \
            0.0, self.config, spawn_pallet=True)

        self.ros_clock = self.diff_cont_cmd_vel_unstamped_publisher.get_clock().now()

        # Get observation
        observation = self._get_obs()

        # Convert nested Dict obs to flat obs array for SB3
        observation_flat, achieved_state, goal_state, observation = flatten_and_concatenate_observation(observation, self)
        # Concat observation_flat with goal_state for sb3
        # import torch
        # observation_flat = torch.concat([observation_flat, torch.tensor(goal_state)], dim=-1)

        obs_dict = {
            'observation': observation_flat.numpy(),
            'achieved_goal': achieved_state.numpy(),
            'desired_goal': goal_state
        }
        # Pause simuation so that obseration does not change until another action is taken
        self.simulation_controller_node.send_pause_physics_client_request()

        # Render
        self._coordinate_image = np.zeros((20, 20 , 3), np.uint8)
        self._coordinate_image = {
            'forklift_coordinates': [],
            'target_coordinate': None
        }
        # Mark forklift location
        self._coordinate_image['forklift_coordinates'].append(self._agent_location)
        # Mark target location
        self._coordinate_image['target_coordinate'] = self._target_transform
        # Initialize and render plot
        plt.close()
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.autoscale()
        self.render(observation)


        # return observation_flat, self._get_info(None, diff_cont_msg, observation)
        return obs_dict

    
    def step(self, action):
        self.cur_iteration += 1

        # Unpause simulation so that action can be taken
        self.simulation_controller_node.send_unpause_physics_client_request()

        # Set diff_cont action
        # diff_cont_action = action['diff_cont_action'] 
        diff_cont_action = action

        # convert diff_cont_action to Twist message
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = float(diff_cont_action[0]) # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 0.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = float(diff_cont_action[1]) # use this one
        # Take diff_cont action
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)


        # set fork_joint_cont action
        # fork_joint_cont_action = action['fork_joint_cont_action']
        # convert fork_joint_cont_action to Float64MultiArray message
        fork_joint_cont_msg = Float64MultiArray()
        # fork_joint_cont_msg.data = fork_joint_cont_action.tolist()
        fork_joint_cont_msg.data = [0.0]
        # Take fork_joint_cont action
        self.fork_joint_cont_cmd_publisher.publish_cmd(fork_joint_cont_msg)


        # Get observation after taking the action
        self.ros_clock = self.diff_cont_cmd_vel_unstamped_publisher.get_clock().now() # will be used to make sure observation is coming from after the action was taken

        observation = self._get_obs()

        # Convert nested Dict obs to flat obs array for SB3
        observation_flat, achieved_state, goal_state, observation = flatten_and_concatenate_observation(observation, self)
        # Concat observation_flat with goal_state for sb3
        # import torch
        # observation_flat = torch.concat([observation_flat, torch.tensor(goal_state)], dim=-1).numpy()

        obs_dict = {
            'observation': observation_flat.numpy(),
            'achieved_goal': achieved_state.numpy(),
            'desired_goal': goal_state
        }

        # Pause simuation so that obseration does not change until another action is taken
        self.simulation_controller_node.send_pause_physics_client_request()

        # Calculate reward
        reward = self.calc_reward(obs_dict['achieved_goal'], obs_dict['desired_goal']) 

        # Check if episode should terminate 
        done = bool(self.cur_iteration >= (self.max_episode_length)) or (self.check_goal_achieved(obs_dict['achieved_goal'], full_obs=False))

        # Get info
        info = self._get_info(reward, diff_cont_msg, observation) 

        # Render
        # Mark forklift location
        self._coordinate_image['forklift_coordinates'].append(obs_dict['achieved_goal'])
        self.render(observation)

        # return observation_flat, reward, done, False, info # (observation, reward, done, truncated, info)
        return obs_dict, reward, done, info # (observation, reward, done, truncated, info)


    def render(self, observation): 
        if self.render_mode is not None:
            self._render_frame(observation)
    
    def _render_frame(self, observation):
        if "draw_coordinates" in self.render_mode:
            # Draw Forklift coordinates
            self.ax.clear()
            fork_coordinates = np.array(self._coordinate_image['forklift_coordinates'])
            x = fork_coordinates[:,0]
            y = fork_coordinates[:,1]
            self.ax.plot(x, y, 'b-', label="forklift coordinates")
            # Draw Target coordinates
            self.ax.plot(self._coordinate_image['target_coordinate'][0], self._coordinate_image['target_coordinate'][1], 'r*', label='pallet coordinates')

            # Label the plot
            self.ax.set_title('Rendering Coordinates')
            self.ax.legend()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


        if "show_depth_camera_img_raw" in self.render_mode:
            depth_camera_img = observation['depth_camera_raw_image_observation']
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
    

    def initialize_collision_detection_subscriber(self, link_name):
        """
        Inputs:
            link_name (str): name of the link that contacts msgs are being published for. For example for a given link_name
                of 'chassis_bottom_link', this will subscribe to ros topic: '/collision_detections/link_name'
        """
        def collision_detection_cb(msg):
            self.collision_detection_states[link_name] = msg

        return CollisionDetectionSubscriber(collision_detection_cb, link_name)


    def observation_space_factory(self, obs_types):
        """
        Returns observation space and corresponding _get_obs method that corresponds to the given obs_type.
        Inputs:
            obs_type (List[ObservationType]): specificies which observations are being used.
        """
        obs_space_dict = {}
        for obs_type in obs_types:
            assert obs_type in ObservationType

            # Extend observation_space according to obs_type 
            if obs_type == ObservationType.FORK_POSITION:
               obs_space_dict["forklift_position_observation"] = spaces.Dict({
                        "chassis_bottom_link": spaces.Dict({
                            "pose": spaces.Dict({
                                "position": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
                                "orientation": spaces.Box(low = -float("inf") * np.ones((4,)), high = float("inf") * np.ones((4,)), dtype = np.float32)
                            }),
                            "twist": spaces.Dict({
                                "linear": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
                                "angular": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32)
                            })
                        })
               })

            elif obs_type == ObservationType.PALLET_POSITION:
               obs_space_dict["pallet_position_observation"] = spaces.Dict({
                        "pallet_model": spaces.Dict({
                            "pose": spaces.Dict({
                                "position": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
                                "orientation": spaces.Box(low = -float("inf") * np.ones((4,)), high = float("inf") * np.ones((4,)), dtype = np.float32)
                            }),
                            "twist": spaces.Dict({
                                "linear": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
                                "angular": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32)
                            })
                        })
               })

            elif obs_type == ObservationType.TARGET_TRANSFORM:
               obs_space_dict["target_transform_observation"] = spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float32) # TODO: set these values to min and max from ros diff_controller

            elif obs_type == ObservationType.DEPTH_CAMERA_RAW_IMAGE:
                obs_space_dict['depth_camera_raw_image_observation'] = spaces.Box(low = -float("inf") * \
                        np.ones(tuple(self.config['depth_camera_raw_image_dimensions'])), \
                            high = float("inf") * np.ones(tuple(self.config['depth_camera_raw_image_dimensions']), dtype = np.float32))
            
            elif obs_type == ObservationType.COLLISION_DETECTION:
                d = {}
                for link_name in self.config['collision_detection_link_names']:
                    d[link_name] = spaces.Dict({
                    "header": spaces.Dict({
                        "stamp": spaces.Dict({
                            "sec": spaces.Box(low = 0.0, high = float("inf"), shape = (1, ), dtype = np.int32),
                            "nanosec": spaces.Box(low = 0.0, high = float("inf"), shape = (1, ), dtype = np.int32)
                        }),
                        "frame_id": spaces.Text(max_length = 500),
                    }),
                    "states": spaces.Sequence(spaces.Dict({
                        "info": spaces.Text(max_length = 1000),
                        "collision1_name": spaces.Text(max_length = 500),
                        "collision2_name": spaces.Text(max_length = 500),
                        "wrenches": spaces.Sequence(spaces.Dict({
                            "force": spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            }),
                            "torque": spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            })
                        })),
                        "total_wrench": spaces.Dict({
                            "force": spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            }),
                            "torque": spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            })
                        }),
                        "contact_positions": spaces.Sequence(spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            })),
                        "contact_normals": spaces.Sequence(spaces.Dict({
                                "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
                                "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
                            })),
                        "depths": spaces.Sequence(spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64))
                    }))
                })


                obs_space_dict['collision_detection_observations'] = spaces.Dict(d)
        
        def func():
            return {}

        _get_obs = func
        
        # Extend _get_obs function according to obs_types
        for obs_type in obs_types:
            if obs_type == ObservationType.FORK_POSITION:
                _get_obs = self._get_obs_forklift_position_decorator(_get_obs) # get absolute position of the forklift_robot's links

            elif obs_type == ObservationType.PALLET_POSITION:
                _get_obs = self._get_obs_pallet_position_decorator(_get_obs) # get absolute position of the pallet model 

            elif obs_type == ObservationType.TARGET_TRANSFORM:
                _get_obs = self._get_obs_target_transform_decorator(_get_obs) # get initial absolute position of the target/pallet 

            elif obs_type == ObservationType.DEPTH_CAMERA_RAW_IMAGE:
                _get_obs = self._get_obs_depth_camera_raw_image_decorator(_get_obs)

            elif obs_type == ObservationType.COLLISION_DETECTION:
                _get_obs = self._get_obs_collision_detection_decorator(_get_obs)

        # return spaces.Dict(obs_space_dict), _get_obs #TODO: change self._get_obs method being returned (use decorator pattern)
        # return spaces.Box(low = -float("inf") * np.ones((15,)), high = float("inf") * np.ones((15,)), dtype = np.float64), _get_obs #TODO: change self._get_obs method being returned (use decorator pattern)
        return spaces.Dict({
            'observation': spaces.Box(low = -float("inf") * np.ones((6,)), high = float("inf") * np.ones((6,)), dtype = np.float64),
            'achieved_goal': spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float64),
            'desired_goal': spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float64)
        }), _get_obs

    
    
    def calculate_reward_factory(self, reward_types):
        """
        Returns a function that calculates reward which corresponds to the given reward_type
        Inputs:
            reward_type (List[RewardType]): specifies which reward function is used.
        """
        for reward_type in reward_types:
            assert reward_type in RewardType

        # Return corresponding reward calculation function by accumulating the rewards returned by the functions specified in the reward_types
        def reward_func(observation, goal_state = None):
            reward = 0

            if RewardType.L2_DIST in reward_types:
                def calc_reward_L2_dist(observation, goal_state):
                    """
                    Returns the negative L2 distance between the (translation_x, translation_y) coordinates 
                    of forklift_robot_transform and target_transform.
                    Inputs:
                        observation: returned by self._get_obs()
                    Returns:
                        reward: negative L2 distance

                    """
                    # forklift_robot_transform = observation['forklift_position_observation']
                    # Return negative L2 distance btw chassis_bottom_link and the target location as reward
                    # Note that we are only using translation here, NOT using rotation information
                    # robot_transform_translation = [forklift_robot_transform['chassis_bottom_link']['pose']['position'].x, \
                    #     forklift_robot_transform['chassis_bottom_link']['pose']['position'].y] # [translation_x, translation_y]
                    robot_transform_translation = [observation[0], observation[1]] # [translation_x, translation_y]
                    if goal_state is None:
                        l2_dist = np.linalg.norm(robot_transform_translation - self._target_transform)
                    else:
                        l2_dist = np.linalg.norm(robot_transform_translation - goal_state)
                    return - l2_dist

                reward += calc_reward_L2_dist(observation, goal_state)
    
            if RewardType.COLLISION_PENALTY in reward_types:
                def calc_reward_collision_penalty(observation):
                    """
                    Returns a Reward which penalizes agent's collision with other (non-ground) objects.

                    """
                    penalty_per_collision = -5 # TODO: fine-tune this value
                    collision_detection_observations = observation['observation']['collision_detection_observations']
                    # Check for agents collisions with non-ground objects
                    unique_non_ground_contacts = {} # holds ContactsState msgs for the collisions with objects that are not the ground
                    for state in collision_detection_observations.values():
                        unique_non_ground_contacts = {**unique_non_ground_contacts, **CollisionDetectionSubscriber.get_non_ground_collisions(contactsState_msg = state)}
                    return penalty_per_collision * len(unique_non_ground_contacts)

                reward += calc_reward_collision_penalty(observation)
            
            if RewardType.BINARY in reward_types:
                def calc_reward_binary(observation, goal_state):
                    """
                    Returns 1 if goal_achieved else 0.
                    Inputs:
                        observation: returned by self._get_obs()
                    Returns:
                        reward: 1 or 0 (binary reward).
                    """
                    return int(self.check_goal_achieved(observation, goal_state, full_obs=False))

                reward += calc_reward_binary(observation, goal_state)

            return reward

        return reward_func


    def action_space_factory(self, act_types):
        """
        Returns observation space that corresponds to obs_type
        Inputs:
            act_type (List(ActionType)): specifies which actions can be taken by the agent.
        """
        for act_type in act_types:
            assert act_type in ActionType

        # Set action space according to act_type 
        d =  {
                "diff_cont_action": spaces.Box(low = -10 * np.ones((2)), high = 10 * np.ones((2)), dtype=np.float32), #TODO: set this to limits from config file
                "fork_joint_cont_action": spaces.Box(low = -2.0, high = 2.0, shape = (1,), dtype = np.float64) # TODO: set its high and low limits correctly
        }

        # return spaces.Dict(d)

        # Flatten action_space for SB3
        return spaces.Box(low= -1 * np.ones((2)), high = 1 * np.ones((2)), dtype=np.float32)
        # return spaces.Box(low= -1 * np.ones((1)), high = 1 * np.ones((1)), dtype=np.float32)



    def check_goal_achieved(self, observation, goal_state = None, full_obs = True):
        """
        Returns True if "chassis_bottom_link" is withing the (self._tolerance_x, self._tolerance_y) 
        distance from the self_target_transform. In other words, checks if goal state is reached by the agent In other words, checks 
        if goal state is reached by the agent.
        """
        goal_achieved_list = []
        # Vectorized implementation -->
        if len(observation.shape) > 1: # Batch inputs
            for i in range(len(observation)):
                cur_observation = observation[i, :]
            
                # Get (translation_x, translation_y) of forklift robot
                if full_obs:
                    agent_location = [cur_observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].x, \
                        cur_observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].y]
                else:
                    agent_location = cur_observation

                # Check if agent is withing the tolerance of target_location
                if goal_state is None:
                    if (abs(agent_location[0] - self._target_transform[0]) <= self._tolerance_x) and \
                        (abs(agent_location[1] - self._target_transform[1]) <= self._tolerance_y):
                        goal_achieved_list.append(True)
                    else:
                        goal_achieved_list.append(False)
                else:
                    cur_goal_state = goal_state[i, :]
                    if (abs(agent_location[0] - cur_goal_state[0]) <= self._tolerance_x) and \
                        (abs(agent_location[1] - cur_goal_state[1]) <= self._tolerance_y):
                        goal_achieved_list.append(True)
                    else:
                        goal_achieved_list.append(False)
        
            return goal_achieved_list

        else: # Non-vectorized implementation:
                # Get (translation_x, translation_y) of forklift robot
                if full_obs:
                    agent_location = [observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].x, \
                        observation["forklift_position_observation"]["chassis_bottom_link"]["pose"]['position'].y]
                else:
                    agent_location = observation
        
                # Check if agent is withing the tolerance of target_location
                if goal_state is None:
                    if (abs(agent_location[0] - self._target_transform[0]) <= self._tolerance_x) and \
                        (abs(agent_location[1] - self._target_transform[1]) <= self._tolerance_y):
                        return True
                    else:
                        return False
                else:
                    cur_goal_state = goal_state
                    if (abs(agent_location[0] - cur_goal_state[0]) <= self._tolerance_x) and \
                        (abs(agent_location[1] - cur_goal_state[1]) <= self._tolerance_y):
                        return True
                    else:
                        return False
            
        
    

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        # Vectorized binary reward calculation # TODO: integrate into calc_reward funciton
        # g = self.check_goal_achieved(achieved_goal, desired_goal, full_obs=False)
        # if type(g) == list:
        #     return np.array([int(success) for success in g])
        # elif type(g) == bool:
        #     return g

        # Process batch data one by one
        goals = []
        for a_goal, d_goal in zip(achieved_goal, desired_goal):
            goals.append(self.calc_reward(a_goal, d_goal))
        return np.array(goals)

    