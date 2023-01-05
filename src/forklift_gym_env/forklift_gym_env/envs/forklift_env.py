import gym
from gym import spaces
import numpy as np
from forklift_gym_env.envs.utils import generateLaunchDescriptionForkliftEnv, startLaunchServiceProcess
from forklift_gym_env.envs.depth_camera_raw_image_subscriber import DepthCameraRawImageSubscriber
import rclpy
import cv2
from cv_bridge import CvBridge
from forklift_gym_env.envs.diff_cont_cmd_vel_unstamped_publisher import DiffContCmdVelUnstampedPublisher
from forklift_gym_env.envs.forklift_robot_tf_subscriber import ForkliftRobotTfSubscriber
from forklift_gym_env.envs.reset_simulation_client import ResetSimulationClientAsync
from geometry_msgs.msg import Twist

class ForkliftEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], # TODO: set this to supported types
        "render_fps": 4 #TODO: set this
    }

    def __init__(self, render_mode = None):
       # set types of observation_space and action_space 
       self.observation_space = spaces.Dict({
        "depth_camera_raw_image_observation": spaces.Box(low = -float("inf") * np.ones((480, 640)), high = float("inf") * np.ones((480, 640), dtype = np.float32)),
        "forklift_robot_tf_observation": spaces.Dict({
            "chassis_bottom_link": spaces.Dict({
                "time": spaces.Box(low = 0.0, high = float("inf"), dtype = int),
                "transform": spaces.Box(low = -float("inf") * np.ones((7,)), high = float("inf") * np.ones((7,)), dtype = float)
                })
            }),
        # "agent": spaces.Box(low = 0, high = 1, shape=(2, ),  dtype=int),
        # "target": spaces.Box(low = 0, high = 1, shape=(2, ),  dtype=int)
       })
    #    self.action_space = spaces.Discrete(4) # TODO: change this
       self.action_space = spaces.Dict({
        "diff_cont_action": spaces.Box(low = -10 * np.ones((2)), high = 10 * np.ones((2)), dtype=np.float32) #TODO: set this to limits from config file
       })

       # set render_mode
       assert render_mode is None or render_mode in self.metadata["render_modes"]
       self.render_mode = render_mode

       # self.clock` will be a clock that is used to ensure that the environment is rendered at the correct frame rate in human-mode.
       self.clock = None
       self.ros_clock = None

       # -------------------- 
       # start gazebo simulation, spawn forklift model, start controllers
       launch_desc = generateLaunchDescriptionForkliftEnv() # generate launch description
       self.launch_subp = startLaunchServiceProcess(launch_desc)# start the generated launch description on a subprocess
       # -------------------- 

       # Subscribe to sensors: ============================== 
       rclpy.init()
       # -------------------- /camera/depth/image/raw
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

       self.depth_camera_raw_image_subscriber = DepthCameraRawImageSubscriber(depth_camera_raw_image_subscriber_cb)
       # rclpy.spin(self.depth_camera_raw_image_subscriber)
       # --------------------  
       # -------------------- /tf
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

       self.forklift_robot_tf_subscriber = ForkliftRobotTfSubscriber(forklift_robot_tf_cb)
    #    rclpy.spin(self.forklift_robot_tf_state)

       # -------------------- 
       # ====================================================

       # Create publisher for controlling forklift robot's joints: ============================== 
       # --------------------  /diff_cont/cmd_vel_unstamped
       self.diff_cont_cmd_vel_unstamped_publisher = DiffContCmdVelUnstampedPublisher()
       # -------------------- 
       # ====================================================

       # Create Clients for Services: ============================== 
       # --------------------  /reset_simulation
       self.reset_sim = ResetSimulationClientAsync()
       # -------------------- 
       # ====================================================
       
       # HYPERPARAMETERS: --------------------  # TODO: get these from config
       self.cur_iteration = 0
       self.max_episode_length = 5
       # -------------------------------------
       self.target_transform = { # TODO: take this as parameter and set it randomly
        'transform': np.asarray([0.0, 0.0, 0.0, \
            0.0, 0.0, 0.0, 0.0]) # [translation_x,translation_y, translation_z, rotation_x, rotation_y, rotation_z, rotation_w]
       }


    def _get_obs(self):
        # Depth_camera_raw_image_observation: -----
        # Update current observation
        rclpy.spin_once(self.depth_camera_raw_image_subscriber)
        current_depth_camera_raw_image_obs = self.depth_camera_img_observation

        # Check that the observation is from after the action was taken
        while (current_depth_camera_raw_image_obs is None) or (int(str(current_depth_camera_raw_image_obs["header"].stamp.sec) + (str(current_depth_camera_raw_image_obs["header"].stamp.nanosec))) < self.ros_clock.nanoseconds):
            # update current observation again
            rclpy.spin_once(self.depth_camera_raw_image_subscriber)
            current_depth_camera_raw_image_obs = self.depth_camera_img_observation
            rclpy.spin_once(self.forklift_robot_tf_subscriber)

        depth_camera_raw_image_observation = current_depth_camera_raw_image_obs["image"] # get image
        # --------------------------------------------


        # Forklift_robot_tf observation -----
        rclpy.spin_once(self.forklift_robot_tf_subscriber)
        current_forklift_robot_tf_obs = self.forklift_robot_tf_state

        # Check taht the observation is from after the action was taken
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


    def _get_info(self): # TODO: implement this
        # return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
        return {}
    

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int) # TODO: set this to agent (forklift) location at start in gazebo

        # # We will sample the target's location randomly until it does not coincide with the agent's location
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )


        # --------------
        # TODO: set self.target to something new and random

        self.cur_iteration = 0

        # Reset the simulation (gazebo)
        future_result = self.reset_sim.send_request()

        self.ros_clock = self.depth_camera_raw_image_subscriber.get_clock().now()

        return self._get_obs(), self._get_info()

    
    def step(self, action):
        # if self.render_mode == "human": # TODO: handle this once the simulation is figured out with gazebo
        #     self._render_frame()

        # -------------------- 
        self.cur_iteration += 1
        # Take action
        diff_cont_action = action['diff_cont_action'] # TODO: set this from action parameter of the step function
        # convert diff_cont_action to Twist message
        diff_cont_msg = Twist()
        diff_cont_msg.linear.x = float(diff_cont_action[0]) # use this one
        diff_cont_msg.linear.y = 0.0
        diff_cont_msg.linear.z = 0.0

        diff_cont_msg.angular.x = 1.0
        diff_cont_msg.angular.y = 0.0
        diff_cont_msg.angular.z = float(diff_cont_action[1]) # use this one
        self.diff_cont_cmd_vel_unstamped_publisher.publish_cmd(diff_cont_msg)

        # Get observation after taking the action
        self.ros_clock = self.depth_camera_raw_image_subscriber.get_clock().now() # will be used to make sure bservation is coming from after the action was taken
        observation = self._get_obs()

        # Calculate reward
        def calc_reward(forklift_robot_transform, target_transform):
            # Return negative L2 distance btw chassis_bottom_link and the target location as reward
            # Note that we are only using translation here, NOT using rotation information
            robot_transform_translation = forklift_robot_transform['chassis_bottom_link']['transform'][:3]
            target_translation = target_transform['transform'][:3]
            l2_dist = np.linalg.norm(robot_transform_translation - target_translation)
            return - l2_dist
            
        reward = calc_reward(observation['forklift_robot_tf_observation'], self.target_transform) # TODO: implement reward function

        # Check if episode should terminate
        done = bool(self.cur_iteration == self.max_episode_length)

        # Set info
        info = self._get_info() # TODO: set this with useful info such as forklift's manhattan distance to target location

        # -------------------- 

        return observation, reward, done, False, info # (observation, reward, done, truncated, info)


    def render(self): # TODO: rewrite for gazebo
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()
        raise NotImplemented
    
    def _render_frame(self):
        raise NotImplemented


    def close(self): # TODO: close any resources that are open (e.g. ros2 nodes, gazebo, rviz e.t.c)
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        # delete ros nodes
        self.depth_camera_raw_image_subscriber.destroy_node()
        self.diff_cont_cmd_vel_unstamped_publisher.destroy_node()
        self.forklift_robot_tf_subscriber.destroy_node()
        self.reset_sim.destroy_node()
        rclpy.shutdown()

        self.launch_subp.join()