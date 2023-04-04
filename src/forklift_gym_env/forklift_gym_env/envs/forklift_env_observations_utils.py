from gym import spaces
from forklift_gym_env.envs.utils import ObservationType 
import numpy as np





def observation_space_factory(env, obs_types):
    """
    Returns observation space and corresponding _get_obs method that corresponds to the given obs_types.
    Inputs:
        env (class ForkliftEnv): Reference to ForkliftEnv which will use the observations implemented here.
        obs_type (List[ObservationType]): specificies which observations are being used.
    """
    for obs_type in obs_types:
        assert obs_type in ObservationType

        # Extended observation_space according to obs_type. Currently NOT being used but left as a reference.
        # obs_space_dict = {}
        # if obs_type == ObservationType.FORK_POSITION:
        #     obs_space_dict["forklift_position_observation"] = spaces.Dict({
        #             "chassis_bottom_link": spaces.Dict({
        #                 "pose": spaces.Dict({
        #                     "position": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
        #                     "orientation": spaces.Box(low = -float("inf") * np.ones((4,)), high = float("inf") * np.ones((4,)), dtype = np.float32)
        #                 }),
        #                 "twist": spaces.Dict({
        #                     "linear": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
        #                     "angular": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32)
        #                 })
        #             })
        #     })

        # elif obs_type == ObservationType.PALLET_POSITION:
        #     obs_space_dict["pallet_position_observation"] = spaces.Dict({
        #             "pallet_model": spaces.Dict({
        #                 "pose": spaces.Dict({
        #                     "position": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
        #                     "orientation": spaces.Box(low = -float("inf") * np.ones((4,)), high = float("inf") * np.ones((4,)), dtype = np.float32)
        #                 }),
        #                 "twist": spaces.Dict({
        #                     "linear": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32),
        #                     "angular": spaces.Box(low = -float("inf") * np.ones((3,)), high = float("inf") * np.ones((3,)), dtype = np.float32)
        #                 })
        #             })
        #     })

        # elif obs_type == ObservationType.TARGET_TRANSFORM:
        #     obs_space_dict["target_transform_observation"] = spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float32) # TODO: set these values to min and max from ros diff_controller

        # elif obs_type == ObservationType.DEPTH_CAMERA_RAW_IMAGE:
        #     obs_space_dict['depth_camera_raw_image_observation'] = spaces.Box(low = -float("inf") * \
        #             np.ones(tuple(env.config['depth_camera_raw_image_dimensions'])), \
        #                 high = float("inf") * np.ones(tuple(env.config['depth_camera_raw_image_dimensions']), dtype = np.float32))
            
        # elif obs_type == ObservationType.COLLISION_DETECTION:
        #     d = {}
        #     for link_name in env.config['collision_detection_link_names']:
        #         d[link_name] = spaces.Dict({
        #         "header": spaces.Dict({
        #             "stamp": spaces.Dict({
        #                 "sec": spaces.Box(low = 0.0, high = float("inf"), shape = (1, ), dtype = np.int32),
        #                 "nanosec": spaces.Box(low = 0.0, high = float("inf"), shape = (1, ), dtype = np.int32)
        #             }),
        #             "frame_id": spaces.Text(max_length = 500),
        #         }),
        #         "states": spaces.Sequence(spaces.Dict({
        #             "info": spaces.Text(max_length = 1000),
        #             "collision1_name": spaces.Text(max_length = 500),
        #             "collision2_name": spaces.Text(max_length = 500),
        #             "wrenches": spaces.Sequence(spaces.Dict({
        #                 "force": spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 }),
        #                 "torque": spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 })
        #             })),
        #             "total_wrench": spaces.Dict({
        #                 "force": spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 }),
        #                 "torque": spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 })
        #             }),
        #             "contact_positions": spaces.Sequence(spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 })),
        #             "contact_normals": spaces.Sequence(spaces.Dict({
        #                     "x": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "y": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64),
        #                     "z": spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64)
        #                 })),
        #             "depths": spaces.Sequence(spaces.Box(low = -float("inf"), high = float("inf"), shape = (1, ), dtype = np.float64))
        #         }))
        #     })


        #     obs_space_dict['collision_detection_observations'] = spaces.Dict(d)
        
    def func():
        return {}

    _get_obs = func
        
    # Extend _get_obs function according to obs_types. (This is similar to the Decorator Pattern in OOP)
    for obs_type in obs_types:
        if obs_type == ObservationType.FORK_POSITION:
            _get_obs = _get_obs_forklift_position_decorator(env, _get_obs) # get absolute position of the forklift_robot's links

        elif obs_type == ObservationType.PALLET_POSITION:
            _get_obs = _get_obs_pallet_position_decorator(env, _get_obs) # get absolute position of the pallet model 

        elif obs_type == ObservationType.TARGET_TRANSFORM:
            _get_obs = _get_obs_target_transform_decorator(env, _get_obs) # get initial absolute position of the target/pallet 

        elif obs_type == ObservationType.DEPTH_CAMERA_RAW_IMAGE:
            _get_obs = _get_obs_depth_camera_raw_image_decorator(env, _get_obs)

        elif obs_type == ObservationType.COLLISION_DETECTION:
            _get_obs = _get_obs_collision_detection_decorator(env, _get_obs)
            
        elif obs_type == ObservationType.LATEST_ACTION:
            _get_obs = _get_obs_latest_action_decorator(env, _get_obs)

    # return spaces.Dict(obs_space_dict), _get_obs #TODO: change env._get_obs method being returned (use decorator pattern)
    # return spaces.Box(low = -float("inf") * np.ones((15,)), high = float("inf") * np.ones((15,)), dtype = np.float64), _get_obs #TODO: change env._get_obs method being returned (use decorator pattern)
    return spaces.Dict({
        'observation': spaces.Box(low = -float("inf") * np.ones((6,)), high = float("inf") * np.ones((6,)), dtype = np.float64),
        'achieved_goal': spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float64),
        'desired_goal': spaces.Box(low = -float("inf") * np.ones((2,)), high = float("inf") * np.ones((2,)), dtype = np.float64)
    }), _get_obs



def _get_obs_forklift_position_decorator(env, func):
    def _get_obs_position(env):
        # Obtain a tf observation which belongs to a time after the action was taken
        current_forklift_robot_position_obs = None
        flag = True
        while (current_forklift_robot_position_obs is None) or flag:
            # Obtain forklift_robot frame observation -----
            t = env.get_entity_state_client.get_entity_state(\
                "forklift_bot::base_link::base_link_fixed_joint_lump__chassis_bottom_link_collision_collision", \
                    "world")
            if t is not None and (t.success == True):
                current_forklift_robot_position_obs = t
                flag = False

                # Make sure that observation was obtained after the action was taken at least 'step_duration' time later
                if (int(str(t.header.stamp.sec) \
                    + (str(t.header.stamp.nanosec))) \
                        < (env.ros_clock.nanoseconds + env.config['step_duration'])):
                        flag = True

        obs =  {
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

        # Calculate the angle between the forklift and the target point
        from tf_transformations import euler_from_quaternion
        theta = 0.0 # Current angle of the robot
        # Get forklifts location
        x = obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x,
        y = obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y,
        # Get forklifts orientation
        rot_q = obs['forklift_position_observation']['chassis_bottom_link']['pose']['orientation']
        (roll, pitch, theta) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w]) # convert quaternion to euler angles
        # Get target(goal) location
        from geometry_msgs.msg import Point
        goal = Point()
        goal.x = env._target_transform[0]
        goal.y = env._target_transform[1]
        # Calculate the angle between the forklift and the target
        inc_x = goal.x - x
        inc_y = goal.y - y
        from math import atan2, pi
        angle_to_goal = atan2(inc_y, inc_x) #calculate angle through distance from robot to goal in x and y (Radians)
        total_angle_difference_to_goal = abs(angle_to_goal - theta)
        # print(f'theta(forklifts orientation): {theta}, angle_to_goal: {angle_to_goal}, total_angle_difference_to_goal: {total_angle_difference_to_goal}')
        # Convert to degrees
        total_angle_difference_to_goal_degrees = total_angle_difference_to_goal * 180.0 / pi; 
        print(f'total_angle_difference_to_goal_degrees: {total_angle_difference_to_goal_degrees}')
        # TODO: normalize total_angle_difference_to_goal_degrees by dividing by 180

        # --------------------------------------------
        obs =  {
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
            'total_angle_difference_to_goal_in_degrees': total_angle_difference_to_goal_degrees
        }

        return obs
        

    def f():
        return {
            **(func()),
            **(_get_obs_position(env))
        }

    return f


def _get_obs_pallet_position_decorator(env, func):
    def _get_obs_position(env):
        # Obtain a tf observation which belongs to a time after the action was taken
        current_pallet_position_obs = None
        flag = True
        while (current_pallet_position_obs is None) or flag:
            # Obtain forklift_robot frame observation -----
            t = env.get_entity_state_client.get_entity_state("pallet", "world")
            if t is not None and (t.success == True):
                current_pallet_position_obs = t
                flag = False

                # Make sure that observation was obtained after the action was taken at least 'step_duration' time later
                if (int(str(t.header.stamp.sec) \
                    + (str(t.header.stamp.nanosec))) \
                        < (env.ros_clock.nanoseconds + env.config['step_duration'])):
                        flag = True

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
            **(_get_obs_position(env))
        }

    return f


def _get_obs_target_transform_decorator(env, func):
    def _get_obs_target_transform(env):
        return {
            'target_transform_observation': env._target_transform,
        }
        

    def f():
        return {
            **(func()),
            **(_get_obs_target_transform(env))
        }
        # do NOT reset observations for next iteration because target_transform_observation does not change through an episode

    return f


def _get_obs_depth_camera_raw_image_decorator(env, func):
    def _get_obs_depth_camera_raw_image(env):
        # Depth_camera_raw_image observation -----
        # Check that the observation is from after the action was taken
        current_depth_camera_raw_image_obs = None
        while (current_depth_camera_raw_image_obs is None) or \
            (int(str(current_depth_camera_raw_image_obs["header"].stamp.sec) \
                + (str(current_depth_camera_raw_image_obs["header"].stamp.nanosec))) \
                    < (env.ros_clock.nanoseconds + env.config['step_duration'])): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.  
            # Obtain new depth_camera_raw_image_observation: -----
            rclpy.spin_once(env.depth_camera_raw_image_subscriber)
            current_depth_camera_raw_image_obs = env.depth_camera_img_observation

        depth_camera_raw_image_observation = current_depth_camera_raw_image_obs["image"] # get image
        # --------------------------------------------
        # reset observations for next iteration
        env.depth_camera_img_observation = None 

        return {
            'depth_camera_raw_image_observation': depth_camera_raw_image_observation,
        }


    def f():
        return {
            **(func()),
            **(_get_obs_depth_camera_raw_image(env))
        }

    return f


def _get_obs_collision_detection_decorator(env, func):
    def _get_obs_collision_detection(env):
        """
        Returns collision detection observations that are being published by ros_gazebo_collision_detection_plugin.
        """
        collision_detection_observations = {} # holds finalized up to date collision detection information from all contact sensors
        for subscriber in env.collision_detection_subscribers:
            # Check that the observation is from after the action was taken
            cur_collision_msg = None
            while (cur_collision_msg is None) or \
                (int(str(cur_collision_msg.header.stamp.sec) \
                    + (str(cur_collision_msg.header.stamp.nanosec))) \
                        < (env.ros_clock.nanoseconds + env.config['step_duration'])): # make sure that observation was obtained after the action was taken by at least 'step_duration' time later.  
                # Obtain new collision_detection_observation: -----
                rclpy.spin_once(subscriber)
                if subscriber.link_name in env.collision_detection_states:
                    cur_collision_msg = env.collision_detection_states[subscriber.link_name]

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
        env.collision_detection_states = {} 

        return {
            'collision_detection_observations': collision_detection_observations,
        }


    def f():
        return {
            **(func()),
            **(_get_obs_collision_detection(env))
        }

    return f


def _get_obs_latest_action_decorator(env, func):
    def _get_latest_action(env):
        return {
            'latest_action': env.action
        }
        

    def f():
        return {
            **(func()),
            **(_get_latest_action(env))
        }

    return f

