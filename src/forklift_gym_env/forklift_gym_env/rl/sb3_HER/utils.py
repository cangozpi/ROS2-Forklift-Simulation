import numpy as np
import torch
import random
from functools import reduce
from forklift_gym_env.envs.utils import ObservationType, ActionType


def flatten_and_concatenate_observation(obs, env):
    obs_flattened = torch.tensor([])
    achieved_state = None
    goal_state = None


    if ObservationType.TARGET_TRANSFORM in env.obs_types:
        # target_tf_obs = torch.tensor(obs['target_transform_observation']) # [translation_x, translation_y] of the initial absolute position of the target
        # obs_flattened = torch.concat((obs_flattened.reshape(-1), target_tf_obs.reshape(-1)), dim=0)

        # Goal state for HER buffer is [translation_x, translation_y] of the target_transform (pallet)
        target_tf_obs = np.array(obs['target_transform_observation']) # [translation_x, translation_y] of the initial absolute position of the target (pallet)
        goal_state = target_tf_obs

    if ObservationType.FORK_POSITION in env.obs_types:
        # achieved_state of the agent
        achieved_state = torch.tensor([
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x, 
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y, 
        ])

        tf_obs = torch.tensor([
            # Add Pose/transformations:
            # obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x, 
            # obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y, 
            # obs['forklift_position_observation']['chassis_bottom_link']['pose']['position'].z,
            # Add Pose/rotations:
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['orientation'].x,
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['orientation'].y, 
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['orientation'].z, 
            obs['forklift_position_observation']['chassis_bottom_link']['pose']['orientation'].w, 
            
            # Add Twist/linear
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['linear'].x, 
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['linear'].y, 
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['linear'].z,
            # Add Twist/angular
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['angular'].x, 
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['angular'].y, 
            # obs['forklift_position_observation']['chassis_bottom_link']['twist']['angular'].z,

            # Add angle difference between the forklifts face and the target location
            obs['total_angle_difference_to_goal_in_degrees'],
            ])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), tf_obs.reshape(-1)), dim=0)

    if ObservationType.PALLET_POSITION in env.obs_types:
        tf_obs = torch.tensor([
            # Add transformations:
            obs['pallet_position_observation']['pallet_model']['pose']['position'].x, 
            obs['pallet_position_observation']['pallet_model']['pose']['position'].y, 
            obs['pallet_position_observation']['pallet_model']['pose']['position'].z,
            # Add rotations:
            obs['pallet_position_observation']['pallet_model']['pose']['orientation'].x,
            obs['pallet_position_observation']['pallet_model']['pose']['orientation'].y, 
            obs['pallet_position_observation']['pallet_model']['pose']['orientation'].z, 
            obs['pallet_position_observation']['pallet_model']['pose']['orientation'].w, 
            ])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), tf_obs.reshape(-1)), dim=0)

    if ObservationType.LATEST_ACTION in env.obs_types:
        tf_obs = torch.tensor(obs['latest_action'])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), tf_obs.reshape(-1)), dim=0)

    # TODO: collision detection case is missing here

    if ObservationType.DEPTH_CAMERA_RAW_IMAGE in env.obs_types:
        depth_camera_raw_image_obs = torch.tensor(obs['depth_camera_raw_image_observation'])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), depth_camera_raw_image_obs.reshape(-1)), dim=0)

    return obs_flattened, achieved_state, goal_state, obs


def seed_everything(seed):
    """
    Set seed to random, numpy, torch, gym environment
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)