import numpy as np
import torch
import random
from functools import reduce
from forklift_gym_env.envs.utils import ObservationType, ActionType


def flatten_and_concatenate_observation(obs, env):
    obs_flattened = torch.tensor([])
    goal_state = None

    # if ObservationType.TARGET_TRANSFORM in env.obs_types:
    #     target_tf_obs = torch.tensor(obs['target_transform_observation'])
    #     obs_flattened = torch.concat((obs_flattened.reshape(-1), target_tf_obs.reshape(-1)), dim=0)

    if ObservationType.TF in env.obs_types:
        tf_obs = torch.tensor(obs['forklift_robot_tf_observation']['chassis_bottom_link']['transform'])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), tf_obs.reshape(-1)), dim=0)
        
        goal_state = tf_obs = obs['forklift_robot_tf_observation']['chassis_bottom_link']['transform'][:2] # [translation_x, translation_y]

    if ObservationType.DEPTH_CAMERA_RAW_IMAGE in env.obs_types:
        depth_camera_raw_image_obs = torch.tensor(obs['depth_camera_raw_image_observation'])
        obs_flattened = torch.concat((obs_flattened.reshape(-1), depth_camera_raw_image_obs.reshape(-1)), dim=0)

    return obs_flattened, goal_state, obs


def flatten_and_concatenate_action(action):
    diff_cont_act = torch.tensor(action['diff_cont_action'])
    fork_joint_cont_act = torch.tensor(action['fork_joint_cont_action'])
    act = torch.concat((diff_cont_act.reshape(-1), fork_joint_cont_act.reshape(-1)), dim=0)
    return act


def convert_agent_action_to_dict(action, env):
    """
    scales [-1,1] range predicted actions to [env.action_space['...'].high, env.action_space['...'].high] range, and 
    sets actions which are not predicted by the agent with zeros.
    """
    action_dict = {}
    if ActionType.DIFF_CONT in env.act_types:
        action_dict = {
            **action_dict,
            **{
                "diff_cont_action": action[0:2].cpu().detach().numpy() * abs(env.action_space['diff_cont_action'].high)
            }
        }
    else:
        action_dict = {
            **action_dict,
            **{
                "diff_cont_action": np.zeros((2)) # take no action
            }
        }


    if ActionType.FORK_JOINT_CONT in env.act_types:
        action_dict = {
            **action_dict,
            **{
                "fork_joint_cont_action": np.asarray([action[2].cpu().detach().numpy()]) * abs(env.action_space['fork_joint_cont_action'].high)
            }
        }
    else:
        action_dict = {
            **action_dict,
            **{
                "fork_joint_cont_action": np.zeros((1)) # take no action
            }
        }
    
    return action_dict


def get_concatenated_obs_and_act_dims(env):
    concatenated_obs_dim = 0
    goal_state_dim = 0
    if ObservationType.TARGET_TRANSFORM in env.obs_types: # This functions as the goal state
        target_tf_obs_dim = env.observation_space['target_transform_observation'].shape # --> [2,]
        goal_state_dim += reduce(lambda a,b: a * b, target_tf_obs_dim)
    if ObservationType.TF in env.obs_types:
        tf_obs_dim = env.observation_space['forklift_robot_tf_observation']['chassis_bottom_link']['transform'].shape # --> [7,]
        concatenated_obs_dim += reduce(lambda a,b: a * b, tf_obs_dim)
    if ObservationType.DEPTH_CAMERA_RAW_IMAGE in env.obs_types:
        depth_camera_raw_img_obs_dim =  env.observation_space['depth_camera_raw_image_observation'].shape # --> [480, 640]
        concatenated_obs_dim += reduce(lambda a,b: a * b, depth_camera_raw_img_obs_dim)

    concatenated_action_dim = 0
    diff_cont_action_dim = env.action_space['diff_cont_action'].shape
    concatenated_action_dim += reduce(lambda a,b: a * b, diff_cont_action_dim)
    fork_joint_cont_action_dim = env.action_space['fork_joint_cont_action'].shape
    concatenated_action_dim += reduce(lambda a,b: a * b, fork_joint_cont_action_dim)
    
    return concatenated_obs_dim, concatenated_action_dim, goal_state_dim


def seed_everything(seed):
    """
    Set seed to random, numpy, torch, gym environment
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)