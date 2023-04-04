from forklift_gym_env.envs.utils import RewardType
import numpy as np


def calculate_reward_factory(env, reward_types):
    """
    Returns a function that calculates reward which corresponds to the given reward_type
    Inputs:
        env (class ForkliftEnv): Reference to ForkliftEnv in which the agent will receive rewards which are implemented here.
        reward_type (List[RewardType]): specifies which reward function is used.
    """
    for reward_type in reward_types:
        assert reward_type in RewardType

    # Return corresponding reward calculation function by accumulating the rewards returned by the functions specified in the reward_types
    def reward_func(achieved_goal, desired_goal, info):

        reward = 0

        if RewardType.L2_DIST in reward_types:
            reward += calc_reward_L2_dist(env, achieved_goal, desired_goal, info)

        if RewardType.NAV_REWARD in reward_types:
            reward += calc_navigation_reward(env, achieved_goal, desired_goal, info)
    
        if RewardType.COLLISION_PENALTY in reward_types:
            reward += calc_reward_collision_penalty(env, observation) #TODO: this should be updated to accept achieved_goal, desired_goal, info
            
        if RewardType.BINARY in reward_types:
            reward += calc_reward_binary(env, observation, goal_state)

        return reward

    return reward_func




def calc_reward_L2_dist(env, achieved_goal, desired_goal, info):
# def calc_reward_L2_dist(observation, goal_state):
    """
    Returns the negative L2 distance between the (translation_x, translation_y) coordinates 
    of forklift_robot_transform and target_transform.
    Inputs:
        observation: returned by env._get_obs()
    Returns:
        reward: negative L2 distance

    """
    # forklift_robot_transform = observation['forklift_position_observation']
    # Return negative L2 distance btw chassis_bottom_link and the target location as reward
    # Note that we are only using translation here, NOT using rotation information
    # robot_transform_translation = [forklift_robot_transform['chassis_bottom_link']['pose']['position'].x, \
    #     forklift_robot_transform['chassis_bottom_link']['pose']['position'].y] # [translation_x, translation_y]
    l2_dist = np.linalg.norm(achieved_goal - desired_goal)
    # return - l2_dist
    # angle penalty
    reward = 0
    obs = info['observation']
    reward += -(obs['total_angle_difference_to_goal_in_degrees'] **2)

    # uncomment below to debug env/AI_agent
    # action = info["action"]
    # reward = action[0] # penalize turning right/negative angular action

    return reward


def calc_navigation_reward(env, achieved_goal, desired_goal, info):
    """
    Returns a high reward for achieving the goal_state, if not returns a reward that favors linear.x velocity (forward movement)
    but penalizes angular.z velocity (turning).
    Refer to https://medium.com/@reinis_86651/deep-reinforcement-learning-in-mobile-robot-navigation-tutorial-part4-environment-7e4bc672f590
    and its definition of a reward function for further details.
    Inputs:
        achieved_goal: state that the robot is in.
        desired_goal: desired goal state (state of the goal).
        info: bears further information that can be used in reward calculation.
    Returns:
        reward 

    """
    action = info["action"]
    # 
    goal_achieved = env.check_goal_achieved(achieved_goal, desired_goal, full_obs=False)
    if goal_achieved:
        return 100.0
    else:
        # Extract linear.x action
        linearX_velocity = action[0]
        # Extract angular.z action
        angularZ_velocity = action[1]
        print(f'linearx: {linearX_velocity}, angularZ: {angularZ_velocity}')
        print("---")
                        
        return abs(linearX_velocity) / 2 - abs(angularZ_velocity) / 2
    

def calc_reward_collision_penalty(env, observation):
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


def calc_reward_binary(env, observation, goal_state):
    """
    Returns 1 if goal_achieved else 0.
    Inputs:
        observation: returned by env._get_obs()
    Returns:
        reward: 1 or 0 (binary reward).
    """
    return int(env.check_goal_achieved(observation, goal_state, full_obs=False))