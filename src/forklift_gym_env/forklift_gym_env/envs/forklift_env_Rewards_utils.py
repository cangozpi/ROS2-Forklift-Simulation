from forklift_gym_env.envs.utils import RewardType
import numpy as np


# -------------------------------------------------------------------------------------------
# ---------------------------------------------- REWARD_FACTORY: 
def calculate_reward_factory(env, reward_types):
    """
    Depending on the env.use_GoalEnv (bool), returns a function which calculates reward which corresponds to 
    the given reward_type. This way ForkliftEnv can support both the gym.Env and gym.GoalEnv.
    Inputs:
        env (class ForkliftEnv): Reference to ForkliftEnv in which the agent will receive rewards which are implemented here.
        reward_type (List[RewardType]): specifies which reward function is used.
    """
    for reward_type in reward_types:
        assert reward_type in RewardType

    # Return corresponding reward calculation function by accumulating the rewards returned by the functions specified in the reward_types
    if env.use_GoalEnv: # Support gym.GoalEnv(HER) and return {'observation', 'desired_goal', 'achieved_goal'}
        def reward_func(achieved_goal, desired_goal, info):
            reward = 0
            if RewardType.L2_DIST in reward_types:
                reward += calc_reward_L2_dist(env, achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)

            if RewardType.NAV_REWARD in reward_types:
                reward += calc_navigation_reward(env, achieved_goal=achieved_goal, desired_goal=desired_goal, info=info)
    
            if RewardType.COLLISION_PENALTY in reward_types:
                reward += calc_reward_collision_penalty(env, achieved_goal=achieved_goal, desired_goal=desired_goal, info=info) #TODO: this should be updated to accept achieved_goal, desired_goal, info
            
            if RewardType.BINARY in reward_types:
                reward += calc_reward_binary(env, achieved_goal=achieved_goal, desired_goal=desired_goal)

            return reward

    else: # Support regular gym.Env and return 'observation' tensor
        def reward_func(observation):
            reward = 0
            if RewardType.L2_DIST in reward_types:
                reward += calc_reward_L2_dist(env, observation=observation)

            if RewardType.NAV_REWARD in reward_types:
                reward += calc_navigation_reward(env, observation=observation)
    
            if RewardType.COLLISION_PENALTY in reward_types:
                reward += calc_reward_collision_penalty(env, observation=observation) #TODO: this should be updated to accept achieved_goal, desired_goal, info
            
            if RewardType.BINARY in reward_types:
                reward += calc_reward_binary(env, observation=observation)

            return reward


    return reward_func




def calc_reward_L2_dist(env, achieved_goal=None, desired_goal=None, info=None, observation=None):
# def calc_reward_L2_dist(observation, goal_state):
    """
    Returns the negative L2 distance between the (translation_x, translation_y) coordinates 
    of forklift_robot_transform and target_transform.
    Inputs:
        observation: returned by env._get_obs()
    Returns:
        reward: negative L2 distance

    """
    # Return negative L2 distance btw chassis_bottom_link and the target location as reward
    # if env.use_GoalEnv:
    #     l2_dist = np.linalg.norm(achieved_goal - desired_goal)
    # else:
    #     robot_transform_translation = [forklift_robot_transform['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x, \
    #         forklift_robot_transform['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y] # [translation_x, translation_y]
    #     l2_dist = np.linalg.norm(robot_transform_translation - env._target_transform)
    # return - l2_dist

    # Return angle penalty as reward
    # if env.use_GoalEnv:
    #     obs = info['observation']
    #     total_angle_differencee_to_goal_in_degrees = info['observation']['total_angle_difference_to_goal_in_degrees']
    # else:
    #     total_angle_differencee_to_goal_in_degrees = observation['total_angle_difference_to_goal_in_degrees']
    # return -(total_angle_differencee_to_goal_in_degrees **2)

    # Return turning clockwise penalty as reward
    if env.use_GoalEnv:
        action = info["observation"]["latest_action"][0] # penalize turning right/negative angular action
        return action
    else:
        action = observation["latest_action"][0] # penalize turning right/negative angular action
        return action


def calc_navigation_reward(env, achieved_goal=None, desired_goal=None, info=None, observation=None):
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
    if env.use_GoalEnv:
        goal_achieved = env.check_goal_achieved(achieved_goal, desired_goal, full_obs=False)
        if goal_achieved:
            return 100.0
        else:
            action = info["observation"]["latest_action"]
            # Extract linear.x action
            linearX_velocity = action[0]
            # Extract angular.z action
            angularZ_velocity = action[1]
            return abs(linearX_velocity) / 2 - abs(angularZ_velocity) / 2
    else:
        ach_goal = [observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x, \
            observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y]
        des_goal = env._target_transform
        goal_achieved = env.check_goal_achieved(ach_goal, des_goal, full_obs=False)
        if goal_achieved:
            return 100.0
        else:
            action = observation["latest_action"]
            # Extract linear.x action
            linearX_velocity = action[0]
            # Extract angular.z action
            angularZ_velocity = action[1]
            return abs(linearX_velocity) / 2 - abs(angularZ_velocity) / 2
    

def calc_reward_collision_penalty(env, achieved_goal=None, desired_goal=None, info=None, observation=None): # TODO: update this function it won't work as it is
    """
    Returns a Reward which penalizes agent's collision with other (non-ground) objects.

    """
    if env.use_GoalEnv:
        penalty_per_collision = -5 # TODO: fine-tune this value
        collision_detection_observations = info['observation']['collision_detection_observations']
        # Check for agents collisions with non-ground objects
        unique_non_ground_contacts = {} # holds ContactsState msgs for the collisions with objects that are not the ground
        for state in collision_detection_observations.values():
            unique_non_ground_contacts = {**unique_non_ground_contacts, **CollisionDetectionSubscriber.get_non_ground_collisions(contactsState_msg = state)}
        return penalty_per_collision * len(unique_non_ground_contacts)
    else:
        penalty_per_collision = -5 # TODO: fine-tune this value
        collision_detection_observations = observation['collision_detection_observations']
        # Check for agents collisions with non-ground objects
        unique_non_ground_contacts = {} # holds ContactsState msgs for the collisions with objects that are not the ground
        for state in collision_detection_observations.values():
            unique_non_ground_contacts = {**unique_non_ground_contacts, **CollisionDetectionSubscriber.get_non_ground_collisions(contactsState_msg = state)}
        return penalty_per_collision * len(unique_non_ground_contacts)


def calc_reward_binary(env, achieved_goal=None, desired_goal=None, info=None, observation=None):
    """
    Returns 1 if goal_achieved else 0.
    Inputs:
        observation: returned by env._get_obs()
    Returns:
        reward: 1 or 0 (binary reward).
    """
    if env.use_GoalEnv:
        return int(env.check_goal_achieved(observation, goal_state, full_obs=False))
    else:
        ach_goal = [observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].x, \
            observation['forklift_position_observation']['chassis_bottom_link']['pose']['position'].y]
        des_goal = env._target_transform
        return int(env.check_goal_achieved(ach_goal, des_goal, full_obs=False))


# ------------------------------------------------------------------------------------------------------------ 
# ------------------------------------------------------------------------------------------------------------ 