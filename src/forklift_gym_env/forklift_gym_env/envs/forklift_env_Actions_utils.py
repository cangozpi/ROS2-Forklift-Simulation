from gym import spaces
from forklift_gym_env.envs.utils import ActionType
import numpy as np


def action_space_factory(env, act_types):
    """
    Returns observation space that corresponds to obs_type
    Inputs:
        env (class ForkliftEnv): Reference to ForkliftEnv in which the agent will take the actions implemented here.
        act_type (List(ActionType)): specifies which actions can be taken by the agent.
    """
    for act_type in act_types:
        assert act_type in ActionType

    # Set action space according to act_type. Currently NOT being used, but left as a reference.
    # d =  {
    #         "diff_cont_action": spaces.Box(low = -10 * np.ones((2)), high = 10 * np.ones((2)), dtype=np.float32), #TODO: set this to limits from config file
    #         "fork_joint_cont_action": spaces.Box(low = -2.0, high = 2.0, shape = (1,), dtype = np.float64) # TODO: set its high and low limits correctly
    # }

    # return spaces.Dict(d)

    return spaces.Box(low= -1 * np.ones((2)), high = 1 * np.ones((2)), dtype=np.float32)
    # return spaces.Box(low= -1 * np.ones((1)), high = 1 * np.ones((1)), dtype=np.float32)


