import pytest
from forklift_gym_env.envs.Forklift_env import ForkliftEnv
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer
import torch
import numpy as np
from copy import deepcopy

def get_ForkliftEnv():
    """
    Uses singleton pattern and returns an instance of ForkliftEnv (of type Gym Env)
    """
    if get_ForkliftEnv.env == None:
        # Read in parameters from config.yaml
        config_path = 'build/forklift_gym_env/forklift_gym_env/test/config_pytest_forklift_env.yaml'
        # Start Env
        get_ForkliftEnv.env = ForkliftEnv(config_path=config_path, use_GoalEnv=False)
    return get_ForkliftEnv.env
get_ForkliftEnv.env = None # static function variable

@pytest.fixture
def initialize_replay_buffer():
    env = get_ForkliftEnv()
    replay_buffer_size = env.config['replay_buffer_size']
    obs_dim = np.prod(env.observation_space['observation'].shape)
    action_dim = np.prod(env.action_space.shape)
    batch_size = env.config['batch_size']
    return ReplayBuffer(replay_buffer_size, obs_dim, action_dim, batch_size), {
        'replay_buffer_size': replay_buffer_size,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'batch_size': batch_size
    }

@pytest.mark.replay_buffer
def test_replay_buffer_initialization(initialize_replay_buffer):
    replay_buffer, replay_buffer_params = initialize_replay_buffer

    assert replay_buffer.replay_buffer_size == replay_buffer_params['replay_buffer_size']
    assert replay_buffer.batch_size == replay_buffer_params['batch_size']
    assert replay_buffer.index == 0 
    assert replay_buffer.buffer_full == False

    assert replay_buffer.state_buffer.shape == (replay_buffer_params['replay_buffer_size'], replay_buffer_params['obs_dim'])
    assert replay_buffer.action_buffer.shape == (replay_buffer_params['replay_buffer_size'], replay_buffer_params['action_dim'])
    assert replay_buffer.reward_buffer.shape == (replay_buffer_params['replay_buffer_size'], 1)
    assert replay_buffer.next_state_buffer.shape == (replay_buffer_params['replay_buffer_size'], replay_buffer_params['obs_dim'])
    assert replay_buffer.terminal_buffer.shape == (replay_buffer_params['replay_buffer_size'], 1)

@pytest.mark.replay_buffer
def test_replay_buffer_append(initialize_replay_buffer):
    env = get_ForkliftEnv()
    replay_buffer, replay_buffer_params = initialize_replay_buffer

    # records ground truth values
    gt_staged_state = []
    gt_staged_action = []
    gt_staged_reward = []
    gt_staged_next_state = []
    gt_staged_term = []

    cur_episode = 0
    cur_iteration = 0

    obs_dict = env.reset()
    obs = torch.tensor(obs_dict['observation']).float()
    while cur_episode < env.config['total_episodes']: 
        cur_iteration += 1 
        action = env.action_space.sample()
        next_obs_dict, reward, done, info = env.step(action)
        next_obs = torch.tensor(next_obs_dict['observation']).float()

        if done and cur_iteration < env.max_episode_length:
            term = True
        else:
            term = False

        replay_buffer.stage_for_append(obs, torch.tensor(action), torch.tensor(reward), next_obs, torch.tensor(term))

        gt_staged_state.append(deepcopy(obs))
        gt_staged_next_state.append(deepcopy(next_obs))
        gt_staged_action.append(deepcopy(torch.tensor(action)))
        gt_staged_reward.append(deepcopy(torch.tensor(reward)))
        gt_staged_term.append(deepcopy(torch.tensor(term)))

        # Test that staged variables are not stored by reference in the ReplayBuffer
        index = len(replay_buffer.staged_state) - 1
        assert (id(obs) != id(replay_buffer.staged_state[index])) and torch.allclose(obs, replay_buffer.staged_state[index])
        assert (id(action) != id(replay_buffer.staged_action[index])) and torch.allclose(torch.tensor(action), replay_buffer.staged_action[index])
        assert (id(reward) != id(replay_buffer.staged_reward[index])) and torch.allclose(torch.tensor(reward), replay_buffer.staged_reward[index])
        assert (id(next_obs) != id(replay_buffer.staged_next_state[index])) and torch.allclose(next_obs, replay_buffer.staged_next_state[index])
        assert (id(term) != id(replay_buffer.staged_term[index])) and torch.allclose(torch.tensor(term), replay_buffer.staged_term[index])

        if done:
            # Commit experiences to replay_buffer
            replay_buffer.commit_append()
            replay_buffer.clear_staged_for_append()
            # Test that staged buffers are cleared upon calling clear_staged_for_append()
            assert (len(replay_buffer.staged_state) == 0 and len(replay_buffer.staged_action) == 0 and \
                len(replay_buffer.staged_reward) == 0 and len(replay_buffer.staged_next_state) == 0 and \
                    len(replay_buffer.staged_term) == 0)

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs_dict['observation']).float()

            cur_episode += 1
            cur_iteration = 0
    
    # Test that Replay Buffer contains all of the experiences so far (i.e. replay_buffer.commit_append() works)
    if replay_buffer.buffer_full == False:
        for i in range(-1, -(len(gt_staged_state) + 1), -1): # i = [-1, ..., -lenght_of_array]
            # i = i % -replay_buffer.replay_buffer_size
            j = replay_buffer.index + i
            assert torch.allclose(gt_staged_state[i], replay_buffer.state_buffer[j])
            assert torch.allclose(gt_staged_action[i], replay_buffer.action_buffer[j])
            assert torch.allclose(gt_staged_reward[i].type(torch.float32), replay_buffer.reward_buffer[j])
            assert torch.allclose(gt_staged_next_state[i], replay_buffer.next_state_buffer[j])
            assert torch.allclose(gt_staged_term[i], replay_buffer.terminal_buffer[j])
    
@pytest.mark.replay_buffer
def test_check_sample_batch(initialize_replay_buffer):
    env = get_ForkliftEnv()
    replay_buffer, replay_buffer_params = initialize_replay_buffer

    cur_episode = 0
    cur_iteration = 0

    obs_dict = env.reset()
    obs = torch.tensor(obs_dict['observation']).float()
    while replay_buffer.can_sample_a_batch() != True: # keep collecting experience until you can sample a batch
        cur_iteration += 1 
        action = env.action_space.sample()
        next_obs_dict, reward, done, info = env.step(action)
        next_obs = torch.tensor(next_obs_dict['observation']).float()

        if done and cur_iteration < env.max_episode_length:
            term = True
        else:
            term = False

        replay_buffer.stage_for_append(obs, torch.tensor(action), torch.tensor(reward), next_obs, torch.tensor(term))

        if done:
            # Commit experiences to replay_buffer
            replay_buffer.commit_append()
            replay_buffer.clear_staged_for_append()

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs_dict['observation']).float()

            cur_episode += 1
            cur_iteration = 0
    
    # Test replay_buffer.sample_batch()
    state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch()
    assert state_batch.shape == (replay_buffer_params['batch_size'], replay_buffer_params['obs_dim'])
    assert action_batch.shape == (replay_buffer_params['batch_size'], replay_buffer_params['action_dim'])
    assert reward_batch.shape == (replay_buffer_params['batch_size'], 1)
    assert next_state_batch.shape == (replay_buffer_params['batch_size'], replay_buffer_params['obs_dim'])
    assert terminal_batch.shape == (replay_buffer_params['batch_size'], 1)
