import pytest
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer

@pytest.fixture
def replay_buffer():
    replay_buffer_size = 10
    obs_dim = 5
    action_dim = 2
    batch_size = 3
    return ReplayBuffer(replay_buffer_size, obs_dim, action_dim, batch_size)

@pytest.mark.replay_buffer
def test_should_pass():
    assert True == True

@pytest.mark.replay_buffer
def test_replay_buffer_store(replay_buffer):
    assert replay_buffer != None