import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, replay_buffer_size, obs_dim, action_dim, batch_size):
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.index = 0

        self.state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.action_buffer = torch.zeros(replay_buffer_size, action_dim, dtype=torch.float32)
        self.reward_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.float32)
        self.next_state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.terminal_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.bool)


    def append(self, obs, action, reward, next_obs, term):
        self.state_buffer[index, :] = obs
        self.action_buffer[index, :] = action
        self.reward_buffer[index, :] = reward
        self.next_state_buffer[index, :] = next_obs
        self.terminal_buffer[index, :] = term

        self.index = (self.index + 1) % self.replay_buffer_size


    def sample_batch(self):
        batch_indices = np.random.choice(np.arange(start=0, stop=self.replay_buffer_size), size=self.batch_size)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        terminal_batch = self.terminal_buffer[batch_indices]

        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch
        