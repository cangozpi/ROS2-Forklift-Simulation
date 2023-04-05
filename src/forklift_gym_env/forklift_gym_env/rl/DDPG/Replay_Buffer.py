import torch
import numpy as np

class ReplayBuffer:
    """
    Hindsight Experience Replay implementation. Refer to https://arxiv.org/pdf/1707.01495.pdf for further information.
    """
    def __init__(self, replay_buffer_size, obs_dim, action_dim, batch_size):
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.index = 0
        self.buffer_full = False

        self.state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.action_buffer = torch.zeros(replay_buffer_size, action_dim, dtype=torch.float32)
        self.reward_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.float32)
        self.next_state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.terminal_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.bool)

        # Arrays used for staging
        self.staged_state = []
        self.staged_action = []
        self.staged_reward = []
        self.staged_next_state = []
        self.staged_term = []


    def append(self, state, action, reward, next_state, term):
        self.state_buffer[self.index, :] = state
        self.action_buffer[self.index, :] = action
        self.reward_buffer[self.index, :] = reward
        self.next_state_buffer[self.index, :] = next_state
        self.terminal_buffer[self.index, :] = term

        if (self.index + 1) >= self.replay_buffer_size:
            self.buffer_full = True
        self.index = (self.index + 1) % self.replay_buffer_size
    

    def clear_staged_for_append(self):
        self.staged_state = []
        self.staged_action = []
        self.staged_reward = []
        self.staged_next_state = []
        self.staged_term = []


    def stage_for_append(self, state, action, reward, next_state, term):
        """
        Stages inputs for the current episode and once it terminates they can be appended to replay buffer using HER by 
        calling the self.commit_append() function.
        Note that goal_state corresponds to the representation of the next_obs as a goal_state for the HER algorithm.
        """
        self.staged_state.append(state.detach().clone())
        self.staged_action.append(action.detach().clone())
        self.staged_reward.append(reward.detach().clone())
        self.staged_next_state.append(next_state.detach().clone())
        self.staged_term.append(term.detach().clone())

    
    def commit_append(self):
        """
        Appends the staged information to replay buffer by using Hindsight Experience Replay (HER).
        """
        # Append original experiences ---
        for i in range(len(self.staged_state)):
            self.append(self.staged_state[i], self.staged_action[i], self.staged_reward[i], \
                self.staged_next_state[i], self.staged_term[i])


    def sample_batch(self):
        if self.buffer_full: # Replay buffer's entries are all filled
            batch_indices = np.random.choice(np.arange(start=0, stop=self.replay_buffer_size), size=self.batch_size)
        elif self.index < self.batch_size:
            raise Exception(f'Can\'t sample a batch since there aren\'t batch_size ({self.batch_size}) many entries in the Replay Buffer yet!')
        else: # Replay Buffer has empty entries after self.index
            batch_indices = np.random.choice(np.arange(start=0, stop=self.index), size=self.batch_size)

        state_batch = self.state_buffer[batch_indices]
        action_batch = self.action_buffer[batch_indices]
        reward_batch = self.reward_buffer[batch_indices]
        next_state_batch = self.next_state_buffer[batch_indices]
        terminal_batch = self.terminal_buffer[batch_indices]
    
        return state_batch.clone(), action_batch.clone(), reward_batch.clone(), next_state_batch.clone(), \
            terminal_batch.clone() # TODO: not sure about keeping .clone() in here
        
    
    def can_sample_a_batch(self):
        """
        Returns True if Replay Buffer has at least batch_size many entires ,else False.
        """
        if self.buffer_full or (self.index >= self.batch_size):
            return True
        else:
            return False