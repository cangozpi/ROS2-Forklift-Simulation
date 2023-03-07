import torch
import numpy as np

class HER_ReplayBuffer:
    """
    Hindsight Experience Replay implementation. Refer to https://arxiv.org/pdf/1707.01495.pdf for further information.
    """
    def __init__(self, replay_buffer_size, obs_dim, action_dim, goal_dim, batch_size):
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.index = 0
        self.buffer_full = False

        self.state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.action_buffer = torch.zeros(replay_buffer_size, action_dim, dtype=torch.float32)
        self.reward_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.float32)
        self.next_state_buffer = torch.zeros(replay_buffer_size, obs_dim, dtype=torch.float32)
        self.terminal_buffer = torch.zeros(replay_buffer_size, 1, dtype=torch.bool)
        self.desired_goal_buffer = torch.zeros(replay_buffer_size, goal_dim, dtype=torch.float32)
        self.achieved_goal_buffer = torch.zeros(replay_buffer_size, goal_dim, dtype=torch.float32)

        # Arrays used for staging
        self.staged_state = []
        self.staged_action = []
        self.staged_reward = []
        self.staged_next_state = []
        self.staged_term = []
        self.staged_desired_goal = []
        self.staged_achieved_goal = []


    def append(self, state, action, reward, next_state, term, desired_goal, achieved_goal): #TODO: make sure inputs are torch Tensors and numpy objects
        self.state_buffer[self.index, :] = state.clone()
        self.action_buffer[self.index, :] = action.clone()
        self.reward_buffer[self.index, :] = reward.clone()
        self.next_state_buffer[self.index, :] = next_state.clone()
        self.terminal_buffer[self.index, :] = term.clone()
        self.desired_goal_buffer[self.index, :] = desired_goal.clone()
        self.achieved_goal_buffer[self.index, :] = achieved_goal.clone()

        if (self.index + 1) >= self.replay_buffer_size:
            self.buffer_full = True
        self.index = (self.index + 1) % self.replay_buffer_size
    

    def clear_staged_for_append(self):
        self.staged_state = []
        self.staged_action = []
        self.staged_reward = []
        self.staged_next_state = []
        self.staged_term = []
        self.staged_desired_goal = []
        self.staged_achieved_goal = []

        self.staged_index = 0


    def stage_for_append(self, state, action, reward, next_state, term, desired_goal, achieved_goal):
        """
        Stages inputs for the current episode and once it terminates they can be appended to replay buffer using HER by 
        calling the self.commit_append() function.
        Note that goal_state corresponds to the representation of the next_obs as a goal_state for the HER algorithm.
        """
        self.staged_state.append(state)
        self.staged_action.append(action)
        self.staged_reward.append(reward)
        self.staged_next_state.append(next_state)
        self.staged_term.append(term)
        self.staged_desired_goal.append(desired_goal)
        self.staged_achieved_goal.append(achieved_goal)

        self.staged_index += 1

    
    def commit_append(self, k, calc_reward_func, check_goal_achieved_func):
        """
        Appends the staged information to replay buffer by using Hindsight Experience Replay (HER).
        """
        # Append original experiences ---
        for i in range(self.staged_index):
            self.append(self.staged_state[i], self.staged_action[i], self.staged_reward[i], \
                self.staged_next_state[i], self.staged_term[i], self.staged_desired_goal[i], self.staged_achieved_goal[i])

        # Append HER experiences  ---
        # Generate k many goals from the last states of the current episode
        for cur_k in range(k):
            cur_goal_obs_index = self.staged_index - 1 - cur_k
            if cur_goal_obs_index < 0:
                print(f'Not enough staged in Replay buffer. Skipping goal generation for k >= {cur_k} after !')
                continue
            cur_achieved_goal = self.staged_achieved_goal[cur_goal_obs_index] 
            for i in range(len(self.staged_state)):
                # Obtain reward for the current goal
                cur_HER_goal_reward = calc_reward_func(self.staged_achieved_goal[i], cur_achieved_goal)
                # Check if terminal state is reached wrt cur_goal
                term = check_goal_achieved_func(self.staged_achieved_goal[i], cur_achieved_goal)
                # Append current HER experiences to the buffer
                self.append(self.staged_state[i], self.staged_action[i], torch.tensor(cur_HER_goal_reward), \
                    self.staged_next_state[i], torch.tensor(term), self.staged_achieved_goal[i], cur_achieved_goal)


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
        desired_goal_batch = self.desired_goal_buffer[batch_indices]
        achieved_goal_batch = self.achieved_goal_buffer[batch_indices]
    
        return state_batch.clone(), action_batch.clone(), reward_batch.clone(), next_state_batch.clone(), \
            terminal_batch.clone(), desired_goal_batch.clone(), achieved_goal_batch.clone() # TODO: not sure about keeping .clone() in here
        
    
    def can_sample_a_batch(self):
        """
        Returns True if Replay Buffer has at least batch_size many entires ,else False.
        """
        if self.buffer_full or (self.index >= self.batch_size):
            return True
        else:
            return False