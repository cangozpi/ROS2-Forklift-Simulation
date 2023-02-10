import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer 

# TODO: don't want to depend on the following two imports in this file
import numpy as np
import torch

# ========== HYPERPARAMTERS ============= TODO: set these in config.yaml
total_episodes = 20 # total number of episodes to train
warmup_steps = 5 # number of steps to take random actions before using the agents policy
update_every = 10 # update model after every update_every steps.
num_updates = 2 # at every update_every steps while updating the model perform num_updates many updates by sampling a new batch for each of them. 
actor_hidden_dims = [400, 300] # holds dimensions of the hidden layers excluding the input layer and the input and the output dimensions for the actor network of ddpg agent.
critic_hidden_dims = [400, 300] # holds dimensions of the hidden layers excluding the input layer and the input and the output dimensions for the critic network of ddpg agent.
epsilon = 1.0 # epsilon used for multiplying the gaussian distribution sample to obtain the noise to add to the agent's action (exploration).
epsilon_decay = 0.001 # every time an action is taken epsilon is reduced by this amount (i.e. epsilon -= epsilon_decay).
gamma = 0.09 # next state discount rate
tau = 0.1 # tau value used in updating the target networks using polyak averaging (i.e. targ_net = tau*targ_net + (1-tau) * net).
replay_buffer_size = 100 # size of the replay buffer
batch_size = 32
# =======================================

def main():
    # Start Env
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0

    # TODO: make this responsive to config.yaml choices
    concatenated_obs_dim = tuple()
    if 'tf' in env.config['observation_types']:
        tf_obs_dim = env.observation_space['forklift_robot_tf_observation']['chassis_bottom_link']['transform'].shape # --> [7,]
        concatenated_obs_dim = (*concatenated_obs_dim, *tf_obs_dim)
    if 'depth_camera_raw_image' in env.config['observation_types']:
        depth_camera_raw_image_obs_dim = 1
        for dim in env.observation_space['depth_camera_raw_image_observation'].shape: # --> [480, 640]
            depth_camera_raw_image_obs_dim *= dim
    concatenated_obs_dim = (*concatenated_obs_dim, depth_camera_raw_image_obs_dim)
    # concatenated_obs_dim = (*tf_obs_dim, *depth_camera_raw_image_obs_dim)

    diff_cont_action_dim = env.action_space['diff_cont_action'].shape
    fork_joint_cont_action_dim = env.action_space['fork_joint_cont_action'].shape
    concatenated_action_dim = (*diff_cont_action_dim, * fork_joint_cont_action_dim)
    # print("HOOOOOOOOOOOOOOOOOOOOOO", env.observation_space, env.action_space, env.action_space.shape, env.observation_space.shape)

    agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        actor_hidden_dims, critic_hidden_dims, epsilon, epsilon_decay, gamma, tau)
    agent.train() # TODO: handle .eval() case for testing the model too.
    replay_buffer = ReplayBuffer(replay_buffer_size, sum(concatenated_obs_dim), sum(concatenated_action_dim), batch_size)

    obs, info = env.reset()
    obs = process_observations(obs)
    while cur_episode < total_episodes: # simulate action taking of an agent for debugging the env
        # For warmup_steps many iterations take random actions to explore better
        if env.cur_iteration < warmup_steps: # take random actions
            action = env.action_space.sample()
            # Take action
            next_obs, reward, done, _, info = env.step(action)
        else: # agent's policy takes action
            obs = torch.unsqueeze(obs, dim=0)
            with torch.no_grad():
                action = agent.choose_action(obs)
            action = torch.squeeze(action, dim=0)
            action = process_action2(action)
            # Take action
            next_obs, reward, done, _, info = env.step(action)

        # Take action
        # next_obs, reward, done, _, info = env.step(action)
        next_obs = process_observations(next_obs)

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if info["iteration"] >= env.max_episode_length:
            term = True
        else:
            term = False
        
        # Store experience in replay buffer
        action = process_action(action) #TODO: refactor this out of this file
        replay_buffer.append(obs, action, reward, next_obs, term) #TODO: create replay_buffer

        # Update current staarray(-0.9692238, dtype=float32)]te
        obs = next_obs

        # Update model if its time
        if (info["iteration"] % update_every == 0) and (info["iteration"] > warmup_steps):
            print(f"Updating the agent with {num_updates} sampled batches.")
            for _ in range(num_updates):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)




        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: {info["agent_location"]}, Target_location: {info["target_location"]}, Reward: {info["reward"]}')

        if done:
            obs, info = env.reset()
            obs = process_observations(obs)
            time.sleep(3)
            cur_episode += 1
            cum_episode_rewards = 0


def process_observations(obs):
    # TODO: refactor this part
    tf_obs = torch.tensor(obs['forklift_robot_tf_observation']['chassis_bottom_link']['transform'])
    depth_camera_raw_image_obs = torch.tensor(obs['depth_camera_raw_image_observation'])
    obs = torch.concat((tf_obs.reshape(-1), depth_camera_raw_image_obs.reshape(-1)), dim=0)
    return obs

def process_action(action):
    # TODO: refactor this part
    diff_cont_act = torch.tensor(action['diff_cont_action'])
    fork_joint_cont_act = torch.tensor(action['fork_joint_cont_action'])
    act = torch.concat((diff_cont_act.reshape(-1), fork_joint_cont_act.reshape(-1)), dim=0)
    return act


def process_action2(action):
    # TODO: refactor this part
    action_dict = {
        "diff_cont_action": action[0:2].cpu().detach().numpy(),
        "fork_joint_cont_action": np.asarray([action[2].cpu().detach().numpy()])
    }
    return action_dict