import pytest
from forklift_gym_env.envs.Forklift_env import ForkliftEnv
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer
import torch
import numpy as np
from copy import deepcopy
import time
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.utils import *
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import datetime

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

    
# TODO: add unit tests to check ForkliftEnv
# @pytest.mark.forklift_env
# def test_XXX():
#     env = get_ForkliftEnv()

#     # Initialize Tensorboard
#     log_dir, run_name = "logs_tensorboard_pytest/", "ForkliftEnv DDPG agent training_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     tb_summaryWriter = SummaryWriter(log_dir + run_name)
#     # seed_everything(env.config["seed"]) # set seed
#     # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
#     cur_episode = 0
#     cur_iteration = 0
#     cur_iteration_btw_updates = 0
#     cum_episode_rewards = 0
#     rewards = deque(maxlen=100)
#     cur_num_updates = 0 # total number of updates including all the episodes

#     concatenated_obs_dim = np.prod(env.observation_space['observation'].shape)
#     concatenated_action_dim = np.prod(env.action_space.shape)

#     agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
#         env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["critic_lr"]), \
#             env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
#                 max_action=torch.tensor(env.action_space.high).float(), policy_update_delay=env.config['policy_update_delay'], logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
#     agent.train_mode()
#     replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, env.config["batch_size"])

#     obs_dict = env.reset()
#     obs = torch.tensor(obs_dict['observation']).float()

#     replay_buffer.clear_staged_for_append()
#     num_warmup_steps_taken = 0
#     while cur_episode < env.config["total_episodes"]: 
#         cur_iteration += 1
#         cur_iteration_btw_updates += 1
#         # For warmup_steps many iterations take random actions to explore better
#         if num_warmup_steps_taken < env.config["warmup_steps"]: # take random actions
#             action = env.action_space.sample()
#             num_warmup_steps_taken += 1
#         else: # agent's policy takes action
#             with torch.no_grad():
#                 action = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
#                 action = torch.squeeze(action, dim=0).numpy()

#         # Take action
#         next_obs_dict, reward, done, info = env.step(np.copy(action))
#         next_obs = torch.tensor(next_obs_dict['observation']).float()

#         cum_episode_rewards += reward

#         # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
#         if done and cur_iteration < env.max_episode_length:
#             term = True
#         else:
#             term = False
        
#         # Stage current (s,a,s') to replay buffer as to be appended at the end of the current episode
#         replay_buffer.stage_for_append(obs, torch.tensor(action), torch.tensor(reward), next_obs, torch.tensor(term))

#         # Update current state
#         obs = next_obs

#         if env.config["verbose"]:
#             # print(f'Episode: {cur_episode}, Iteration: {cur_iteration}/{info["max_episode_length"]},', 
#             # f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}),', \
#             #     f'Action: {info["observation"]["latest_action"].tolist()}, Reward: {reward:.3f}')
#             print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')


#         if done:
#             mean_ep_reward = cum_episode_rewards/cur_iteration
#             rewards.append(cum_episode_rewards)
#             # Log to Tensorboard
#             tb_summaryWriter.add_scalar("Training Reward/[per episode]", mean_ep_reward, cur_episode)
#             tb_summaryWriter.add_scalar("Training Reward/[ep_rew_mean]", np.mean(rewards), cur_episode)
#             tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

#             # Commit experiences to replay_buffer
#             replay_buffer.commit_append()
#             replay_buffer.clear_staged_for_append()

#             # Reset env
#             obs_dict = env.reset()
#             # Concatenate observation with goal_state for regular DDPG agent
#             obs = torch.tensor(obs_dict['observation']).float()

#             time.sleep(3)
#             # Reset episode parameters for a new episode
#             cur_episode += 1
#             cum_episode_rewards = 0
#             cur_iteration = 0


#         # Update model if its time
#         if (cur_iteration_btw_updates % env.config["update_every"]== 0) and (num_warmup_steps_taken >= env.config["warmup_steps"]) and replay_buffer.can_sample_a_batch():
#             print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
#             for _ in range(env.config["num_updates"]):
#                 state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
#                 agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
#                 cur_num_updates += 1

#                 # Save the model
#                 if ((cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 )) or \
#                     (cur_episode + 1 == env.config["total_episodes"]) :
#                     print("Saving the model ...")
#                     agent.save_model()
                
#             cur_iteration_btw_updates = 0
