import gym
import time
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer
from forklift_gym_env.rl.DDPG.utils import *

from forklift_gym_env.envs.Forklift_env import ForkliftEnv

from torch.utils.tensorboard import SummaryWriter
from collections import deque
import datetime

def main():
    # Read in parameters from config.yaml
    config_path = 'build/forklift_gym_env/forklift_gym_env/config/config_DDPG_forklift_env.yaml'

    # Start Env
    # env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    env = ForkliftEnv(config_path=config_path, use_GoalEnv=False)
    env.seed(42) # TODO: not sure if this works if not make this work

    mode = env.config["mode"]
    if mode == "train": # train agent
        train_agent(env)
    elif mode == "test": # test pre-trained agent
        test_agent(env)
    else:
        raise Exception('\'mode\' must be either [\'train\', \'test\']')


def train_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "ForkliftEnv DDPG agent training_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cur_iteration = 0
    cur_iteration_btw_updates = 0
    cum_episode_rewards = 0
    rewards = deque(maxlen=100)
    cur_num_updates = 0 # total number of updates including all the episodes

    concatenated_obs_dim = sum(env.observation_space['observation'].shape) # TODO: ochange sum() with np.prod()
    concatenated_action_dim = sum(env.action_space.shape)

    agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["critic_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space.high).float(), policy_update_delay=env.config['policy_update_delay'], logger=tb_summaryWriter, log_full_detail=env.config['log_full_detail'])
    agent.train_mode()

    # Fix model parameters to a good initialization (Proportional Controller)
    agent.actor.model_layers[0].weight.data[:] = torch.tensor([[0.0, 0.3], [1.5, 0.00]]) # will correspond to [-0.01*theta, -0.1*l2_dist]
    agent.actor.model_layers[0].bias.data[:] = torch.zeros_like(agent.actor.model_layers[0].bias.data)

    replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, env.config["batch_size"])

    obs_dict = env.reset()
    obs = torch.tensor(obs_dict['observation']).float()

    replay_buffer.clear_staged_for_append()
    num_warmup_steps_taken = 0
    while cur_episode < env.config["total_episodes"]: 
        cur_iteration += 1
        cur_iteration_btw_updates += 1
        # For warmup_steps many iterations take random actions to explore better
        if num_warmup_steps_taken < env.config["warmup_steps"]: # take random actions
            action = env.action_space.sample()
            num_warmup_steps_taken += 1
        else: # agent's policy takes action
            with torch.no_grad():
                action = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
                action = torch.squeeze(action, dim=0).numpy()

        # Take action
        next_obs_dict, reward, done, info = env.step(np.copy(action))
        next_obs = torch.tensor(next_obs_dict['observation']).float()
        env.logger.log_tabular(key="train_agent() > after env.step()", value=f'obs: {info["observation"]}, ros_clock: {env.ros_clock}')

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if done and cur_iteration < env.max_episode_length:
            term = True
        else:
            term = False
        
        # Stage current (s,a,s') to replay buffer as to be appended at the end of the current episode
        replay_buffer.stage_for_append(obs, torch.tensor(action), torch.tensor(reward), next_obs, torch.tensor(term))
        env.logger.store(state=obs.numpy().copy(), action=action.copy(), reward=reward.copy()) 

        # Update current state
        obs = next_obs

        if env.config["verbose"]:
            # print(f'Episode: {cur_episode}, Iteration: {cur_iteration}/{info["max_episode_length"]},', 
            # f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}),', \
            #     f'Action: {info["observation"]["latest_action"].tolist()}, Reward: {reward:.3f}')
            print("-"*20)
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')
            print('total_angle_difference_to_goal_in_degrees:', info['observation']['total_angle_difference_to_goal_in_degrees'])
            print('forklift_theta:', info['observation']['forklift_theta'])
            print('forklift_position:', info['observation']['forklift_position_observation']['chassis_bottom_link']['pose']['position'])
            print('target_tf:', info['observation']['target_transform_observation'])
            print('latest_action:', info['observation']['latest_action'])
            print(f'done: {done}, term: {term}')


        if done:
            mean_ep_reward = cum_episode_rewards/cur_iteration
            rewards.append(cum_episode_rewards)
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward/[per episode]", mean_ep_reward, cur_episode)
            tb_summaryWriter.add_scalar("Training Reward/[ep_rew_mean]", np.mean(rewards), cur_episode)
            tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)
            env.logger.log_tensorboard(tb_summaryWriter, agent.critic, cur_episode) # TODO: unit test this function

            # Commit experiences to replay_buffer
            replay_buffer.commit_append()
            replay_buffer.clear_staged_for_append()

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs_dict['observation']).float()

            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0
            cur_iteration = 0


        # Update model if its time
        if (cur_iteration_btw_updates % env.config["update_every"]== 0) and (num_warmup_steps_taken >= env.config["warmup_steps"]) and replay_buffer.can_sample_a_batch():
            print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
            for _ in range(env.config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                cur_num_updates += 1

                # Save the model
                if ((cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 )) or \
                    (cur_episode + 1 == env.config["total_episodes"]) :
                    print("Saving the model ...")
                    agent.save_model()
                
            cur_iteration_btw_updates = 0


def test_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "ForkliftEnv DDPG agent testing_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0
    cur_iteration = 0

    concatenated_obs_dim = sum(env.observation_space['observation'].shape)
    concatenated_action_dim = sum(env.action_space.shape)

    agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["actor_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config['act_noise'], env.config['target_noise'], env.config['clip_noise_range'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space.high).float())
    agent.load_model()
    if env.config["verbose"]:
        print("Loaded a pre-trained agent...")
    agent.eval_mode()

    obs_dict = env.reset()
    obs = torch.tensor(obs_dict['observation']).float()

    while True:
        cur_iteration += 1
        with torch.no_grad():
            action = agent.choose_action(obs.detach().clone()) # predict action in the range of [-1,1]
            action = torch.squeeze(action, dim=0).numpy()

        # Take action
        next_obs_dict, reward, done, info = env.step(np.copy(action))
        next_obs = torch.tensor(next_obs_dict['observation']).float()

        cum_episode_rewards += reward

        # Update current state
        obs = next_obs

        if env.config["verbose"]:
            # print(f'Episode: {cur_episode}, Iteration: {cur_iteration}/{info["max_episode_length"]},', 
            #     f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}),', \
            #         f'Action: {info["observation"]["latest_action"].tolist()}, Reward: {reward:.3f}')
            print(f'episode: {cur_episode}, iteration: {cur_iteration}, action: {action}, reward: {reward}')


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Testing Reward", cum_episode_rewards/cur_iteration, cur_episode)

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = torch.tensor(obs_dict['observation']).float()

            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0
            cur_iteration = 0
    

if __name__ == "__main__":
    main()