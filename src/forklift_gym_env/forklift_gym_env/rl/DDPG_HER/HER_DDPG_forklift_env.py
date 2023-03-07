import gym
import time
from forklift_gym_env.envs.forklift_env_HER import ForkliftEnvHER
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG_HER.HER_Replay_Buffer import HER_ReplayBuffer
from forklift_gym_env.rl.DDPG_HER.utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime


def main():
    # Read in parameters from config.yaml
    config_path = 'build/forklift_gym_env/forklift_gym_env/config/config_HER_DDPG_forklift_env.yaml'

    # Start Env
    # env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    env = ForkliftEnvHER(config_path=config_path)

    mode = env.config["mode"]
    if mode == "train": # train agent
        train_agent(env)
    elif mode == "test": # test pre-trained agent
        test_agent(env)
    else:
        raise Exception('\'mode\' must be either [\'train\', \'test\']')


def train_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "train_DDPG_HER_agent_forklift_env" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0
    cur_num_updates = 0 # total number of updates including all the episodes

    concatenated_obs_dim, concatenated_action_dim, goal_state_dim = get_concatenated_obs_and_act_dims(env)

    agent = DDPG_Agent((concatenated_obs_dim + goal_state_dim), concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["actor_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space["diff_cont_action"].high).float())
    # agent.train() # TODO: handle .eval() case for testing the model too.
    her_replay_buffer = HER_ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, goal_state_dim, env.config["batch_size"])

    obs_dict = env.reset()
    # Concatenate observation with goal_state for regular DDPG agent
    obs = np.concatenate((obs_dict['observation'].reshape(-1), obs_dict['desired_goal'].reshape(-1)), axis=0)

    her_replay_buffer.clear_staged_for_append()
    while cur_episode < env.config["total_episodes"]: # simulate action taking of an agent for debugging the env
        # For warmup_steps many iterations take random actions to explore better
        if env.cur_iteration < env.config["warmup_steps"]: # take random actions
            action = env.action_space.sample()
        else: # agent's policy takes action
            with torch.no_grad():
                action = agent.choose_action(torch.tensor(obs).float()) # predict action in the range of [-1,1]
                action = torch.squeeze(action, dim=0)
            action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it

        # Take action
        next_obs_dict, reward, done, info = env.step(action)
        # Concatenate observation with goal_state for regular DDPG agent
        next_obs = np.concatenate((next_obs_dict['observation'].reshape(-1), next_obs_dict['desired_goal'].reshape(-1)), axis=0)

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if info["iteration"] >= env.max_episode_length:
            term = False # TODO: Check if I set it right and if so change it for the other DDPG env too
        else:
            term = True
        
        # Store experience in replay buffer
        action = flatten_and_concatenate_action(env, action) 
        # Stage current (s,a,s') to replay buffer to be appended using HER at the end of the current episode
        her_replay_buffer.stage_for_append(torch.tensor(obs_dict['observation']), torch.tensor(action), torch.tensor(reward), \
            torch.tensor(next_obs_dict['observation']), torch.tensor(term), torch.tensor(next_obs_dict['desired_goal']), torch.tensor(next_obs_dict['achieved_goal']))

        # Update current state
        obs = next_obs
        obs_dict = next_obs_dict
        # obs_flattened = next_obs_flattened #TODO: add this to DDPG only training too @bugfix !

        # Update model if its time
        if (info["iteration"] % env.config["update_every"]== 0) and (info["iteration"] > env.config["warmup_steps"]) and her_replay_buffer.can_sample_a_batch():
            print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
            critic_loss = 0
            actor_loss = 0
            for _ in range(env.config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, \
                    desired_goal_batch, achieved_goal_batch = her_replay_buffer.sample_batch() 

                # Concatenate observation with goal_state for regular DDPG agent
                obs_batch = torch.concat((state_batch, desired_goal_batch), dim=-1)
                next_obs_batch = torch.concat((next_state_batch, desired_goal_batch), dim=-1)

                cur_critic_loss, cur_actor_loss = agent.update(obs_batch, action_batch, reward_batch, next_obs_batch, terminal_batch)

                # Accumulate loss for logging
                critic_loss += cur_critic_loss
                actor_loss += cur_actor_loss

                cur_num_updates += 1

            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Critic Loss", critic_loss, cur_num_updates)
            tb_summaryWriter.add_scalar("Actor Loss", actor_loss, cur_num_updates)


        
        # Save the model
        if (cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 ):
            print("Saving the model ...")
            agent.save_model()



        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}),', \
                f'Action: {action.tolist()}, Reward: {info["reward"]:.3f}')
        


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward", cum_episode_rewards/info["iteration"], cur_episode)
            tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

            # Commit HER experiences to replay_buffer
            her_replay_buffer.commit_append(k=env.config["k"], calc_reward_func=env.calc_reward, check_goal_achieved_func=env.check_goal_achieved) # TODO: set k from config.yaml
            # replay_buffer.commit_append(k=env.config["k"], calc_reward_func=env.compute_reward, check_goal_achieved_func=env.check_goal_achieved) # TODO: set k from config.yaml
            her_replay_buffer.clear_staged_for_append()

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = np.concatenate((obs_dict['observation'].reshape(-1), obs_dict['desired_goal'].reshape(-1)), axis=0)

            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0




def test_agent(env):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "test_DDPG_HER_agent_forklift_env_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0

    concatenated_obs_dim, concatenated_action_dim, goal_state_dim = get_concatenated_obs_and_act_dims(env)

    agent = DDPG_Agent((concatenated_obs_dim + goal_state_dim), concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["actor_lr"]), float(env.config["actor_lr"]), \
            env.config["initial_epsilon"], env.config["epsilon_decay"], env.config['min_epsilon'], env.config["gamma"], env.config["tau"], \
                max_action=torch.tensor(env.action_space["diff_cont_action"].high).float())
    agent.load_model()
    if env.config["verbose"]:
        print("Loaded a pre-trained agent...")
    agent.eval_mode() # TODO: handle .eval() case for testing the model too.

    obs_dict = env.reset()
    # Concatenate observation with goal_state for regular DDPG agent
    obs = np.concatenate((obs_dict['observation'].reshape(-1), obs_dict['desired_goal'].reshape(-1)), axis=0)

    while True: 
        with torch.no_grad():
            action = agent.choose_action(torch.tensor(obs).float()) # predict action in the range of [-1,1]
            action = torch.squeeze(action, dim=0)
        action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it

        # Take action
        next_obs_dict, reward, done, info = env.step(action)
        # Concatenate observation with goal_state for regular DDPG agent
        next_obs = np.concatenate((next_obs_dict['observation'].reshape(-1), next_obs_dict['desired_goal'].reshape(-1)), axis=0)

        cum_episode_rewards += reward

        # Store experience in replay buffer
        action = flatten_and_concatenate_action(env, action) 

        # Update current state
        obs = next_obs
        obs_dict = next_obs_dict

        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}),', \
                f'Action: {action.tolist()}, Reward: {info["reward"]:.3f}')
        


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Testing Reward", cum_episode_rewards/info["iteration"], cur_episode)

            # Reset env
            obs_dict = env.reset()
            # Concatenate observation with goal_state for regular DDPG agent
            obs = np.concatenate((obs_dict['observation'].reshape(-1), obs_dict['desired_goal'].reshape(-1)), axis=0)

            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0