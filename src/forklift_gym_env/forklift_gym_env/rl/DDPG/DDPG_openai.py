import gym
import time
from forklift_gym_env.envs.forklift_env_HER import ForkliftEnvHER
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer 
from forklift_gym_env.rl.DDPG.utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime

from forklift_gym_env.envs.utils import read_yaml_config


def main():
    # Read in parameters from config.yaml
    config_path = 'build/forklift_gym_env/forklift_gym_env/config/config_DDPG_openai_env.yaml'
    config = read_yaml_config(config_path)

    # Start Env
    env = gym.make('Pendulum-v1', g=9.81)

    agent = config["agent"]
    assert agent in ["my_DDPG_agent", "sb3_DDPG_agent"]

    if agent == "my_DDPG_agent":
        # Read in parameters from config.yaml
        my_DDPG_agent(env, config)
    elif agent == "sb3_DDPG_agent":
        mode = config["sb3_mode"] # ["train", "test"]
        assert mode in ["train", "test"]

        sb3_DDPG_agent(env, mode=mode)
    else:
        raise Exception("\'agent\' should be either [\'my_DDPG_agent\', \'sb3_DDPG_agent\']")


def my_DDPG_agent(env, config):
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "openAI_pendulum_my_DDPG_agent_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # Start Env
    seed_everything(42) # set seed
    cur_episode = 0
    cum_episode_rewards = 0
    cur_num_updates = 0 # total number of updates including all the episodes
    cur_iteration = 0
    verbose = True
    render = False
    actor_lr = 1e-3
    critic_lr = 1e-3
    warmup_done = False

    # concatenated_obs_dim, concatenated_action_dim, goal_state_dim = get_concatenated_obs_and_act_dims(env)
    concatenated_obs_dim = env.observation_space.shape[-1]
    concatenated_action_dim = env.action_space.shape[-1]

    agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        config["actor_hidden_dims"], config["critic_hidden_dims"], actor_lr, critic_lr, \
            config["initial_epsilon"], config["epsilon_decay"], config['min_epsilon'], config["gamma"], config["tau"], max_action=env.action_space.high.item())
    # agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, env.action_space.high.item())
    # agent.train() # TODO: handle .eval() case for testing the model too.
    replay_buffer = ReplayBuffer(config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, config["batch_size"])
    # replay_buffer = ReplayBuffer(max_size=config["replay_buffer_size"])

    cur_state = env.reset()
    # obs_flattened, goal_state, obs = flatten_and_concatenate_observation(obs, env)
    replay_buffer.clear_staged_for_append()
    while cur_episode < config["total_episodes"]: # simulate action taking of an agent for debugging the env
        cur_iteration += 1
        # For warmup_steps many iterations take random actions to explore better
        if cur_iteration < config["warmup_steps"] and not warmup_done: # take random actions
            if cur_episode > config["warmup_steps"]:
                warmup_done = True

            action = env.action_space.sample()
            # action /= 2 # [-1, +1] range
            # Take action
            # next_obs_dict, reward, done, info = env.step(action)
        else: # agent's policy takes action
            with torch.no_grad():
                action = agent.choose_action(torch.tensor(cur_state)) # predict action in the range of [-1,1]
                # action = agent.select_action(torch.tensor(cur_state)) # predict action in the range of [-1,1]
                action = torch.squeeze(action, dim=0)
                # action *= env.action_space.high
            # action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it
            # action = action.numpy()

        # Take action
        next_obs_dict, reward, done, info = env.step(action)

        reward /= 10 # reward normalization for pendulum env
        # reward += 2
        if render:
            env.render()

        # next_obs_flattened, goal_state, next_obs = flatten_and_concatenate_observation(next_obs, env)

        cum_episode_rewards += reward

        # Store experience in replay buffer
        # action = flatten_and_concatenate_action(env, action) 
        # Stage current (s,a,s') to replay buffer to be appended using HER at the end of the current episode
        replay_buffer.stage_for_append(torch.tensor(cur_state), torch.tensor(action), torch.tensor(reward), \
            torch.tensor(next_obs_dict), torch.tensor(done))
        # agent.replay_buffer.push((cur_state, next_obs_dict, action, reward, done))

        # Update current state
        cur_state = next_obs_dict
        # obs_flattened = next_obs_flattened #TODO: add this to DDPG only training too @bugfix !

        # Update model if its time
        if (cur_iteration % config["update_every"]== 0) and (cur_iteration > config["warmup_steps"]) and replay_buffer.can_sample_a_batch() and \
            cur_episode > 2:
            print(f"Updating the agent with {config['num_updates']} sampled batches.")
            critic_loss = 0
            actor_loss = 0
            for _ in range(config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                # x, y, u, r, d = replay_buffer.sample(config["batch_size"]) 

                # state_batch = torch.FloatTensor(x)
                # action_batch = torch.FloatTensor(u)
                # next_state_batch = torch.FloatTensor(y)
                # terminal_batch = torch.FloatTensor(d)
                # reward_batch = torch.FloatTensor(r)

                cur_critic_loss, cur_actor_loss = agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)
                # cur_critic_loss, cur_actor_loss = agent.update()

                # Accumulate loss for logging
                critic_loss += cur_critic_loss
                actor_loss += cur_actor_loss

            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Critic Loss", critic_loss, cur_num_updates)
            tb_summaryWriter.add_scalar("Actor Loss", actor_loss, cur_num_updates)

            cur_num_updates += 1

        
        # Save the model
        if (cur_num_updates % config["save_every"] == 0) and (cur_num_updates > 0 ):
            print("Saving the model ...")
            agent.save_model()



        if verbose:
            print(f'Episode: {cur_episode}, Iteration: {cur_iteration} Action: {action}, Reward: {reward:.3f}')
        


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward", cum_episode_rewards/cur_iteration, cur_episode)
            tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

            # Commit HER experiences to replay_buffer
            replay_buffer.commit_append(k=0, calc_reward_func=lambda f: (), check_goal_achieved_func=lambda f: ()) # TODO: set k from config.yaml
            # replay_buffer.commit_append(k=config["k"], calc_reward_func=env.calc_reward, check_goal_achieved_func=env.check_goal_achieved) # TODO: set k from config.yaml
            # replay_buffer.commit_append(k=config["k"], calc_reward_func=env.compute_reward, check_goal_achieved_func=env.check_goal_achieved) # TODO: set k from config.yaml

            # Reset env
            cur_state = env.reset()
            # obs_flattened, goal_state, obs = flatten_and_concatenate_observation(obs, env)
            replay_buffer.clear_staged_for_append()
            # time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0
            cur_iteration = 0
        
        if cur_episode % 7 == 0:
            render = True
        else:
            render = False
        if cur_episode > 30:
            render = True



# SB3 ------------
import gym

from stable_baselines3.common.env_checker import check_env
from forklift_gym_env.envs.forklift_env_sb3_HER import ForkliftEnvSb3HER

from forklift_gym_env.rl.sb3_HER.utils import seed_everything

import numpy as np

from stable_baselines3 import DDPG, PPO, HerReplayBuffer 
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import QRDQN, TQC


def sb3_DDPG_agent(env, mode="train"):
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    seed_everything(42) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

    if mode == "train":
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Initialize the model
        # policy_kwargs = dict(n_critics=2, n_quantiles=25)
        # model = TQC(
            # "MultiInputPolicy", 
            # env, 
            # top_quantiles_to_drop_per_net=2, 
            # verbose=1, 
            # policy_kwargs=policy_kwargs,
            # replay_buffer_class=HerReplayBuffer,
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=4,
            #     goal_selection_strategy=goal_selection_strategy,
            #     online_sampling=True,
            #     max_episode_length=200,
            # ),
            # tensorboard_log="sb3_tensorboard/"
        # )
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            # replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=4,
            #     goal_selection_strategy=goal_selection_strategy,
            #     online_sampling=True,
            #     max_episode_length=1000,
            # ),
            verbose=1,
            tensorboard_log="sb3_tensorboard/"
        )


        # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
        model.learn(total_timesteps=30_000, tb_log_name="openai pendulum sb3 run", reset_num_timesteps=False, log_interval=1, progress_bar=True)
        model.save("sb3_saved_model")
        print("Finished training the agent !")

        # env = model.get_env()

        # del model # remove to demonstrate saving and loading

        mode = "test"

    if mode == "test":
        # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
        model = DDPG.load("sb3_saved_model", env=env) # HER requires env passed in

        # Testing the agent
        print("Testing the model:")
        obs = env.reset()
        while True: 
            # action, _states = model.predict(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print("action: ", action, "obs:", obs, "reward:", reward)
            env.render()

            if done:
                print("Resetting the env !")
                env.reset()



