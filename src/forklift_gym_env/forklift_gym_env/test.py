import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer 
from forklift_gym_env.rl.DDPG.utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime


def main():
    # Initialize Tensorboard
    log_dir, run_name = "logs_tensorboard/", "DDPG_agent_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # Start Env
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    seed_everything(env.config["seed"]) # set seed
    time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    cur_episode = 0
    cum_episode_rewards = 0

    concatenated_obs_dim, concatenated_action_dim = get_concatenated_obs_and_act_dims(env)

    agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, \
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["lr"]), \
            env.config["epsilon"], env.config["epsilon_decay"], env.config["gamma"], env.config["tau"])
    # Load a pre-trained agent_model
    agent.load_model()
    if env.config["verbose"]:
        print("Loaded a pre-trained agent...")

    agent.eval() # TODO: handle .eval() case for testing the model too.

    obs, info = env.reset(seed=env.config["seed"])
    obs = flatten_and_concatenate_observation(obs, env)
    while cur_episode < env.config["total_episodes"]: # simulate action taking of an agent for debugging the env
        with torch.no_grad():
            obs = torch.unsqueeze(obs, dim=0)
            action = agent.choose_action(obs) # predict action in the range of [-1,1]
            action = torch.squeeze(action, dim=0)
        action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it
        # Take action
        next_obs, reward, done, _, info = env.step(action)
        next_obs = flatten_and_concatenate_observation(next_obs, env)

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if info["iteration"] >= env.max_episode_length:
            term = True
        else:
            term = False
        
        # Store experience in replay buffer
        action = flatten_and_concatenate_action(action) 

        # Update current state
        obs = next_obs

        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: {info["agent_location"]}, Target_location: {info["target_location"]}, Reward: {info["reward"]}')
        


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Test Reward", reward, cur_episode)

            # Reset env
            obs, info = env.reset(seed=env.config["seed"])
            obs = flatten_and_concatenate_observation(obs, env)
            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0


