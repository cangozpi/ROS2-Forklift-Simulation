import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv
from forklift_gym_env.rl.DDPG.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG.Replay_Buffer import ReplayBuffer 
from forklift_gym_env.rl.DDPG.utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime

# ==============================  TODO: move to config.yaml or make it save upon higher eval score performance
save_every = 2
# ============================== 

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
        env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], \
            env.config["epsilon"], env.config["epsilon_decay"], env.config["gamma"], env.config["tau"])
    agent.train() # TODO: handle .eval() case for testing the model too.
    replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, env.config["batch_size"])

    obs, info = env.reset(seed=env.config["seed"])
    obs = flatten_and_concatenate_observation(obs, env)
    while cur_episode < env.config["total_episodes"]: # simulate action taking of an agent for debugging the env
        # For warmup_steps many iterations take random actions to explore better
        if env.cur_iteration < env.config["warmup_steps"]: # take random actions
            action = env.action_space.sample()
            # Take action
            next_obs, reward, done, _, info = env.step(action)
        else: # agent's policy takes action
            with torch.no_grad():
                obs = torch.unsqueeze(obs, dim=0)
                action = agent.choose_action(obs) # predict action
                action = torch.squeeze(action, dim=0)
            action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it
            # Take action
            next_obs, reward, done, _, info = env.step(action)

        # Take action
        next_obs = flatten_and_concatenate_observation(next_obs, env)

        cum_episode_rewards += reward

        # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
        if info["iteration"] >= env.max_episode_length:
            term = True
        else:
            term = False
        
        # Store experience in replay buffer
        action = flatten_and_concatenate_action(action) 
        replay_buffer.append(obs, action, reward, next_obs, term) 

        # Update current state
        obs = next_obs

        # Update model if its time
        if (info["iteration"] % env.config["update_every"]== 0) and (info["iteration"] > env.config["warmup_steps"]):
            print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
            for _ in range(env.config["num_updates"]):
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = replay_buffer.sample_batch() 
                agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch)

        
        # Save the model
        if (info["iteration"] > env.config["warmup_steps"]) and (info["iteration"] % save_every == 0):
            print("Saving the model ...")
            agent.save_model()



        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: {info["agent_location"]}, Target_location: {info["target_location"]}, Reward: {info["reward"]}')
        


        if done:
            # Log to Tensorboard
            tb_summaryWriter.add_scalar("Training Reward", reward, cur_episode)

            # Reset env
            obs, info = env.reset(seed=env.config["seed"])
            obs = flatten_and_concatenate_observation(obs, env)
            time.sleep(3)
            # Reset episode parameters for a new episode
            cur_episode += 1
            cum_episode_rewards = 0


