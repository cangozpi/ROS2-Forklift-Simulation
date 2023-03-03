import gym
import time
from forklift_gym_env.envs.forklift_env_HER import ForkliftEnv
from forklift_gym_env.rl.DDPG_HER.DDPG_Agent import DDPG_Agent
from forklift_gym_env.rl.DDPG_HER.Replay_Buffer import ReplayBuffer 
from forklift_gym_env.rl.DDPG_HER.utils import *

from torch.utils.tensorboard import SummaryWriter
import datetime

from stable_baselines3.common.env_checker import check_env
from forklift_gym_env.envs.forklift_env_HER import ForkliftEnv



import numpy as np

from stable_baselines3 import DDPG, PPO, HerReplayBuffer 
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import QRDQN, TQC

from stable_baselines3.common.evaluation import evaluate_policy



def main():

    mode = "train"
    assert mode in ["train", "test"]

    # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
    env = gym.make("MountainCarContinuous-v0")
    print(f'action_space: {env.action_space} action_space.shape: {env.action_space.shape}, observation_space: {env.observation_space}')
    # It will check your custom environment and output additional warnings if needed
    # check_env(env)

    # seed_everything(env.config["seed"]) # set seed
    seed_everything(42) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

    if mode == "train":
        # Available strategies (cf paper): future, final, episode
        goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

        # The noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

        # Initialize the model
        policy_kwargs = dict(n_critics=2, n_quantiles=25)
        # model = TQC(
        #     "MultiInputPolicy", 
        #     env, 
        #     top_quantiles_to_drop_per_net=2, 
        #     verbose=1, 
        #     policy_kwargs=policy_kwargs,
        #     replay_buffer_class=HerReplayBuffer,
        #     replay_buffer_kwargs=dict(
        #         n_sampled_goal=4,
        #         goal_selection_strategy=goal_selection_strategy,
        #         online_sampling=False,
        #         max_episode_length=1000,
        #     ),
        #     tensorboard_log="sb3_tensorboard/"
        # )
        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            # replay_buffer_class=HerReplayBuffer,
            # # Parameters for HER
            # replay_buffer_kwargs=dict(
            #     n_sampled_goal=4,
            #     goal_selection_strategy=goal_selection_strategy,
            #     online_sampling=True,
            #     max_episode_length=1000,
            # ),
            verbose=1,
            tensorboard_log="sb3_tensorboard/",
            # actor_lr=0.0001, 
            # critic_lr=0.001,
            learning_rate=1e-4,
            batch_size=512,
            gamma=0.9999,
            # nb_train_steps=50,
            tau=0.001
        )
        # model = PPO(
        #     policy="MlpPolicy",
        #     env=env,
        #     seed=0,
        #     batch_size=256,
        #     ent_coef=0.00429,
        #     learning_rate=7.77e-05,
        #     n_epochs=10,
        #     n_steps=8,
        #     gae_lambda=0.9,
        #     gamma=0.9999,
        #     clip_range=0.1,
        #     max_grad_norm =5,
        #     vf_coef=0.19,
        #     use_sde=True,
        #     policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
        #     verbose=1,
        #     tensorboard_log="sb3_tensorboard/"
        # )


        # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
        for i in range(1):
            model.learn(total_timesteps=20_000, tb_log_name="2 first run", reset_num_timesteps=False, progress_bar=True, log_interval=1) # log_interval=10
            model.save("sb3_saved_model")
            print(f'Training checkpoint: {i} done.')
        print("Finished training the agent !")

        # Evaluate the agent
        mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
        print(f'Evaluating the model: mean_reward: {mean_reward}, std_reward: {std_reward}')

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
            action, _ = model.predict(obs, deterministic=True) # deterministic=True
            obs, reward, done, info = env.step(action)
            print("action: ", action, "obs:", obs, "reward:", reward)
            env.render()

            if done:
                print("Resetting the env !")
                env.reset()


    # MY CODE BELOW -----------------------------
    # Initialize Tensorboard
    # log_dir, run_name = "logs_tensorboard/", "DDPG_HER_agent_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tb_summaryWriter = SummaryWriter(log_dir + run_name)

    # # Start Env
    # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
    # seed_everything(env.config["seed"]) # set seed
    # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow
    # cur_episode = 0
    # cum_episode_rewards = 0
    # cur_num_updates = 0 # total number of updates including all the episodes

    # concatenated_obs_dim, concatenated_action_dim, goal_state_dim = get_concatenated_obs_and_act_dims(env)

    # agent = DDPG_Agent(concatenated_obs_dim, concatenated_action_dim, goal_state_dim, \
    #     env.config["actor_hidden_dims"], env.config["critic_hidden_dims"], float(env.config["lr"]), \
    #         env.config["epsilon"], env.config["epsilon_decay"], env.config["gamma"], env.config["tau"])
    # agent.train() # TODO: handle .eval() case for testing the model too.
    # replay_buffer = ReplayBuffer(env.config["replay_buffer_size"], concatenated_obs_dim, concatenated_action_dim, goal_state_dim, env.config["batch_size"])

    # obs, info = env.reset()
    # obs_flattened, goal_state, obs = flatten_and_concatenate_observation(obs, env)
    # replay_buffer.clear_staged_for_append()
    # while cur_episode < env.config["total_episodes"]: # simulate action taking of an agent for debugging the env
    #     # For warmup_steps many iterations take random actions to explore better
    #     if env.cur_iteration < env.config["warmup_steps"]: # take random actions
    #         action = env.action_space.sample()
    #         # Take action
    #         next_obs, reward, done, _, info = env.step(action)
    #     else: # agent's policy takes action
    #         with torch.no_grad():
    #             action = agent.choose_action(obs_flattened, torch.tensor(goal_state)) # predict action in the range of [-1,1]
    #             action = torch.squeeze(action, dim=0)
    #         action = convert_agent_action_to_dict(action, env) # convert agent action to action dict so that env.step() can parse it
    #         # Take action
    #         next_obs, reward, done, _, info = env.step(action)

    #     next_obs_flattened, goal_state, next_obs = flatten_and_concatenate_observation(next_obs, env)

    #     cum_episode_rewards += reward

    #     # Check if "done" stands for the terminal state or for end of max_episode_length (important for target value calculation)
    #     if info["iteration"] >= env.max_episode_length:
    #         term = False # TODO: Check if I set it right and if so change it for the other DDPG env too
    #     else:
    #         term = True
        
    #     # Store experience in replay buffer
    #     action = flatten_and_concatenate_action(action) 
    #     # Stage current (s,a,s') to replay buffer to be appended using HER at the end of the current episode
    #     replay_buffer.stage_for_append(obs_flattened, action, reward, next_obs_flattened, term, goal_state, next_obs)

    #     # Update current state
    #     obs = next_obs
    #     obs_flattened = next_obs_flattened #TODO: add this to DDPG only training too @bugfix !

    #     # Update model if its time
    #     if (info["iteration"] % env.config["update_every"]== 0) and (info["iteration"] > env.config["warmup_steps"]):
    #         print(f"Updating the agent with {env.config['num_updates']} sampled batches.")
    #         critic_loss = 0
    #         actor_loss = 0
    #         for _ in range(env.config["num_updates"]):
    #             state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, goal_state_batch = replay_buffer.sample_batch() 
    #             cur_critic_loss, cur_actor_loss = agent.update(state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, goal_state_batch)

    #             # Accumulate loss for logging
    #             critic_loss += cur_critic_loss
    #             actor_loss += cur_actor_loss

    #         # Log to Tensorboard
    #         tb_summaryWriter.add_scalar("Critic Loss", critic_loss, cur_num_updates)
    #         tb_summaryWriter.add_scalar("Actor Loss", actor_loss, cur_num_updates)

    #         cur_num_updates += 1

        
    #     # Save the model
    #     if (cur_num_updates % env.config["save_every"] == 0) and (cur_num_updates > 0 ):
    #         print("Saving the model ...")
    #         agent.save_model()



    #     if info["verbose"]:
    #         print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
    #         f'Agent_location: ({info["agent_location"][0]:.2f}, {info["agent_location"][1]:.2f}), Target_location: ({info["target_location"][0]:.2f}, {info["target_location"][1]:.2f}), Reward: {info["reward"]:.3f}')
        


    #     if done:
    #         # Log to Tensorboard
    #         tb_summaryWriter.add_scalar("Training Reward", reward, cur_episode)
    #         tb_summaryWriter.add_scalar("Training epsilon", agent.epsilon, cur_episode)

    #         # Commit HER experiences to replay_bufferj
    #         replay_buffer.commit_append(k=env.config["k"], calc_reward_func=env.calc_reward, check_goal_achieved_func=env.check_goal_achieved) # TODO: set k from config.yaml

    #         # Reset env
    #         obs, info = env.reset(seed=env.config["seed"])
    #         obs_flattened, goal_state, obs = flatten_and_concatenate_observation(obs, env)
    #         replay_buffer.clear_staged_for_append()
    #         time.sleep(3)
    #         # Reset episode parameters for a new episode
    #         cur_episode += 1
    #         cum_episode_rewards = 0


