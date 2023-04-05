import gym

from forklift_gym_env.envs.Forklift_env import ForkliftEnv
from forklift_gym_env.rl.sb3_HER.utils import *


from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DDPG, PPO, HerReplayBuffer
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3_contrib import QRDQN, TQC


# SET THIS TO YOUR RL ALGORITHM OF CHOICE
rl_algorithm = "PPO" 
rl_algorithms = ["DDPG", "DDPG_HER", "TQC_HER", "PPO"]
assert rl_algorithm in rl_algorithms
config_path = 'build/forklift_gym_env/forklift_gym_env/config/config.yaml'

def main():
    
# -------------------------------------------------------------------------------------------
# ---------------------------------------------- PPO:
    if rl_algorithm == "PPO":
        mode = "train"
        assert mode in ["train", "test"]

        # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
        env =  ForkliftEnv(config_path=config_path, use_GoalEnv=False)
        # It will check your custom environment and output additional warnings if needed
        # check_env(env)

        seed_everything(env.config["seed"]) # set seed
        # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

        if mode == "train":
            # Initialize the model
            model = PPO(
                "MultiInputPolicy", 
                env, 
                policy_kwargs={
                    # 'activation_fn':torch.nn.LeakyReLU,
                    'net_arch':{
                        'pi':[16, 8], 'vf':[16, 8]
                        }
                },
                verbose=1, 
                tensorboard_log="sb3_tensorboard/"
            )

            model.learn(total_timesteps=500_000, tb_log_name="forklift_env sb3 PPO", reset_num_timesteps=False, log_interval=1, progress_bar=True)
            model.save("sb3_saved_model PPO")
            print("Finished training the agent !")

            # env = model.get_env()
            # del model # remove to demonstrate saving and loading

            mode = "test"

        if mode == "test":
            # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
            model = PPO.load("sb3_saved_model PPO", env=env) # HER requires env passed in

            # Testing the agent
            print("Testing the model:")
            obs = env.reset()
            while True: 
                # action, _states = model.predict(obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                print("action: ", action, "obs:", obs, "reward:", reward)
                # env.render()

                if done:
                    print("Resetting the env !")
                    env.reset()



# -------------------------------------------------------------------------------------------
# ---------------------------------------------- DDPG:
    elif rl_algorithm == "DDPG":
        mode = "train"
        assert mode in ["train", "test"]

        # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
        env =  ForkliftEnv(config_path=config_path, use_GoalEnv=False)
        # It will check your custom environment and output additional warnings if needed
        # check_env(env)

        seed_everything(env.config["seed"]) # set seed
        # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

        if mode == "train":
            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            # Initialize the model
            model = DDPG(
                "MultiInputPolicy",
                env,
                # action_noise=action_noise,
                learning_rate=1e-5,
                # train_freq=(10, 'episode'),
                # seed=3,
                learning_starts=100,
                policy_kwargs={
                    # 'activation_fn':torch.nn.LeakyReLU,
                    'net_arch':{
                        'pi':[16, 8], 'qf':[16, 8]
                        }
                    },
                verbose=1,
                tensorboard_log="sb3_tensorboard/"
            )

            # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
            model.learn(total_timesteps=500_000, tb_log_name="forklift_env sb3 DDPG", reset_num_timesteps=False, log_interval=1, progress_bar=True)
            model.save("sb3_saved_model DDPG")
            print("Finished training the agent !")

            # env = model.get_env()
            # del model # remove to demonstrate saving and loading

            mode = "test"

        if mode == "test":
            # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
            model = DDPG.load("sb3_saved_model DDPG", env=env) # HER requires env passed in

            # Testing the agent
            print("Testing the model:")
            obs = env.reset()
            while True: 
                # action, _states = model.predict(obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                print("action: ", action, "obs:", obs, "reward:", reward)
                # env.render()

                if done:
                    print("Resetting the env !")
                    env.reset()



# -------------------------------------------------------------------------------------------
# ---------------------------------------------- DDPG_HER:
    elif rl_algorithm == "DDPG_HER":
        mode = "train"
        assert mode in ["train", "test"]

        # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
        env =  ForkliftEnv(config_path=config_path, use_GoalEnv=True)
        # It will check your custom environment and output additional warnings if needed
        # check_env(env)

        seed_everything(env.config["seed"]) # set seed
        # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

        if mode == "train":
            # Available strategies (cf paper): future, final, episode
            goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

            # The noise objects for DDPG
            n_actions = env.action_space.shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            # Initialize the model
            model = DDPG(
                "MultiInputPolicy",
                env,
                # action_noise=action_noise,
                replay_buffer_class=HerReplayBuffer,
                learning_rate=1e-5,
                # train_freq=(10, 'episode'),
                # seed=3,
                learning_starts=100,
                # Parameters for HER
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                    online_sampling=True,
                    max_episode_length=100,
                ),
                policy_kwargs={
                    # 'activation_fn':torch.nn.LeakyReLU,
                    'net_arch':{
                        'pi':[16, 8], 'qf':[16, 8]
                        }
                    },
                verbose=1,
                tensorboard_log="sb3_tensorboard/"
            )


            # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
            model.learn(total_timesteps=500_000, tb_log_name="forklift_env sb3 DDPG+HER", reset_num_timesteps=False, log_interval=1, progress_bar=True)
            model.save("sb3_saved_model DDPG_HER")
            print("Finished training the agent !")

            # env = model.get_env()
            # del model # remove to demonstrate saving and loading

            mode = "test"

        if mode == "test":
            # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
            model = DDPG.load("sb3_saved_model DDPG_HER", env=env) # HER requires env passed in

            # Testing the agent
            print("Testing the model:")
            obs = env.reset()
            while True: 
                # action, _states = model.predict(obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                print("action: ", action, "obs:", obs, "reward:", reward)
                # env.render()

                if done:
                    print("Resetting the env !")
                    env.reset()



# -------------------------------------------------------------------------------------------
# ---------------------------------------------- TQC_HER:
    elif rl_algorithm == "TQC_HER":
        mode = "train"
        assert mode in ["train", "test"]

        # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
        env =  ForkliftEnv(config_path=config_path, use_GoalEnv=True)
        # It will check your custom environment and output additional warnings if needed
        # check_env(env)

        seed_everything(env.config["seed"]) # set seed
        # time.sleep(15.0) # delay to compensate for gazebo client window showing up slow

        if mode == "train":
            # Available strategies (cf paper): future, final, episode
            goal_selection_strategy = "future" # equivalent to GoalSelectionStrategy.FUTURE

            # Initialize the model
            model = TQC(
                "MultiInputPolicy", 
                env, 
                top_quantiles_to_drop_per_net=2, 
                policy_kwargs={
                    'n_critics': 2,
                    'n_quantiles': 25,
                    # 'activation_fn':torch.nn.LeakyReLU,
                    'net_arch':{
                        'pi':[16, 8], 'qf':[16, 8]
                        }
                },
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy=goal_selection_strategy,
                    online_sampling=False,
                    max_episode_length=1000,
                ),
                verbose=1, 
                tensorboard_log="sb3_tensorboard/"
            )


            # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
            model.learn(total_timesteps=500_000, tb_log_name="forklift_env sb3 TQC+HER", reset_num_timesteps=False, log_interval=1, progress_bar=True)
            model.save("sb3_saved_model TQC_HER")
            print("Finished training the agent !")

            # env = model.get_env()
            # del model # remove to demonstrate saving and loading

            mode = "test"

        if mode == "test":
            # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
            model = TQC.load("sb3_saved_model TQC", env=env) # HER requires env passed in

            # Testing the agent
            print("Testing the model:")
            obs = env.reset()
            while True: 
                # action, _states = model.predict(obs)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                print("action: ", action, "obs:", obs, "reward:", reward)
                # env.render()

                if done:
                    print("Resetting the env !")
                    env.reset()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------