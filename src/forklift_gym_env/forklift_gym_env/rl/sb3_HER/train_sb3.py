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



def main():
    mode = "train"
    assert mode in ["train", "test"]

    # env = gym.make('forklift_gym_env/ForkliftWorld-v1')
    env =  ForkliftEnvSb3HER()
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
        policy_kwargs = dict(n_critics=2, n_quantiles=25)
        model = TQC(
            "MultiInputPolicy", 
            env, 
            top_quantiles_to_drop_per_net=2, 
            verbose=1, 
            policy_kwargs=policy_kwargs,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy=goal_selection_strategy,
                online_sampling=False,
                max_episode_length=4000,
            ),
            tensorboard_log="sb3_tensorboard/"
        )
        # model = DDPG(
        #     "MultiInputPolicy",
        #     env,
        #     action_noise=action_noise,
        #     replay_buffer_class=HerReplayBuffer,
        #     # Parameters for HER
        #     replay_buffer_kwargs=dict(
        #         n_sampled_goal=4,
        #         goal_selection_strategy=goal_selection_strategy,
        #         online_sampling=True,
        #         max_episode_length=1000,
        #     ),
        #     verbose=1,
        #     tensorboard_log="sb3_tensorboard/"
        # )


        # model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log="sb3_tensorboard/")
        model.learn(total_timesteps=30_000, tb_log_name="forklift_env fix sb3 run", reset_num_timesteps=False, log_interval=1, progress_bar=True)
        model.save("sb3_saved_model")
        print("Finished training the agent !")

        # env = model.get_env()

        # del model # remove to demonstrate saving and loading

        mode = "test"

    if mode == "test":
        # model = DDPG.load("sb3_saved_model") # Non-HER models can use this to load model
        model = TQC.load("sb3_saved_model", env=env) # HER requires env passed in

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
