import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def main():
    # Start Env
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    time.sleep(15.0)
    cur_episode = 1
    obs, info = env.reset()
    while True: # simulate action taking of an agent for debugging the env
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)

        if info["verbose"]:
            print(f'Episode: {cur_episode}, Iteration: {info["iteration"]}/{info["max_episode_length"]},', 
            f'Agent_location: {info["agent_location"]}, Target_location: {info["target_location"]}')

        if done:
            obs, info = env.reset()
            time.sleep(3)
            cur_episode += 1