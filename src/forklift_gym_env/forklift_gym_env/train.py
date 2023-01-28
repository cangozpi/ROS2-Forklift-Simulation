import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def debug_console():
    while True:
        var_name = input("Type the name of the variable you want to print its value of:")
        test = 1
        print(eval(var_name))

def main():
    # Start debugging console
    from threading import Thread
    debug_console_thread = Thread(target=debug_console)
    debug_console_thread.daemon = True
    debug_console_thread.start()

    # Start Env
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    time.sleep(15.0)
    cur_episode = 1
    obs, info = env.reset()
    while True: # simulate action taking of an agent for debugging the env
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f'Episode: {cur_episode}, Iteration: {info["iteration"] + 1}/{info["max_episode_length"]},', 
        f'Agent_location: {info["agent_location"]}, Target_location: {info["target_location"]}')
        if done:
            obs, info = env.reset()
            time.sleep(3)
            cur_episode += 1