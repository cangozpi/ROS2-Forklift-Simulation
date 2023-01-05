import gym
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def main():
    print("start:")
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    print(env)
    obs, info = env.reset()
    while True: # simulate action taking of an agent for debugging the env
        action = None
        # action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(env.cur_iteration, "EEEEEEEEE")
        if done:
            obs, info = env.reset()
            print("Environment is reset!")
        # print(obs, "AAAAAAAAA")
    print("end")