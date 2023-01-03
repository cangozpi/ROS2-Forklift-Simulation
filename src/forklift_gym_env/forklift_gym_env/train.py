import gym
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def main():
    print("start:")
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    print(env)
    print("end")