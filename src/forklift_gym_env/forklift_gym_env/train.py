import gym
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def main():
    print("start:")
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    print(env)
    obs, info = env.reset()
    while True: # simulate action taking of an agent for debugging the env
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(env.cur_iteration, "EEEEEEEEE")
        if done:
            obs, info = env.reset()
            print("Environment is reset!")
        # print(obs['depth_camera_raw_image_observation'].shape, "AAAAAAAAA")
        # print(obs['forklift_robot_tf_observation']['chassis_bottom_link']['transform'].dtype == float, "AAAAAAAAA")
        # print(obs['forklift_robot_tf_observation']['chassis_bottom_link']['time'], "AAAAAAAAA")
    print("end")