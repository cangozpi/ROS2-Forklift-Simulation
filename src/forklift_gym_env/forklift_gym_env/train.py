import gym
import time
from forklift_gym_env.envs.forklift_env import ForkliftEnv

def debug_console():
    while True:
        var_name = input("Type the name of the variable you want to print its value of:")
        test = 1
        print(eval(var_name))

def main():
    # start debugging console
    from threading import Thread
    debug_console_thread = Thread(target=debug_console)
    debug_console_thread.daemon = True
    debug_console_thread.start()

    # Start Env
    print("start:")
    env = gym.make('forklift_gym_env/ForkliftWorld-v0')
    print(env)
    time.sleep(15.0)
    obs, info = env.reset()
    while True: # simulate action taking of an agent for debugging the env
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(env.cur_iteration, "EEEEEEEEE")
        if done:
            obs, info = env.reset()
            time.sleep(5)
            #TODO: sleep ?
            print("Environment is reset!")
        # print(obs['depth_camera_raw_image_observation'].shape, "AAAAAAAAA")
        # print(obs['forklift_robot_tf_observation']['chassis_bottom_link']['transform'].dtype == float, "AAAAAAAAA")
        # print(obs['forklift_robot_tf_observation']['chassis_bottom_link']['time'], "AAAAAAAAA")
    print("end")