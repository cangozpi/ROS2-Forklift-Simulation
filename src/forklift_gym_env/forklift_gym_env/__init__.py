from gym.envs.registration import register

register(
    id='forklift_gym_env/ForkliftWorld-v0',
    entry_point='forklift_gym_env.envs:ForkliftEnv',
    # max_episode_steps=300, # TODO: set this maybe
)