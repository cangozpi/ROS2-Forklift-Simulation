from gym.envs.registration import register


# ForkliftEnv which supports both gym.Env and gym.GoalEnv
register(
    id='forklift_gym_env/ForkliftWorld-v0',
    entry_point='forklift_gym_env.envs.Forklift_env:ForkliftEnv',
    # max_episode_steps=300, # TODO: set this maybe
)