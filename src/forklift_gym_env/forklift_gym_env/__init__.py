from gym.envs.registration import register

# DDPG environment
register(
    id='forklift_gym_env/ForkliftWorld-v0',
    entry_point='forklift_gym_env.envs.forklift_env_HER:ForkliftEnvHER',
    # max_episode_steps=300, # TODO: set this maybe
)

# # DDPG_HER environment
register(
    id='forklift_gym_env/ForkliftWorld-v1',
    entry_point='forklift_gym_env.envs.forklift_env_sb3_HER:ForkliftEnvSb3HER',
    # max_episode_steps=300, # TODO: set this maybe
)