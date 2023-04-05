from setuptools import setup
import os
from glob import glob

package_name = 'forklift_gym_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('worlds/*')), # includes files inside /worlds directory
        # (os.path.join('share', package_name), glob('config/*')), # includes config files inside /config directory
        (os.path.join('share', package_name), glob('models/pallet/*.sdf')), # includes files inside /models directory (e.g. /pallet).
        (os.path.join('share', package_name), glob('models/pallet/meshes/*.dae')), # includes files inside /models directory (e.g. /pallet).
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cangozpi',
    maintainer_email='cangozpinar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'forklift_gym_env_DDPG_training = forklift_gym_env.rl.DDPG.train_DDPG:main',
            'forklift_gym_env_HER_DDPG = forklift_gym_env.rl.DDPG_HER.HER_DDPG_forklift_env:main',

            'openai_gym_env_DDPG = forklift_gym_env.rl.DDPG.DDPG_openai:main',

            'forklift_gym_env_training = forklift_gym_env.train:main',
            'forklift_gym_env_sb3_training = forklift_gym_env.rl.sb3_HER.train_sb3:main',
            'forklift_gym_env_testing = forklift_gym_env.test:main',
            'gui_controller = forklift_gym_env.gui_controller.gui_controller:main',
        ],
    },
)
