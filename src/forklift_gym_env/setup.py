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
        (os.path.join('share', package_name), glob('config/*')), # includes config files inside /config directory
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
            'forklift_gym_env_training = forklift_gym_env.train:main',
        ],
    },
)