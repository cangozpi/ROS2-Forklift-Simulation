from setuptools import setup
# Imports below are added by me
import os
from glob import glob
from setuptools import find_packages

package_name = 'forklift_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ones below are added by me:
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')), # includes files inside /launch directory
        (os.path.join('share', package_name), glob('urdf/*')), # includes files inside /urdf directory
        (os.path.join('share', package_name), glob('rviz/*')), # includes files inside /rviz directory
        (os.path.join('share', package_name), glob('config/*')), # includes files inside /config directory
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
            'fork_controller_publisher = forklift_robot.fork_controller_publisher:main',
            'camera_raw_image_subscriber = forklift_robot.camera_raw_image_subscriber:main',
            'depth_camera_raw_image_subscriber = forklift_robot.depth_camera_raw_image_subscriber:main',
            'lidar_scan_subscriber = forklift_robot.lidar_scan_subscriber:main',
            'odom_subscriber = forklift_robot.odom_subscriber:main',
            'diff_cont_cmd_vel_unstamped_publisher = forklift_robot.diff_cont_cmd_vel_unstamped_publisher:main'
        ],
    },
)
