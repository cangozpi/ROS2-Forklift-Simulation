rm -r build/ install/ log/
colcon build --symlink-install
source install/setup.bash

ros2 launch forklift_robot gazebo_launch.launch.py
