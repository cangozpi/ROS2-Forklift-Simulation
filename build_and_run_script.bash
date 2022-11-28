rm -r build/ install/ log/
colcon build --symlink-install
source install/setup.bash

ros2 launch forklift_robot rviz_launch.launch.py
