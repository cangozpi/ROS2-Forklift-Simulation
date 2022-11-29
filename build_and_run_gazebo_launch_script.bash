rm -r build/ install/ log/
colcon build --symlink-install
source install/setup.bash

gnome-terminal -- ros2 run teleop_twist_keyboard teleop_twist_keyboard &
ros2 launch forklift_robot gazebo_launch.launch.py