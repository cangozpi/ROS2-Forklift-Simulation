rm -r build/ install/ log/
colcon build --symlink-install
source install/setup.bash

ros2 launch forklift_robot ros2Control_gazebo_launch.launch.py &
gnome-terminal -- ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/diff_cont/cmd_vel_unstamped