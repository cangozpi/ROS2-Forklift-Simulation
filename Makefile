SHELL := /bin/bash
BUILD_FILES = log build install

clean:
	rm -r $(BUILD_FILES)

build:
	colcon build --symlink-install

clean_build: clean build

train:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_training

gui_controller:
	source install/setup.bash && ros2 run forklift_gym_env gui_controller

install_python_deps:
	pip install -r requirements.txt

kill_gazebo_processes:
	kill -9 $(shell pidof gzserver) $(shell pidof gzclient) $(shell pidof gazebo)

try: 
	export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}:"/home/cangozpi/Desktop/forklift_ws/build/ros_gazebo_plugins" && gazebo src/ros_gazebo_plugins/collision_detection.world --verbose
exportGZ:
	export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}"/home/cangozpi/Desktop/forklift_ws/build/ros_gazebo_plugins:"