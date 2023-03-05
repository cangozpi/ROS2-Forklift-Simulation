SHELL := /bin/bash
BUILD_FILES = log build install

clean:
	rm -r $(BUILD_FILES)

build:
	colcon build --symlink-install

clean_build: clean build

train_DDPG:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_training

train_HER:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_HER_training

train_sb3:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_sb3_training

train_openai_gym_HER:
	source install/setup.bash && ros2 run forklift_gym_env openai_gym_env_HER_training

test:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_testing

gui_controller:
	source install/setup.bash && ros2 run forklift_gym_env gui_controller

install_python_deps:
	pip install -r requirements.txt

kill_gazebo_processes:
	kill -9 $(shell pidof gzserver) $(shell pidof gzclient) $(shell pidof gazebo)

manual_launch: 
	source install/setup.bash && ros2 run forklift_gym_env gui_controller & \
	export GAZEBO_PLUGIN_PATH=${GAZEBO_PLUGIN_PATH}"/home/cangozpi/Desktop/forklift_ws/build/ros_gazebo_plugins:" && \
	source install/setup.bash && ros2 launch forklift_robot demo_launch.launch.py 

start_tensorboard:
	tensorboard --logdir logs_tensorboard

