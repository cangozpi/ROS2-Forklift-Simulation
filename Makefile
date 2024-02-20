SHELL := /bin/bash
BUILD_FILES = log build install

clean:
	rm -r $(BUILD_FILES)

build:
	colcon build --symlink-install

clean_build: clean build



train_sb3:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_sb3_training

train_DDPG:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_DDPG_training

train_HER_DDPG:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_HER_DDPG_training



DDPG_openAI_gym:
	source install/setup.bash && ros2 run forklift_gym_env openai_gym_env_DDPG



gui_controller:
	source install/setup.bash && ros2 run forklift_gym_env gui_controller

run_pytest:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_pytest 
	@make -s kill_gazebo_processes


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

start_tensorboard_sb3:
	tensorboard --logdir sb3_tensorboard


install_requirements:
	xargs sudo apt -y install < ros2_pkg_requirements.txt && \
	pip install -r requirements.txt

start_rviz:
	rviz2 -d src/forklift_robot/rviz/forklift_with_sensors.rviz
