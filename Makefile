SHELL := /bin/bash
BUILD_FILES = log build install

clean:
	rm -r $(BUILD_FILES)

build:
	colcon build --symlink-install

clean_build: clean build

train:
	source install/setup.bash && ros2 run forklift_gym_env forklift_gym_env_training

install_python_deps:
	pip install -r requirements.txt

kill_gazebo_processes:
	kill $(shell pidof gzserver) $(shell pidof gzclient) $(shell pidof gazebo)
