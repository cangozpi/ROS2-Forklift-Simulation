FROM osrf/ros:humble-desktop

# change shell to BASH
SHELL ["/bin/bash", "-c"]

# update apt pkgs
RUN apt update && apt upgrade -y
RUN apt-get update && apt-get upgrade -y

# install Gazebo-11.10.2
RUN curl -sSL http://get.gazebosim.org | sh


# setup ros2 and Gazebo everytime using bashrc
RUN echo "# Lines below are added by me -->" >> /root/.bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source /usr/share/gazebo/setup.bash" >> /root/.bashrc
RUN echo "# <-- Lines added by me ends here " >> /root/.bashrc
RUN source /opt/ros/humble/setup.bash
RUN source /usr/share/gazebo/setup.bash


#install pip for python3.10
RUN apt install python3-pip -y


# copy code files and set it as the work dir
RUN mkdir Docker_shared && cd /Docker_shared
COPY ./ /Docker_shared/ROS2-FORKLIFT-SIMULATION
WORKDIR /Docker_shared/ROS2-FORKLIFT-SIMULATION

# install ros2 dependencies
# RUN xargs sudo apt -y install < ros2_pkg_requirements.txt


# install python dependencies (Note: ros2_pkg_requirements.txt must be installed first to avoid errors)
# RUN pip install -r requirements.txt
