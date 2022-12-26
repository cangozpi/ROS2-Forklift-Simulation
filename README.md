## forklift_robot package

---

### Build & Run forklift_robot in gazebo+rviz+teleop_twist_keyboard

This will open forklift model visualized in gazebo and rviz2 with teleop_twist_keyboard in another terminal to control the model manually.

1. Building the package:
   ```bash
   rm -r build/ install/ log/
   colcon build --symlink-install
   source install/setup.bash
   ```
2. Launching:

   ```bash
   ros2 launch gazebo_launch.launch.py
   ```

or

- You can skip these steps 1 and 2 by running a bash script that does the same think:

  ```bash
  source build_and_run_gazebo_launch_script.bash
  ```

---

### Build & Run forklift_robot in rviz pkg

This will open forklift model visualized in rviz2 with joint_state_publisher_gui to control the model manually.

1. Building the package:
   ```bash
   rm -r build/ install/ log/
   colcon build --symlink-install
   source install/setup.bash
   ```
2. Launching:

   ```bash
   ros2 launch forklift_robot rviz_launch.launch.py
   ```

or

- You can skip these steps 1 and 2 by running a bash script that does the same think:

  ```bash
  source build_and_run__rviz_launch_script.bash
  ```

---

### Build & Run forklift_robot in gazebo with **ros2_control**

This will start ros2 controllers, spawn robot in gazebo.

- Started Ros2 Controllers are as follows:

  1. Differential Controller for wheels of the forklift (_'diff_cont'_)
  2. Forward Command Controller for lifting the fork of the forklift (_'fork_joint_controller'_)
  3. Joint State Broadcaster for broadcasting the joint states ('_joint_broad_')

- Available sensors are as follows:

  1. Lidar
  2. Depth Camera

     (_Note that you can swap Depth Camera with a regular Camera by modifying forklift.urdf.xacro file as follows_):

     ```xml
       <!-- <xacro:include filename="camera.xacro"/> --> <!-- Uncomment this line, and comment the line below to switch between Depth Camera and a Regular Camera -->
       <xacro:include filename="depth_camera.xacro"/>
     ```

1. Building the package:

```bash
rm -r build/ install/ log/
colcon build --symlink-install
```

2. Launching:

   On another terminal:

   ```bash
   source install/setup.bash
   ros2 launch forklift_robot demo_launch.launch.py
   ```

3. Testing (sending commands to controllers):

   - Commanding '_fork_joint_controller_'

     On another terminal do:

     ```bash
     ros2 topic pub /fork_joint_controller/commands std_msgs/msg/Float64MultiArray "{data: [3.0]}"
     ```

   - Commanding '_diff_cont_'

     On another terminal do:

     ```bash
     ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/diff_cont/cmd_vel_unstamped
     ```

   * Inspecting available hardware_interfaces and controllers:

     ```bash
     # To list loaded controllers
     ros2 control list_controllers

     # To list available hardware interfaces:
     ros2 control list_hardware_interfaces
     ```

   * Visualize/stream the depth camera sensor's _/camera/image_raw_ topic.

   ```bash
   ros2 run forklift_robot depth_camera_raw_image_subscriber
   ```

---
