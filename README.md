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
