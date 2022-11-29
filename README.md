## forklift_robot package

---

### Build & Run forklift_robot pkg

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
