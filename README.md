## forklift_robot package

---

### Build & Run forklift_robot pkg

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

---
