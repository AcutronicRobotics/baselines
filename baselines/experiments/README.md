## Required pkgs
 1. ROS 2
 2. [control_msgs](https://github.com/erlerobot/control_msgs)
 3. [ROS2 to gazebo bridge](https://github.com/erlerobot/hros_pkgs/tree/master/hros_bridges/scara_bridge_inverse)
 4. [scara_e1_gazebo](https://github.com/erlerobot/scara_e1)
 5. [PyKDL](https://github.com/erlerobot/orocos_kinematics_dynamics) placed in your ros2_ws

## How to Run?
 1. Add all the pkgs in your ROS2 workspace. Compile with ament build, source
 2. launch scara_e1_gazebo
 3. launch scara_bridge_inverse
 4. run: python3 baselines/experiments/scara_3joints/run_scara_3jnts.py
