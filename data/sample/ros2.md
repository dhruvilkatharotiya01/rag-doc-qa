# ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) is a middleware framework for building robot software from many cooperating processes called nodes. Each node does one job, such as reading a camera, estimating the robot pose or planning a path.

Nodes communicate in three main ways. Topics are named, typed publish-subscribe channels for continuous data streams such as sensor readings. Services are request-response calls for short operations, for example resetting a counter. Actions are for long-running goals such as moving an arm to a pose: the client sends a goal, receives periodic feedback and can cancel the goal before it finishes.

Unlike ROS 1, ROS 2 has no central master. It uses DDS (Data Distribution Service) for discovery and transport, so nodes find each other automatically on the network. Quality of Service (QoS) settings control delivery: reliable delivery suits commands that must not be lost, while best-effort delivery suits high-rate sensor data where only the newest message matters.

The tf2 library keeps track of coordinate frames over time, for example the transform between the camera frame and the robot base frame. Launch files start and configure many nodes together, and parameters let the same node be reused with different settings.

MoveIt 2 builds on ROS 2 to provide motion planning, inverse kinematics and collision checking for robot arms.
