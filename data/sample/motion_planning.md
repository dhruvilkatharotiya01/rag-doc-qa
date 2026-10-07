# Motion and Path Planning

Path planning finds a collision-free route from a start configuration to a goal configuration. Motion planning additionally respects the robot's dynamics and produces a time-parameterised trajectory.

Grid-based planners such as A* search a discretised map. A* expands nodes in order of cost-so-far plus a heuristic estimate of the remaining cost; with an admissible heuristic it is guaranteed to find the shortest path on the grid. Its weakness is the curse of dimensionality: discretising the joint space of a seven-axis robot arm is impractical.

Sampling-based planners avoid explicit discretisation. RRT (Rapidly-exploring Random Tree) grows a tree by sampling random configurations and extending the nearest node towards them. RRT finds a feasible path quickly but not an optimal one. RRT* adds a rewiring step that makes the solution converge towards the optimal path as more samples are drawn. PRM (Probabilistic Roadmap) instead builds a reusable graph of collision-free samples, which suits many queries in a static environment.

For manipulators such as the KUKA LBR iiwa, the planner works in joint space while collision checking is done against a model of the robot and its environment. The raw path is then smoothed, for example with cubic splines, and time-parameterised to respect joint velocity and acceleration limits.

Grasp selection and path planning interact: a grasp that looks good in isolation may be unreachable or force an awkward arm configuration, so task-aware systems score grasps by both grasp quality and the feasibility of the resulting motion.
