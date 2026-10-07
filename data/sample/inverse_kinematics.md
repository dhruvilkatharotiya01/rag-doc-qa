# Forward and Inverse Kinematics of Robot Arms

Forward kinematics computes the pose of the end-effector from the joint angles. For a serial arm it is a chain of homogeneous transformations, one per joint, often written with Denavit-Hartenberg parameters. Forward kinematics always has exactly one answer.

Inverse kinematics (IK) solves the opposite problem: which joint angles place the end-effector at a desired pose. It may have no solution when the pose is outside the workspace, a finite number of solutions, or infinitely many. A seven-axis arm such as the KUKA LBR iiwa is redundant: it has one more joint than the six needed to fix a pose in space, so the elbow can move along a so-called null space while the end-effector stays still. This redundancy can be used to avoid obstacles or joint limits.

Analytical IK solvers derive closed-form equations for a specific robot and are very fast. Numerical solvers iterate using the Jacobian, which maps joint velocities to end-effector velocities. Near a singularity the Jacobian loses rank, the arm loses the ability to move in some direction and the required joint velocities can become very large. The damped least squares method adds a small damping term to keep the solution stable near singularities.

In grasp planning, IK is used to check whether a candidate grasp pose is reachable at all before a motion planner searches for a collision-free path to it.
