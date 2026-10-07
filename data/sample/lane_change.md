# Lane Changes and Collision Avoidance in Automated Driving

An automated lane change has three parts: deciding whether to change lanes, planning the trajectory, and checking that the manoeuvre is safe throughout its execution.

The decision layer compares the utility of staying in the current lane with the utility of the target lane, for example the achievable speed, and only commits when the gain exceeds a threshold. This hysteresis prevents the vehicle from oscillating between lanes.

Trajectory planning commonly uses polynomial curves. A quintic polynomial in the lateral direction can match position, velocity and acceleration at both the start and the end of the manoeuvre, which yields smooth, comfortable motion. Cubic spline interpolation is a simpler alternative for smoothing a sequence of waypoints.

Safety checking predicts the motion of surrounding vehicles and verifies that a minimum gap is kept. Time-to-collision (TTC) is a widely used metric: the time until two vehicles would collide if both kept their current speed. A lane change is aborted when the predicted TTC to any vehicle falls below a safety threshold, often in the range of two to three seconds.

Scenario-based testing evaluates these functions across many traffic situations, such as dense traffic, a slow vehicle ahead or a fast vehicle approaching from behind, and across a range of ego speeds, to expose edge cases before real-world testing.
