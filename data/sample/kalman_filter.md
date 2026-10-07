# Kalman Filter and State Estimation

A Kalman filter estimates the hidden state of a linear dynamic system from noisy measurements. It assumes Gaussian process noise and Gaussian measurement noise, and under these assumptions it is the optimal estimator in the minimum mean squared error sense.

Each cycle has two steps. The prediction step propagates the previous state estimate and its covariance through the motion model. The update step corrects the prediction with a new measurement. The Kalman gain decides how much to trust the measurement compared with the prediction: when the measurement noise is large, the gain is small and the filter relies more on the model.

Tuning is done through two covariance matrices. The process noise covariance Q describes how much the true system deviates from the model, and the measurement noise covariance R describes sensor accuracy. Choosing Q too small makes the filter overconfident and slow to react to real changes.

Real vehicles and robots are usually nonlinear. The Extended Kalman Filter linearises the motion and measurement models around the current estimate using Jacobians. The Unscented Kalman Filter instead passes a small set of carefully chosen sigma points through the nonlinear functions, which often handles strong nonlinearities better and avoids computing Jacobians.

Typical uses include fusing wheel odometry, an inertial measurement unit and GNSS to localise a vehicle, and tracking the position and speed of surrounding cars from radar or camera detections.
