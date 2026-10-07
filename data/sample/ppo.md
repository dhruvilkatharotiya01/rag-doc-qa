# Proximal Policy Optimization (PPO)

Proximal Policy Optimization is an on-policy policy-gradient algorithm. It collects a batch of trajectories with the current policy, estimates how much better each action was than expected, and then improves the policy with several epochs of minibatch gradient steps on that batch.

The key idea is to keep each policy update small. PPO computes the probability ratio between the new and the old policy for every sampled action and clips this ratio to a narrow interval, typically between 0.8 and 1.2. Clipping removes the incentive to move the policy far away from the one that collected the data, which prevents destructive updates without the second-order optimisation used by TRPO.

Advantages are usually computed with Generalised Advantage Estimation (GAE). Its lambda parameter trades bias against variance: values close to one give low bias but high variance, smaller values the opposite.

PPO trains a value function alongside the policy and often adds an entropy bonus to the loss to keep exploring. Because it discards data after each update, PPO needs many environment interactions, but it is simple, robust to hyperparameters and easy to parallelise across many simulated environments. This makes it a popular choice for training robot locomotion in massively parallel simulators.
