# Soft Actor-Critic (SAC)

Soft Actor-Critic is an off-policy reinforcement learning algorithm for continuous action spaces. Instead of maximising only the expected return, SAC maximises the expected return plus the entropy of the policy. The entropy term rewards the agent for acting as randomly as possible while still solving the task, which improves exploration and makes training more stable.

SAC learns three kinds of networks: a stochastic policy (the actor), and two Q-functions (the critics). Using two critics and taking the minimum of their estimates reduces the overestimation bias that is common in Q-learning. Target networks for the critics are updated slowly with Polyak averaging.

The temperature parameter, often written as alpha, controls the trade-off between reward and entropy. A high temperature encourages exploration; a low temperature makes the policy more deterministic. Modern SAC implementations tune alpha automatically so that the policy entropy stays close to a target value.

Because SAC is off-policy, it stores transitions in a replay buffer and can reuse old experience many times. This makes it far more sample-efficient than on-policy methods such as PPO, which is important when every sample comes from a slow simulator or a real robot.

Typical applications include robotic manipulation, locomotion and vehicle control tasks such as eco-driving, where the action is a continuous acceleration command.
