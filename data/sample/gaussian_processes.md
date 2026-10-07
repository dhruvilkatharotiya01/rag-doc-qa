# Gaussian Process Regression

A Gaussian process (GP) is a distribution over functions. Any finite set of function values is jointly Gaussian, fully described by a mean function and a covariance function, also called the kernel.

The kernel encodes assumptions about the function. The squared exponential (RBF) kernel produces smooth functions; its length-scale hyperparameter controls how quickly correlation decays with distance. The Matern kernel allows rougher functions and is often a better default for physical systems.

Given training data, Gaussian Process Regression returns a full predictive distribution for each new input: a predicted mean and a predictive variance. The variance grows in regions far from the training data, so a GP tells you not only what it predicts but how uncertain it is. This calibrated uncertainty is the main reason GPs are used in safety-relevant control and in Bayesian optimisation.

Kernel hyperparameters and the noise level are usually fitted by maximising the log marginal likelihood. Exact inference requires inverting an n by n covariance matrix, so the cost grows cubically with the number of training points. Sparse approximations with inducing points reduce this cost for larger datasets.

In model-based reinforcement learning, a GP can serve as a learned dynamics or traffic model, and its uncertainty estimate can be used to avoid actions in poorly understood states.
