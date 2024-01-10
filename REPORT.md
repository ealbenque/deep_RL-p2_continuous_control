# Report

## Implementation
The resolution of the environment involves the utilization of a deep reinforcement learning agent.
The corresponding implementation is available in the continuous_control folder :
* continuous_control.ipynb contains the main code with the training loop and results
* agent.py contains the reinforcement learning agent
* model.py includes the neural networks serving as the estimators
* the weights folder contains the saved weight of the different neural networks.

The starting point for the code was borrowed from the Udacity ddpg-pendulum exercise and subsequently modified to suit the requirements of this specific problem.

## Learning algorithm
The learning algorithm used here is [DDPG](https://arxiv.org/abs/1509.02971).

DDPG, or Deep Deterministic Policy Gradient, is a reinforcement learning algorithm designed for solving continuous action space problems. It combines ideas from deep learning and policy gradient methods to handle high-dimensional state and action spaces.
Here's a brief explanation of the key components and concepts of DDPG:
* DDPG employs an actor-critic architecture, where the actor is responsible for learning a deterministic policy, mapping states to specific actions, and the critic evaluates the state-action pairs.
* Deterministic Policy: the actor in DDPG learns a deterministic policy, meaning it directly outputs the action to be taken given a particular state. This is in contrast to stochastic policies that output a probability distribution over actions.
* Experience Replay Buffer: DDPG uses an experience replay buffer to store past experiences like in DQN algorithms (tuples of state, action, reward, next state) for training. This helps in breaking the temporal correlations in the data and provides more stable learning.
* Target Networks: to stabilize training, DDPG uses target networks for both the actor and the critic. These are copies of the original networks that are slowly updated over time using a soft update mechanism.
* Q-function Approximation: the critic approximates the action-value function (Q-function), which estimates the expected cumulative reward of taking a specific action in a given state and following a particular policy.
* Policy Gradient: DDPG uses the policy gradient method to update the actor network. The gradient is computed with respect to the expected return and is used to adjust the actor's parameters in the direction that increases the expected return.
* Target Q-value: The target Q-value is used to update the critic network. It is computed using the Bellman equation and is used as a target for the critic's Q-value prediction.
By combining these elements, DDPG is able to learn a deterministic policy for continuous action spaces.

## Hyperparameters
The following hyperparameters were used:
* replay buffer size: 1e5
* max timesteps: 1000
* batch size: 128
* discount factor: 0.99
* tau (soft update for target networks factor): 1e-3
* learning rate actor: 1e-3
* learning rate critic : 1e-3

## Neural networks
The actor model is a simple feedforward network: maps state to action
* Input layer: 33  neurons (the state size)
* 1st hidden layer: fully connected, 400 neurons with ReLu activation
* 2nd hidden layer: fully connected, 300 neurons with ReLu activation
* output layer: 4 neurons (1 for each action) (tanh)

The critic model: maps state action pairs to value
* Batch normalization
* Input layer: 33 neurons (the state size) + 4 neurons (1 for each actions)
* 1st hidden layer: fully connected, 400 neurons with ReLu activation
* 2nd hidden layer: fully connected, 300 neurons with ReLu activation
* output layer: 1 neuron

## Results
The agent was able to solve the environment after 108 episodes achieving an average score over 30 over the last 100 episodes
of the training.

The average scores of the 20 agents during the training process:
![plot_01](https://github.com/ealbenque/demo-repo/assets/137990986/1de45239-0c0f-41d4-aa59-2d94a799e177)

## Possible improvements
Possible improvements include :
- PPO to address issues related to policy optimization instability
- D4PG that implement experience sharing among multiple agents to improve sample efficiency
