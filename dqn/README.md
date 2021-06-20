# DQN

DQN - Deep Q learning with experience replay

References
1) Playing Atari with Deep Reinforcement Learning [link](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
2) Human-level control through deep reinforcement learning [link](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)

Trained models on
1) [Mountain Car ](https://gym.openai.com/envs/MountainCar-v0/) - A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
<img src="media/mountain_car_v0_trained.gif" width="50%" height="50%"/>
2) [Cart Pole](https://gym.openai.com/envs/CartPole-v1/) - A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.
<img src="media/cartpole_v1_trained.gif" width="50%" height="50%"/>