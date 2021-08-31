# DDPG

DDPG - Deep Deterministic Policy Gradient algorithm.

## References
* "Continuous control with deep reinforcement learning", Lillicrap et al. [Link](https://arxiv.org/abs/1509.02971).

## Tested on

* [Pendulum](https://gym.openai.com/envs/Pendulum-v0/) - Swing up a pendulum.

<p align="center">
Cumulative Reward (total reward collected during episode) vs Episode, during training.
</p>
<p align="center">
<img src="media/pendulum_train.png" width="75%" height="75%"/>
</p>

<p align="center">
Evaluate agent performance every 1000 episodes during training. Each time calculate average cumulative reward over 100 episodes.
</p>
<p align="center"> 
<img src="media/pendulum_eval.png" width="75%" height="75%"/>
</p>

<p align="center">
Trained agent
</p>
<p align="center">
<img src="media/pendulum_trained.gif" width="50%" height="50%"/>
</p>

* [Bipedal Walker](https://gym.openai.com/envs/BipedalWalker-v2/) - Train a bipedal robot to walk.

<p align="center">
Cumulative Reward (total reward collected during episode) vs Episode, during training.
</p>
<p align="center">
<img src="media/bipedal_train.png" width="75%" height="75%"/>
</p>

<p align="center">
Evaluate agent performance every 500 episodes during training. Each time calculate average cumulative reward over 100 episodes.
</p>
<p align="center"> 
<img src="media/bipedal_eval.png" width="75%" height="75%"/>
</p>

<p align="center">
Trained agent
</p>
<p align="center">
<img src="media/bipedal_trained.gif" width="50%" height="50%"/>
</p>
