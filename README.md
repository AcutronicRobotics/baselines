# Baselines

Baselines is a fork of OpenAI's baselines repository with a set of high-quality implementations of reinforcement learning algorithms. The algorithms modified to be used in robotics.

| Algorithm | Status | Action space | Examples |
| -------- | ------- | -------- | -------- |
|[A2C](baselines/a2c) | WIP | discrete | |
|[A3C](baselines/a3c) | WIP | discrete, continious (WIP) | |
|[ACKTR](baselines/acktr) | **working** | continuous, discrete | [continuous](https://github.com/erlerobot/ros_rl/tree/master/examples/modular_scara_3dof_v3/train_acktr.py) |  
|[DDPG](baselines/ddpg) | | | |
|[DQN](baselines/deepq) | WIP | | |
|[PPO](baselines/ppo1) | **working** | continuous | [continuous](https://github.com/erlerobot/ros_rl/tree/master/examples/modular_scara_3dof_v3/train_ppo1.py) |
|[TRPO](baselines/trpo_mpi) | | | |


## Install

```bash
cd baselines
pip install -e .
```
