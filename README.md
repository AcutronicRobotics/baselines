# Baselines

Baselines is a fork of OpenAI's baselines repository with a set of high-quality implementations of reinforcement learning algorithms. The algorithms modified to be used in robotics.

| Algorithm | Status | Action space | Examples |
| -------- | ------- | -------- | -------- |
|[A2C](baselines/a2c) |  | discrete | |
|[A3C](baselines/a3c) |  | discrete | |
|[ACKTR](baselines/acktr) | **working** | continuous, discrete | [continuous](https://github.com/erlerobot/ros_rl/tree/master/examples/modular_scara_3dof_v3/train_acktr.py) |  
|[DDPG](baselines/ddpg) | **working** | continuous | |
|[DQN](baselines/deepq) | **working** | discrete | |
|[DQN+NAF](https://github.com/erlerobot/continuous-deep-q-learning) | **working** (requires Tensorflow `1.3.0`) | continuous | |
|[PPO](baselines/ppo1) | **working** | continuous | [continuous](https://github.com/erlerobot/ros_rl/tree/master/examples/modular_scara_3dof_v3/train_ppo1.py) |
|[TRPO](baselines/trpo_mpi) | **working** | continuous | |
|[VPG](baselines/vpg) | **not working** | continuous | [MountainCarContinuous](baselines/vpg/train_mountaincarcontinuous.py) |


## Install

```bash
cd baselines
pip install -e .
```
