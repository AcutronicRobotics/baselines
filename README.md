**Status:** Active (under active development, breaking changes may occur)

[![Build status](https://travis-ci.org/AcutronicRobotics/baselines.svg?branch=master)](https://travis-ci.org/AcutronicRobotics/baselines)

# Baselines

Baselines is a fork of OpenAI's baselines repository with a set of high-quality implementations of reinforcement learning algorithms. The algorithms modified to be used in robotics.

| Algorithm | Status | Action space | Examples |
| -------- | ------- | -------- | -------- |
|[A2C](baselines/a2c) |  | discrete | |
|[A3C](baselines/a3c) |  | discrete | |
|[ACKTR](baselines/acktr) | **working** | continuous, discrete | [continuous](https://github.com/AcutronicRobotics/ros2learn/tree/master/examples/modular_scara_3dof_v3/train_acktr.py) |
|[DDPG](baselines/ddpg) | **working** | continuous | |
|[DQN](baselines/deepq) | **working** | discrete | |
|[DQN+NAF](https://github.com/erlerobot/continuous-deep-q-learning) | **working** (requires Tensorflow `1.3.0`) | continuous | |
|[PPO](baselines/ppo1) | **working** | continuous | [continuous](https://github.com/AcutronicRobotics/ros2learn/tree/master/examples/modular_scara_3dof_v3/train_ppo1.py) |
|[TRPO](baselines/trpo_mpi) | **working** | continuous | |
|[VPG](baselines/vpg) | **not working** | continuous | [MountainCarContinuous](baselines/vpg/train_mountaincarcontinuous.py) |

## Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows
### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

## Virtual environment
From the general python package sanity perspective, it is a good idea to use virtual environments (virtualenvs) to make sure packages from different projects do not interfere with each other. You can install virtualenv (which is itself a pip package) via
```bash
pip install virtualenv
```
Virtualenvs are essentially folders that have copies of python executable and all python packages.
To create a virtualenv called venv with python3, one runs
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv:
```
. /path/to/venv/bin/activate
```
More thorough tutorial on virtualenvs and options can be found [here](https://virtualenv.pypa.io/en/stable/)

## Install

## Installation
- Clone the repo and cd into it:
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```
- If you don't have TensorFlow installed already, install your favourite flavor of TensorFlow. In most cases,
    ```bash
    pip install tensorflow-gpu # if you have a CUDA-compatible gpu and proper drivers
    ```
    or
    ```bash
    pip install tensorflow
    ```
    should be sufficient. Refer to [TensorFlow installation guide](https://www.tensorflow.org/install/)
    for more details.

- Install baselines package
    ```bash
    pip install -e .
    ```

### MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest
pytest
```

## Training models
Most of the algorithms in baselines repo are used as follows:
```bash
python -m baselines.run --alg=<name of the algorithm> --env=<environment_id> [additional arguments]
```
### Example 1. PPO with MuJoCo Humanoid
For instance, to train a fully-connected network controlling MuJoCo humanoid using PPO2 for 20M timesteps
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7
```
Note that for mujoco environments fully-connected network is default, so we can omit `--network=mlp`
The hyperparameters for both network and the learning algorithm can be controlled via the command line, for instance:
```bash
python -m baselines.run --alg=ppo2 --env=Humanoid-v2 --network=mlp --num_timesteps=2e7 --ent_coef=0.1 --num_hidden=32 --num_layers=3 --value_network=copy
```
will set entropy coefficient to 0.1, and construct fully connected network with 3 layers with 32 hidden units in each, and create a separate network for value function estimation (so that its parameters are not shared with the policy network, but the structure is the same)

See docstrings in [common/models.py](baselines/common/models.py) for description of network parameters for each type of model, and
docstring for [baselines/ppo2/ppo2.py/learn()](baselines/ppo2/ppo2.py#L152) for the description of the ppo2 hyperparameters.

### Example 2. DQN on Atari
DQN with Atari is at this point a classics of benchmarks. To run the baselines implementation of DQN on Atari Pong:
```
python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --num_timesteps=1e6
```

## Saving, loading and visualizing models
The algorithms serialization API is not properly unified yet; however, there is a simple method to save / restore trained models.
`--save_path` and `--load_path` command-line option loads the tensorflow state from a given path before training, and saves it after the training, respectively.
Let's imagine you'd like to train ppo2 on Atari Pong,  save the model and then later visualize what has it learnt.
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=2e7 --save_path=~/models/pong_20M_ppo2
```
This should get to the mean reward per episode about 20. To load and visualize the model, we'll do the following - load the model, train it for 0 steps, and then visualize:
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=0 --load_path=~/models/pong_20M_ppo2 --play
```

*NOTE:* Mujoco environments require normalization to work properly, so we wrap them with VecNormalize wrapper. Currently, to ensure the models are saved with normalization (so that trained models can be restored and run without further training) the normalization coefficients are saved as tensorflow variables. This can decrease the performance somewhat, so if you require high-throughput steps with Mujoco and do not need saving/restoring the models, it may make sense to use numpy normalization instead. To do that, set 'use_tf=False` in [baselines/run.py](baselines/run.py#L116). 

## Loading and vizualizing learning curves and other training metrics
See [here](docs/viz/viz.ipynb) for instructions on how to load and display the training data.

## Subpackages

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [GAIL](baselines/gail)
- [HER](baselines/her)
- [PPO1](baselines/ppo1) (obsolete version, left here temporarily)
- [PPO2](baselines/ppo2)
- [TRPO](baselines/trpo_mpi)



## Benchmarks
Results of benchmarks on Mujoco (1M timesteps) and Atari (10M timesteps) are available
[here for Mujoco](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm)
and
[here for Atari](https://htmlpreview.github.com/?https://github.com/openai/baselines/blob/master/benchmarks_atari10M.htm)
respectively. Note that these results may be not on the latest version of the code, particular commit hash with which results were obtained is specified on the benchmarks page.

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai and Zhokhov, Peter},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
