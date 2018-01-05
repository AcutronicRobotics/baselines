import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.deepqnaf
from baselines.deepqnaf import experiment
import baselines.deepqnaf.naf as naf
from baselines.deepqnaf.experiment import learn
from baselines.deepqnaf.experiment import recursive_experiment
import baselines.common.tf_util as U

import gym
import gym_gazebo
import tensorflow as tf
from mpi4py import MPI


graph = True
render = True
env_id="InvertedPendulum-v1"
#env_id="HalfCheetah-v1"
#env_id = "GazeboModularScara3DOF-v3"
noise_type='ou_0.2'
repeats = 1
episodes = 1000
max_episode_steps = 200
train_steps = 5
learning_rate = 0.001
batch_normalize = True
#gamma = 0.8
gamma = 0.99
tau = 0.99
epsilon = 0.1
hidden_size = 100
#hidden_size = [16, 32, 200]
#hidden_size = [64, 64, 64]
hidden_n = 2
hidden_activation = tf.nn.relu
batch_size = 128
memory_capacity = 10000
load_path = None
covariance = "original"
solve_threshold = None
v = 0

# Create envs.
env = gym.make(env_id)
env.reset()

logdir = '/tmp/rosrl/' + str(env_id.__class__.__name__) +'/deepq_naf/monitor/'
logger.configure(os.path.abspath(logdir))
env = bench.MonitorRobotics(env, logger.get_dir() and os.path.join(logger.get_dir()), allow_early_resets=True) #, allow_early_resets=True
gym.logger.setLevel(logging.WARN)

rewards = learn(env,
                v,
                graph,
                render,
                repeats,
                episodes,
                max_episode_steps,
                train_steps,
                batch_normalize,
                learning_rate,
                gamma,
                tau,
                epsilon,
                hidden_size,
                hidden_n,
                hidden_activation,
                 batch_size,
                 memory_capacity,
                 load_path,
                 covariance)

