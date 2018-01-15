
import gym
import gym_gazebo
import logging
import numpy as np
import tensorflow as tf

from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
from baselines.deepqnaf.naf import NAF
from baselines.deepqnaf.network import Network
from baselines.deepqnaf.statistic import Statistic
from baselines.deepqnaf.exploration import OUExploration, BrownianExploration, LinearDecayExploration

# from naf import NAF
# from network import Network
# from statistic import Statistic
# from exploration import OUExploration, BrownianExploration, LinearDecayExploration

from baselines.deepqnaf.utils import get_model_dir, preprocess_conf


def learn (env,
            noise = 'ou',
            noise_scale = 0.2,
            hidden_dims = [100,100],
            use_batch_norm = True,
            use_seperate_networks = False,
            hidden_w = 'uniform_big',
            action_w = 'uniform_big',
            hidden_fn = 'tanh',
            action_fn = 'tanh',
            w_reg = None,
            clip_action = False,
            tau = 0.001,
            discount = 0.99,
            learning_rate = 0.001,
            batch_size = 100,
            max_steps = 200,
            update_repeat = 5,
            max_episodes = 1000):
  # set random seed
  tf.set_random_seed(123)
  np.random.seed(123)


  with tf.Session() as sess:
    # environment
    env = gym.make(env)
    env._seed(123)

    assert isinstance(env.observation_space, gym.spaces.Box), \
      "observation space must be continuous"
    assert isinstance(env.action_space, gym.spaces.Box), \
      "action space must be continuous"

    # exploration strategy
    if noise == 'ou':
      strategy = OUExploration(env, sigma=noise_scale)
    elif noise == 'brownian':
      strategy = BrownianExploration(env, noise_scale)
    elif noise == 'linear_decay':
      strategy = LinearDecayExploration(env)
    else:
      raise ValueError('Unkown exploration strategy: %s' % noise)

    # networks
    shared_args = {
      'sess': sess,
      'input_shape': env.observation_space.shape,
      'action_size': env.action_space.shape[0],
      'hidden_dims': hidden_dims,
      'use_batch_norm': use_batch_norm,
      'use_seperate_networks': use_seperate_networks,
      'hidden_w':hidden_w, 'action_w': action_w,
      'hidden_fn': hidden_fn, 'action_fn': action_fn,
      'w_reg': w_reg,
    }

    # logger.info("Creating prediction network...")
    pred_network = Network(
      scope='pred_network', **shared_args
    )

    # logger.info("Creating target network...")
    target_network = Network(
      scope='target_network', **shared_args
    )
    target_network.make_soft_update_from(pred_network, tau)

    # statistic
    #stat = Statistic(sess, env, model_dir, pred_network.variables, update_repeat)

    # agent = NAF(sess, env, strategy, pred_network, target_network, stat,
    #             discount, batch_size, learning_rate,
    #             max_steps, update_repeat, max_episodes)

    agent = NAF(sess, env, strategy, pred_network, target_network,
                discount, batch_size, learning_rate,
                max_steps, update_repeat, max_episodes)
    #agent.run(conf.monitor, conf.display, conf.is_train)
    agent.run(False, False, True)
    #agent.run2(conf.monitor, conf.display, True)
