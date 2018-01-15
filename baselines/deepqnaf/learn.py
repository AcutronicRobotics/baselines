import gym
import logging
import numpy as np
import tensorflow as tf
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from .naf import NAF
from .network import Network
from .statistic import Statistic
from .exploration import OUExploration, BrownianExploration, LinearDecayExploration
from .utils import get_model_dir, preprocess_conf


def learn(env,
            sess,
            noise = 'ou',
            noise_scale = 0.2,
            hidden_dims = [64,64],
            use_batch_norm = False,
            use_seperate_networks = False,
            hidden_w = tf.random_uniform_initializer(-0.05, 0.05),
            action_w = tf.random_uniform_initializer(-0.05, 0.05),
            hidden_fn = tf.nn.tanh,
            action_fn = tf.nn.tanh,
            w_reg = None,
            clip_action = False,
            tau = 0.001,
            discount = 0.99,
            learning_rate = 0.001,
            batch_size = 100,
            max_steps = 200,
            update_repeat = 5,
            max_episodes = 1000,
            outdir="/tmp/rosrl/experiments/continuous/deepqnaf/"):

    # # set random seed
    # tf.set_random_seed(123)
    # np.random.seed(123)

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
    #             max_steps, update_repeat, max_episodes, outdir)

    agent = NAF(sess, env, strategy, pred_network, target_network,
                discount, batch_size, learning_rate,
                max_steps, update_repeat, max_episodes, outdir)
    #agent.run(conf.monitor, conf.display, conf.is_train)
    agent.run(False, False, True)
    #agent.run2(conf.monitor, conf.display, True)
