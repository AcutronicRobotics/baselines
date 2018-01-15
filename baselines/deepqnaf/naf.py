from logging import getLogger
logger = getLogger(__name__)
import logging
import os
import gym
from baselines import logger, bench
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.contrib.framework import get_variables

from .utils_time import get_timestamp

class NAF(object):
  # def __init__(self, sess,
  #              env, strategy, pred_network, target_network, stat,
  #              discount, batch_size, learning_rate,
  #              max_steps, update_repeat, max_episodes):
  def __init__(self, sess,
               env, strategy,
               pred_network, target_network,
               discount, batch_size,
               learning_rate,
               max_steps, update_repeat, max_episodes,
               outdir):

    self.outdir = outdir
    self.sess = sess
    self.env = env
    self.strategy = strategy
    self.pred_network = pred_network
    self.target_network = target_network
    #self.stat = stat

    self.discount = discount
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.action_size = env.action_space.shape[0]

    self.max_steps = max_steps
    self.update_repeat = update_repeat
    self.max_episodes = max_episodes

    self.prestates = []
    self.actions = []
    self.rewards = []
    self.poststates = []
    self.terminals = []

    with tf.name_scope('optimizer'):
      self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
      self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(self.pred_network.Q)), name='loss')

      self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

  def run(self, monitor=False, display=False, is_train=True):
    #self.stat.load_model()
    #self.target_network.hard_copy_from(self.pred_network)

    # given by self.outdir
    # logdir = '/tmp/rosrl/' + str(self.env.__class__.__name__) +'/deepq_naf/monitor/'
    # outdir = '/tmp/rosrl/' + str(self.env.__class__.__name__) +'/deepq_naf/tensorboard'

    logdir = self.outdir
    logger.configure(os.path.abspath(logdir))
    # TODO: review this and the modifcations introduced in MonitorRobotics to make it work
    env = bench.MonitorRobotics(self.env, logger.get_dir() and os.path.join(logger.get_dir()),
            allow_early_resets=True, robotics=False)
    gym.logger.setLevel(logging.WARN)


    summary_writer_ep_mean = tf.summary.FileWriter('/tmp/rosrl/' + str(self.env.__class__.__name__) +'/deepq_naf/tensorboard/ep_mean/', graph=tf.get_default_graph())
    summary_writer_ep_mean_100 = tf.summary.FileWriter('/tmp/rosrl/' + str(self.env.__class__.__name__) +'/deepq_naf/tensorboard/ep_mean_100/', graph=tf.get_default_graph())

    # Create the tensorboard writer
    summary_writer = tf.summary.FileWriter(self.outdir, graph=tf.get_default_graph())

    #if monitor:
      #self.env.monitor.start('/tmp/%s-%s' % (self.stat.env_name, get_timestamp()))
    init = tf.global_variables_initializer() ###
    self.sess.run(init) ###
    episode_rewards = []
    episode_rewards_100 = deque(maxlen=100)
    episodes_solved = 0
    it = 0
    display = False

    for self.idx_episode in range(self.max_episodes):
          print("Episode", self.idx_episode)
          state = env.reset()
          #for t in xrange(0, self.max_steps):
          episode_reward = 0
          episodes_solved = 0

          for t in range(0, self.max_steps):
            #it = it + 1
            if display: self.env.render()

            # 1. predict
            action = self.predict(state)

            # 2. step
            self.prestates.append(state)
            state, reward, terminal, _ = self.env.step(action)

            # Tensorboard, capture per-episode rewards
            summary = tf.Summary(value=[tf.Summary.Value(tag="Episode Reward", simple_value = reward)])
            summary_writer.add_summary(summary, it)

            self.poststates.append(state)

            terminal = True if t == self.max_steps - 1 else terminal

            # 3. perceive
            # TODO: Nora, please include the update_repeat=5 in the loop
            if is_train:

              #q, v, a, l, it = self.perceive(state, reward, action, terminal, it)
              q, v, a, l, it = self.perceive(state, reward, action, terminal, it, episode_rewards, episode_rewards_100)

              #if self.stat:
                #self.stat.on_step(action, reward, terminal, q, v, a, l)
            episode_reward += reward
            if terminal:
                # print("Episode_reward", episode_reward)
                episode_rewards.append(episode_reward)
                episode_rewards_100.append(episode_reward)
                ep_rew_mean = np.mean(episode_rewards)
                ep_rew_mean_100 = np.mean(episode_rewards_100)
                # print("EpRewMean", np.mean(episode_rewards))
                # print("EpRewStd", np.std(episode_rewards))
                # logger.info("REWARDS")
                # logger.record_tabular("EpRewMean", np.mean(episode_rewards))
                # logger.record_tabular("EpRewStd", np.std(episode_rewards))
                # logger.record_tabular("TimestepsSoFar", it)
                # logger.dump_tabular()
                # # Create the writer for TensorBoard logs
                # summary = tf.Summary(value=[tf.Summary.Value(tag="EpRewMean", simple_value = ep_rew_mean)])
                # print("It tensorboard", it)
                # summary_writer.add_summary(summary, it)
                summary_ep_mean = tf.Summary(value=[tf.Summary.Value(tag="EpRewMean", simple_value = ep_rew_mean)])
                summary_writer_ep_mean.add_summary(summary_ep_mean, it)
                summary_writer_ep_mean.flush()
                summary_ep_mean_100 = tf.Summary(value=[tf.Summary.Value(tag="EpRewMean100", simple_value = ep_rew_mean_100)])
                summary_writer_ep_mean_100.add_summary(summary_ep_mean_100, it)
                summary_writer_ep_mean_100.flush()
                episode_reward = 0
                episodes_solved +=1
                observation = self.env.reset()
                self.strategy.reset()
              #break

    #if monitor:
      #self.env.monitor.close()

  def run2(self, monitor=False, display=False, is_train=True):
    target_y = tf.placeholder(tf.float32, [None], name='target_y')
    loss = tf.reduce_mean(tf.squared_difference(target_y, tf.squeeze(self.pred_network.Q)), name='loss')

    optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    # self.stat.load_model()
    # self.target_network.hard_copy_from(self.pred_network)

    # replay memory
    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []
    episode_rewards = []
    episodes_solved = 0
    init = tf.global_variables_initializer()###
    self.sess.run(init)###
    # the main learning loop
    total_reward = 0


    for i_episode in range(self.max_episodes):


      # Create the writer for TensorBoard logs
      summary_writer_ep_mean = tf.summary.FileWriter(outdir, graph=tf.get_default_graph())
      summary_writer_ep_mean_100 = tf.summary.FileWriter(outdir, graph=tf.get_default_graph())

      observation = self.env.reset()
      episode_reward = 0
      episodes_solved = 0
      for t in range(self.max_steps):
        print("Timestep", t)
        if display:
          self.env.render()

        # predict the mean action from current observation
        x_ = np.array([observation])
        u_ = self.pred_network.mu.eval({self.pred_network.x: x_})[0]

        action = u_ + np.random.randn(1) / (i_episode + 1)

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = self.env.step(action)
        episode_reward += reward

        rewards.append(reward); poststates.append(observation); terminals.append(done)

        if len(prestates) > 10:
          loss_ = 0
          for k in range(self.update_repeat):
            if len(prestates) > self.batch_size:
              indexes = np.random.choice(len(prestates), size=self.batch_size)
            else:
              indexes = range(len(prestates))

            # Q-update
            v_ = self.target_network.V.eval({self.target_network.x: np.array(poststates)[indexes]})
            y_ = np.array(rewards)[indexes] + self.discount * np.squeeze(v_)

            tmp1, tmp2 = np.array(prestates)[indexes], np.array(actions)[indexes]
            loss_ += l_

            self.target_network.soft_update_from(self.pred_network)

        if done:
            episode_rewards.append(episode_reward)
            episodes_solved +=1
            observation = self.env.reset()
          #break

    #   print ("average loss:", loss_/k)
      print ("Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward))
      total_reward += episode_reward
      ep_rew_mean = np.mean(episode_rewards)
    #   print("EpRewMean", np.mean(episode_rewards))
    #   print("EpRewStd", np.std(episode_rewards))
      logger.record_tabular("EpRewMean", np.mean(episode_rewards))
      logger.record_tabular("EpRewStd", np.std(episode_rewards))

      # Log in Tensorboard
      summary = tf.Summary(value=[tf.Summary.Value(tag="EpRewMean", simple_value = ep_rew_mean)])
      summary_writer.add_summary(summary, timesteps_so_far)

    total_episodes.append(episodes_solved)
    episodes_solved = 0
    print ("Average reward per episode {}".format(total_reward / self.episodes))

  def predict(self, state):
    u = self.pred_network.predict([state])[0]

    return self.strategy.add_noise(u, {'idx_episode': self.idx_episode})

  #def perceive(self, state, reward, action, terminal, it):
  def perceive(self, state, reward, action, terminal, it, episode_rewards, episode_rewards_100):
    self.rewards.append(reward)
    self.actions.append(action)

    #return self.q_learning_minibatch(it)
    return self.q_learning_minibatch(it, episode_rewards, episode_rewards_100)


  #def q_learning_minibatch(self, it):
  def q_learning_minibatch(self, it, episode_rewards, episode_rewards_100):
    q_list = []
    v_list = []
    a_list = []
    l_list = []

    #for iteration in xrange(self.update_repeat):
    for iteration in range(self.update_repeat):

      it = it + 1
      if len(episode_rewards) > 0:
          logger.record_tabular("EpRewMean", np.mean(episode_rewards))
          logger.record_tabular("EpRewStd", np.std(episode_rewards))
          logger.record_tabular("EpRewMean100", np.mean(episode_rewards_100))
          logger.record_tabular("EpRewStd100", np.std(episode_rewards_100))
          logger.record_tabular("TimestepsSoFar", it)
          logger.dump_tabular()


      if len(self.rewards) >= self.batch_size:
        indexes = np.random.choice(len(self.rewards), size=self.batch_size)
      else:
        indexes = np.arange(len(self.rewards))

      x_t = np.array(self.prestates)[indexes]
      x_t_plus_1 = np.array(self.poststates)[indexes]
      r_t = np.array(self.rewards)[indexes]
      u_t = np.array(self.actions)[indexes]

      v = self.target_network.predict_v(x_t_plus_1, u_t)
      target_y = self.discount * np.squeeze(v) + r_t

      _, l, q, v, a = self.sess.run([
        self.optim, self.loss,
        self.pred_network.Q, self.pred_network.V, self.pred_network.A,
      ], {
        self.target_y: target_y,
        self.pred_network.x: x_t,
        self.pred_network.u: u_t,
        self.pred_network.is_train: True,
      })

      q_list.extend(q)
      v_list.extend(v)
      a_list.extend(a)
      l_list.append(l)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("q: %s, v: %s, a: %s, l: %s" \
        % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))

    return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list), it
