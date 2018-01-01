import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

import sklearn.pipeline
import sklearn.preprocessing
# import plotting
from baselines import logger

from sklearn.kernel_approximation import RBFSampler

"""
Vanilla Policy Gradient implemented using Actor-Critic
method to reduce the variance.

This algorithm implements an Actor Critic Model (ACM)
which separates the policy from the value approximation process by
parameterizing the policy separately.

Pseudocode:
1. Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
2. For iteration=1,2,... do
    2.1 Collect a set of trajectories by executing the current policy obtaining $\mathbf{s}_{0:H},\mathbf{a}_{0:H},r_{0:H}$
    2.2 At each timestep in each trajectory, compute
        2.2.1 the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$ and
        2.2.2 the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
    2.3 Re-fit the baseline (recomputing the value function) by minimizing
        $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.

          $b=\frac{\left\langle \left(  \sum\nolimits_{h=0}^{H} \mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert \mathbf{s}_{h}\right.  \right)  \right)  ^{2}\sum\nolimits_{l=0}^{H} \gamma r_{l}\right\rangle }{\left\langle \left(
          \sum\nolimits_{h=0}^{H}\mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}
          }\left(  \mathbf{a}_{h}\left\vert \mathbf{x}_{h}\right.  \right)  \right)
          ^{2}\right\rangle }$

    2.4 Update the policy, using a policy gradient estimate $\hat{g}$,
        which is a sum of terms $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$.
        In other words:

          $g_{k}=\left\langle \left(  \sum\nolimits_{h=0}^{H}\mathbf{\nabla
          }_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert
          \mathbf{s}_{h}\right.  \right)  \right)  \left(  \sum\nolimits_{l=0}^{H}
          \gamma r_{l}-b\right)  \right\rangle$
3. **end for**
"""

def preprocess(env):
    # Feature Preprocessing: Normalize to zero mean and unit variance
    # We use a few samples from the observation space to do this
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    # TODO: read more about this
    featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
            ])
    featurizer.fit(scaler.transform(observation_examples))
    # return both primitives to be used within the code
    return scaler, featurizer

def featurize_state(state, env):
    """
    Returns the featurized representation for a state.
    """
    # print("state ", state)
    scaled = scaler.transform([state])
    # print("scaled ", scaled)
    featurized = featurizer.transform(scaled)
    # print("featurized ", featurized)
    return featurized[0]

class PolicyEstimator():
    """
    Policy Function approximator. Actor.
    """

    def __init__(self, env, learning_rate=0.01, scope="policy_estimator"):
        self.env = env
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.mu = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)
            self.mu = tf.squeeze(self.mu)

            self.sigma = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.sigma = tf.squeeze(self.sigma)
            self.sigma = tf.nn.softplus(self.sigma) + 1e-5
            self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
            self.action = self.normal_dist._sample_n(1)
            self.action = tf.clip_by_value(self.action, env.action_space.low[0], env.action_space.high[0])

            # Loss and train op
            self.loss = -self.normal_dist.log_prob(self.action) * self.target
            # Add cross entropy cost to encourage exploration
            self.loss -= 1e-1 * self.normal_dist.entropy()

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state, self.env)
        return sess.run(self.action, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state, self.env)
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

class ValueEstimator():
    """
    Value Function approximator. Critic.
    """

    def __init__(self, env, learning_rate=0.1, scope="value_estimator"):
        self.env = env
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state, self.env)
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        state = featurize_state(state, self.env)
        feed_dict = { self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.
    Parameters
    ----------
    env: object
        OpenAI environment.
    estimator_policy: object
        Policy Function to be optimized
    estimator_value: object
        Value function approximator, used as a critic
    num_episodes: int
        Number of episodes to run for
    discount_factor: float
        Time-discount factor
    Returns
    -------
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()
        episode = []
        # One step in the environment
        for t in itertools.count():
            # env.render()
            # Take a step
            action = estimator_policy.predict(state)
            next_state, reward, done, _ = env.step(action)
            # Keep track of the transition
            episode.append(Transition(
              state=state, action=action, reward=reward, next_state=next_state, done=done))
            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t
            # Calculate TD Target
            value_next = estimator_value.predict(next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - estimator_value.predict(state)
            # Update the value estimator
            estimator_value.update(state, td_target)
            # Update the policy estimator
            # using the td error as our advantage estimate
            estimator_policy.update(state, td_error, action)
            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            if done:
                break
            state = next_state
    return stats

def learn(env, estimator_policy, estimator_value,
            max_timesteps=1000,
            discount_factor=1.0,
            print_freq=100,
            outdir="/tmp/experiments/continuous/VPG/"):
    """
    Vanilla Policy Gradient (VPG) extended using basic Actor-Critic techniques to reduce the variance.
    This method optimizes the value function approximator using policy gradient.

    Parameters
    ----------
    env: object
        OpenAI environment.
    estimator_policy: object
        Policy Function to be optimized
    estimator_value: object
        Value function approximator, used as a critic
    max_timesteps: int
        Number of steps to run for
    discount_factor: float
        Time-discount factor (gamma)
    print_freq: int
        Period (in episodes) to log results
    outdir: string
        Directory where to store tensorboard results

    Returns
    -------
    An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
                        # 1. Initialize policy (e.g. NNs) parameter $\theta$ and baseline $b$
                        #       in this particular case, they come as params
    # tensorboard logging
    summary_writer = tf.summary.FileWriter(outdir, graph=tf.get_default_graph())
    # Keeps track of useful statistics
    # stats = plotting.EpisodeStats(
    #     episode_lengths=np.zeros(num_episodes),
    #     episode_rewards=np.zeros(num_episodes))

    # # Variable to represent the number of steps executed
    # Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    global scaler, featurizer
    scaler, featurizer = preprocess(env)

    # Record number of episodes
    num_episodes = 0
    # Reset the environment and get firs state
    state = env.reset()
    # each episode's reward
    episode_reward = 0
                        # 2. For iteration=1,2,... do
                        #     2.1 Collect a set of trajectories by executing the current policy obtaining $\mathbf{s}_{0:H},\mathbf{a}_{0:H},r_{0:H}$
    for timestep in range(max_timesteps):
                        #     2.2 At each timestep in each trajectory, compute
        # episode = []
        # One step in the environment
        # for t in itertools.count():

        # env.render()
        action = estimator_policy.predict(state)
        next_state, reward, done, _ = env.step(action)

        # # Keep track of the transition
        # episode.append(Transition(
        #   state=state, action=action, reward=reward, next_state=next_state, done=done))

        # Update statistics
        # stats.episode_rewards[num_episodes] += reward
        episode_reward += reward
        # stats.episode_lengths[num_episodes] = timestep

        # Calculate TD Target
        #   More about TD-learning at:
            # http://www.scholarpedia.org/article/Reinforcement_learning
            # http://www.scholarpedia.org/article/TD-learning

        # calculate the bias b
        value_next = estimator_value.predict(next_state)
                        #         2.2.1 the return $R_t = \sum_{t'=t}^{T-1} \gamma^{t'-t}r_{t'}$ and
        td_target = reward + discount_factor * value_next
                        #         2.2.2 the advantage estimate $\hat{A_t} = R_t - b(s_t)$.
        td_error = td_target - estimator_value.predict(state)
                        #     2.3 Re-fit the baseline (recomputing the value function) by minimizing
                        #         $|| b(s_t) - R_t||^2$, summed over all trajectories and timesteps.
                        #
                        #           $b=\frac{\left\langle \left(  \sum\nolimits_{h=0}^{H} \mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert \mathbf{s}_{h}\right.  \right)  \right)  ^{2}\sum\nolimits_{l=0}^{H} \gamma r_{l}\right\rangle }{\left\langle \left(
                        #           \sum\nolimits_{h=0}^{H}\mathbf{\nabla}_{\theta_{k}}\log\pi_{\mathbf{\theta}
                        #           }\left(  \mathbf{a}_{h}\left\vert \mathbf{x}_{h}\right.  \right)  \right)
                        #           ^{2}\right\rangle }$
        # Update the value estimator
        estimator_value.update(state, td_target)
                        #     2.4 Update the policy, using a policy gradient estimate $\hat{g}$,
                        #         which is a sum of terms $\nabla_\theta log\pi(a_t | s_t,\theta)\hat(A_t)$.
                        #         In other words:
                        #
                        #           $g_{k}=\left\langle \left(  \sum\nolimits_{h=0}^{H}\mathbf{\nabla
                        #           }_{\theta_{k}}\log\pi_{\mathbf{\theta}}\left(  \mathbf{a}_{h}\left\vert
                        #           \mathbf{s}_{h}\right.  \right)  \right)  \left(  \sum\nolimits_{l=0}^{H}
                        #           \gamma r_{l}-b\right)  \right\rangle$
        # Update the policy estimator
        # using the td error as our advantage estimate
        estimator_policy.update(state, td_error, action)

        # # Print out which step we're on, useful for debugging.
        # print("\rStep {} @ Episode {} ({})".format(
        #         timestep + 1, num_episodes, episode_reward), end="")

        if done:
            # Log the episode reward
            # episode_total_rew = stats.episode_rewards[num_episodes]
            summary = tf.Summary(value=[tf.Summary.Value(tag="Episode reward",
                simple_value = episode_reward)])
            summary_writer.add_summary(summary, timestep)
            summary_writer.flush()

            # Reset the environment and get firs state
            state = env.reset()

            if print_freq is not None and num_episodes % print_freq == 0:
                logger.record_tabular("steps", timestep)
                logger.record_tabular("episode", num_episodes)
                logger.record_tabular("reward", episode_reward)
                # logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            # Iterate episodes
            num_episodes +=1

            # Reset the episode reward
            episode_reward = 0
        else:
            state = next_state
                        # 3. **end for**
    return estimator_policy

def act(observation):
    """
    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph

    Returns
    ----------
    action: object
        An action for the environment
    """
    # TODO: implement if necessary
    pass
