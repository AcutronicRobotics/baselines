import os
import tempfile
import pandas
import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np
import random
import gym
import baselines.common.tf_util as U
from baselines import logger
from baselines.common.schedules import LinearSchedule


from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.build_graph_robotics import build_train


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepqn.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            U.load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            U.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)

def step(env, action, state):
    """
    Implementation of "step" which uses the given action
    as an offset of the existing state. This function overloads
    the environment step method and should be used when actions-as-offsets
    are required for this particular environment.
    """
    # if action == (0.0, 0.0, 0.0):
        # print("robot decided to stay still!")
    offset_action = [a + s for a,s in zip(list(action), state)]
    # print("step: action: ", action)
    # print("step: state: ", state)
    #print("step: offset_action: ", offset_action)

    observation, reward, done, info = env.step(offset_action)
    return observation, reward, done, info


def learn(env,
          q_func,
          lr=5e-4,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          ):
    """Train a deepqn model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    lr: float
        learning rate for adam optimizer
    max_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to max_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    action_no: int
        number of actions available in action space
    actions_discr: Box space
        Discretized actions
    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """


    # Create all the functions necessary to train the model

    sess = tf.Session()
    sess.__enter__()


    #TODO This should not go here. Instead pass both action_no and actions as arguments to learn function
    #Discrete actions
    goal_average_steps = 2
    max_number_of_steps = 20
    last_time_steps = np.ndarray(0)
    n_bins = 10
    epsilon_decay_rate = 0.99 ########
    it = 1 ######


    # Number of states is huge so in order to simplify the situation
    # typically, we discretize the space to: n_bins ** number_of_features
    joint1_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    joint2_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    joint3_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]
    action_bins = pandas.cut([-np.pi/2, np.pi/2], bins=n_bins, retbins=True)[1][1:-1]

    difference_bins = abs(joint1_bins[0] - joint1_bins[1])
    actions_discr = [(5*difference_bins, 0.0, 0.0), (-5*difference_bins, 0.0, 0.0),
         (0.0, 5*difference_bins, 0.0), (0.0, -5*difference_bins, 0.0),
         (0.0, 0.0, 5*difference_bins), (0.0, 0.0, -5*difference_bins),
         (0.0, 0.0, 0.0)]
    action_no = 7
    actions = [0, 1, 2, 3, 4, 5, 6]

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space_shape = env.observation_space.shape
    def make_obs_ph(name):
        return U.BatchInput(observation_space_shape, name=name)

    #Tensorboard
    #writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    #summary_writer = tf.summary.FileWriter(logs_path,graph=tf.get_default_graph())



    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=action_no,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise
    )


    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        #'num_actions': env.action_space.n,
        'num_actions': action_no,
    }

    act = ActWrapper(act, act_params)

    replay_buffer = ReplayBuffer(buffer_size)
    beta_schedule = None

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * max_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    with tempfile.TemporaryDirectory() as td:
        model_saved = False
        model_file = os.path.join(td, "model")
        for t in range(max_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            kwargs = {}

            update_eps = exploration.value(t)
            update_param_noise_threshold = 0.

            # Choose action
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            #action = random.choice(actions)
            #action = 6 #Action in order for the robot not to move

            reset = False

            new_obs, rew, done, _  = step(env, actions_discr[action], obs[:3])

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                print("Learning starts ----")

                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None

                #td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                [td_errors, weighted_error] = train(obses_t, actions, rewards, obses_tp1, dones, weights)

                print("td_errors", td_errors)
                print("weighted_error", weighted_error)

                # Tensorboard summary for average loss
                #tf.scalar_summary("av_td_error", mean(td_errors))
                #tf.scalar_summary("av_weighted_error", mean(weighted_error))

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    U.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            U.load_state(model_file)

    return act
