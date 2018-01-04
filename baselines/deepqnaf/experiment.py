#TODO improved logging, include if env solved

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym
import os
import time
import argparse
import baselines.deepqnaf.naf as naf
from baselines import logger
from baselines.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #silence TF compilation warnings

def fill_episodes(rewards, n, value):
  return rewards + [value]*n

#get legends for plot
def recursive_legend(keys, remaining_vals, vals):
  if remaining_vals == []:
    legend = ""
    for i,l in enumerate(vals):
      legend += str(keys[i]) + "=" + str(l) + ","
    legend = legend[:-1] #remove last comma
    return [legend]
  else:
    legend = []
    for l in remaining_vals[0]:
      legend += recursive_legend(keys, remaining_vals[1:], vals + [l])
    return legend

#run experiment for every combination of hyperparameters
def recursive_experiment(keys, remaining_vals, vals):
  if remaining_vals == []:
    return [experiment(dict(zip(keys,vals)))]
  else:
    rewards = []
    if type(remaining_vals[0]) != list:
      rewards += recursive_experiment(keys, remaining_vals[1:], vals + [remaining_vals[0]])
    else:
      for r in remaining_vals[0]:
        rewards += recursive_experiment(keys, remaining_vals[1:],  vals + [r])
    return rewards

def learn(env,
            v = 0,
            graph = True,
            render = True,
            repeats = 1,
            episodes = 1000,
            max_episode_steps = 200,
            train_steps = 5,
            batch_normalize = True,
            learning_rate = 0.001,
            gamma = 0.99,
            tau = 0.99,
            epsilon = 0.1,
            hidden_size = 100,
            hidden_n = 2,
            hidden_activation = tf.nn.relu,
            batch_size = 128,
            memory_capacity = 10000,
            load_path = None,
            covariance = "original"):
  if v > 0:
    print("Experiment " + str(args))

  experiments_rewards = []
  for i in range(repeats):
    agent = naf.Agent(v, env.observation_space, env.action_space, learning_rate, batch_normalize, gamma, tau, epsilon, hidden_size,hidden_n, hidden_activation,batch_size, memory_capacity, load_path, covariance)
    experiment_rewards = []
    terminate = None
    solved = 0 #only relevant if solved_threshold is set

    for j in range(episodes):
      if terminate is not None:
        fill_value = 0
        if terminate == "solved":
          fill_value = solve_threshold
        experiment_rewards = fill_episodes(experiment_rewards, episodes-j, fill_value)
        break

      rewards = 0
      state = env.reset()

      for k in range(max_episode_steps):
        if render:
          env.render()

        action = agent.get_action(state)
        if np.isnan(np.min(action)): #if NaN action (neural network exploded)
          print("Warning: NaN action, terminating agent")
          with open("error.txt", "a") as error_file:
            error_file.write(str(args) + " repeat " + str(i) + " episode " + str(j) + " step " + str(k) + " NaN\n")
          rewards = 0 #TODO ?
          terminate = "nan"
          break
        #print(action)
        state_next,reward,terminal,_ = env.step(agent.scale(action, env.action_space.low, env.action_space.high))

        if k-1 >= max_episode_steps:
          terminal = True

        agent.observe(state,action,reward,state_next,terminal)

        for l in range(train_steps):
          agent.learn()

        state = state_next
        rewards += reward
        if terminal:
          agent.reset()
          break
      experiment_rewards += [rewards]

    #   if solve_threshold is not None:
    #     if rewards >= solve_threshold:
    #       solved += 1
    #     else:
    #       solved = 0
    #     if solved >= 10: #number of repeated rewards above threshold to consider environment solved = 10
    #       print("[Solved]")
    #       terminate = "solved"


      #print("logger directory", logger.get_dir())
      #print("rewards", rewards)
      #print("np.std(experiment_rewards)", np.std(experiment_rewards))
      #print("EpRew",  mpi_mean(np.mean(experiment_rewards)))
      #print("EpRewStd",  np.std(experiment_rewards))
      logger.record_tabular("EpRew",  mpi_mean(np.mean(experiment_rewards)))
      logger.record_tabular("EpRewStd",  np.std(experiment_rewards))
      logger.dump_tabular()

      #tensorboard
      tensorboard_outdir = '/tmp/rosrl/GazeboModularScara3DOF-v3/deepq_naf/'+ str(j)
      summary_writer = tf.summary.FileWriter(tensorboard_outdir, graph=tf.get_default_graph())
      summary = tf.Summary(value=[tf.Summary.Value(tag="Experiment reward", simple_value = mpi_mean(np.mean(experiment_rewards)))])
      summary_writer.add_summary(summary, j)

      #print("experiment_rewards", experiment_rewards)

    #   if solve_threshold is not None:
    #     if rewards >= solve_threshold:
    #       solved += 1
    #     else:
    #       solved = 0
    #     if solved >= 10: #number of repeated rewards above threshold to consider environment solved = 10
    #       print("[Solved]")
    #       terminate = "solved"

    # if args['v'] > 0:
    #     print("Reward(" + str(i) + "," + str(j) + "," + str(k) + ")=" + str(rewards))
    # if args['v'] > 1:
    #   print(np.mean(experiment_rewards[-10:]))
    experiments_rewards += [experiment_rewards]

  #print("experiments_rewards", mpi_mean(np.mean(experiment_rewards)))



  return experiments_rewards

