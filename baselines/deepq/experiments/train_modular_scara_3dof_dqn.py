import gym
from gym import spaces
import gym_gazebo

import numpy as np
import pandas
from baselines import deepq
from  baselines.deepq import models
from  baselines.deepq import build_graph_robotics
from  baselines.deepq import replay_buffer
from  baselines.deepq.simple_robotics import learn, load

# Use algorithms from baselines
#from baselines import deepq
def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def main():
    env = gym.make("GazeboModularScara3DOF-v2")

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
    action_bins = [(difference_bins, 0.0, 0.0), (-difference_bins, 0.0, 0.0),
            (0.0, difference_bins, 0.0), (0.0, -difference_bins, 0.0),
            (0.0, 0.0, difference_bins), (0.0, 0.0, -difference_bins),
            (0.0, 0.0, 0.0)]
    discrete_action_space = spaces.Discrete(7)
    model = models.mlp([64])

    act = learn(
        env,
        q_func=model,
        lr=1e-3,
        gamma=0.5,
        max_timesteps=500,
        buffer_size=5000,
        checkpoint_freq = 100,
        learning_starts = 500,
        target_network_update_freq = 100,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback)

    #print("Saving model to cartpole_model.pkl")
    act.save("scara_model.pkl")

if __name__ == '__main__':
    main()
