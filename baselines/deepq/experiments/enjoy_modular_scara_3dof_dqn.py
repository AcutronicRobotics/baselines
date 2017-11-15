import gym
from gym import spaces
import gym_gazebo
from baselines import deepq
import pandas
import numpy as np

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


def main():
    env = gym.make("GazeboModularScara3DOF-v2")
    act = deepq.load("scara_model.pkl")

    #Discrete actions
    goal_average_steps = 2
    max_number_of_steps = 20
    last_time_steps = np.ndarray(0)
    n_bins = 10

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



    while True:
        obs, done = env.reset(), False
        print("obs", obs)

        episode_rew = 0
        while not done:
            env.render()
            #obs, rew, done, _ = env.step(act(obs[None])[0])
            action = act(obs[None])[0]
            print("action", action)
            print("action_bins[action]", action_bins[action])
            obs, rew, done, _  = step(env, action_bins[action], obs[:3])
            print("reward", rew)
            print("observation", obs[:3])
            episode_rew += rew
            print("accumulated_reward", episode_rew)
            print("done", done)
        print("Episode reward", episode_rew)



if __name__ == '__main__':
    main()
