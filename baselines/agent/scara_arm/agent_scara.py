import os
import time
import copy
import json
import numpy as np # Used pretty much everywhere.
import matplotlib.pyplot as plt
import threading # Used for time locks to synchronize position data.
import rclpy
import tensorflow as tf

from timeit import default_timer as timer
from scipy import stats
from scipy.interpolate import spline
import geometry_msgs.msg

from os import path
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from baselines.agent.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix# For getting points and velocities.
# from gps.algorithm.policy.controller_prior_gmm import ControllerPriorGMM
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing scara joint angles.
from control_msgs.msg import JointTrajectoryControllerState
from std_msgs.msg import String

from baselines.agent.scara_arm.tree_urdf import treeFromFile # For KDL Jacobians
from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians
from collections import namedtuple
import scipy.ndimage as sp_ndimage
from functools import partial

from baselines.agent.utility import error
from baselines.agent.utility import seeding
from baselines.agent.utility import spaces

StartEndPoints = namedtuple('StartEndPoints', ['start', 'target'])
class MSG_INVALID_JOINT_NAMES_DIFFER(Exception):
    """Error object exclusively raised by _process_observations."""
    pass

class ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH(Exception):
    """Error object exclusively raised by reset."""
    pass


class AgentSCARAROS(object):
    """Connects the SCARA actions and Deep Learning algorithms."""

    def __init__(self, init_node=True): #hyperparams, , urdf_path, init_node=True
        """Initialized Agent.
        init_node:   Whether or not to initialize a new ROS node."""

        print("I am in init")
        self._observation_msg = None
        self.scale = None  # must be set from elsewhere based on observations
        self.bias = None
        self.x_idx = None
        self.obs = None
        self.reward = None
        self.done = None
        self.reward_dist = None
        self.reward_ctrl = None
        self.action_space = None
        # to work with baselines a2c need this ones
        self.num_envs = 1
        self.remotes = [0]
    #
    #     # Setup the main node.
        print("Init ros node")
        rclpy.init(args=None)
        node = rclpy.create_node('robot_ai_node')
        global node
        self._pub = node.create_publisher(JointTrajectory,self.agent['joint_publisher'])
        self._sub = node.create_subscription(JointTrajectoryControllerState, self.agent['joint_subscriber'], self._observation_callback, qos_profile=qos_profile_sensor_data)
        assert self._sub
        self._time_lock = threading.RLock()
        print("setting time clocks")

        if self.agent['tree_path'].startswith("/"):
            fullpath = self.agent['tree_path']
            print(fullpath)
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", self.agent['tree_path'])
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)

        print("I am in reading the file path: ", fullpath)


        # self._valid_joint_set = [set(hyperparams['joint_order'][ii]) for ii in xrange(self.parallel_num)]
        # self._valid_joint_index = [{joint: index for joint, index in
        #                            enumerate(hyperparams['joint_order'][ii])} for ii in xrange(self.parallel_num)]


        # Initialize a tree structure from the robot urdf.
        # Note that the xacro of the urdf is updated by hand.
        # Then the urdf must be compiled.
        _, self.ur_tree = treeFromFile(self.agent['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.ur_chain = self.ur_tree.getChain(self.agent['link_names'][0], self.agent['link_names'][-1])
        print("Nr. of jnts: ", self.ur_chain.getNrOfJoints())

    #     # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.ur_chain)
        print(self.jac_solver)
        self._observations_stale = [False for _ in range(1)]
        print("after observations stale")

    #
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

        # taken from mujoco in OpenAi how to initialize observation space and action space.
        observation, _reward, done, _info = self._step(np.zeros(self.ur_chain.getNrOfJoints()))
        assert not done
        self.obs_dim = observation.size
        # print(observation, _reward)
        # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # bounds = self.model.actuator_ctrlrange.copy()
        low = -np.pi/2.0 * np.ones(self.ur_chain.getNrOfJoints()) #bounds[:, 0]
        high = np.pi/2.0 * np.ones(self.ur_chain.getNrOfJoints()) #bounds[:, 1]
        print("Action spaces: ", low, high)
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        # self.seed()
    def _observation_callback(self, message):
        # print("Trying to call observation msgs")
        # global _observation_msg
        self._observation_msg =  message
        # print(self._observation_msg.joint_names)
        # print(_observation_msg)
        # """This callback is set on the subscriber node in self.__init__().
        # It's called by ROS every 40 ms while the subscriber is listening.
        # Primarily updates the present and latest times.
        # This callback is invoked asynchronously, so is effectively a
        # "subscriber thread", separate from the control flow of the rest of
        # GPS, which runs in the "main thread".
        # message: observation from the robot to store each listen."""
        # with self._time_lock:
        #     self._observations_stale[robot_id] = False
        #     self._observation_msg = message
        #     if self._currently_resetting[robot_id]:
        #         epsilon = 1e-3
        #         reset_action = self.reset_joint_angles[robot_id]
        #         now_action = np.asarray(
        #             self._observation_msg.actual.positions[:len(reset_action)])
        #         du = np.linalg.norm(reset_action-now_action, float('inf'))
        #         if du < epsilon:
        #             self._currently_resetting[robot_id] = False
                    # self._reset_cv.notify_all()
        # print('robot call back ', robot_id)

    def reset(self, robot_id=0):
        """Not necessarily a helper function as it is inherited.
        Reset the agent for a particular experiment condition.
        condition: An index into hyperparams['reset_conditions']."""

        # Set the reset position as the initial position from agent hyperparams.
        self.reset_joint_angles[robot_id] = self.agent['reset_conditions']['initial_positions']

        # Prepare the present positions to see how far off we are.
        now_position = np.asarray(self._observation_msg.actual.positions[:len(self.reset_joint_angles[robot_id])])

        # Raise error if robot has made contact with the ground in simulation.
        # This occurs because Gazebo sets joint angles beyond what they can possibly
        # be when the robot makes contact with the ground and "breaks."
        if max(abs(now_position)) >= 2*np.pi:
            raise ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH

        # Wait until the arm is within epsilon of reset configuration.
        action_msg = JointTrajectory()
        with self._time_lock:
            self._currently_resetting = True
            action_msg = self._get_trajectory_message(self.reset_joint_angles[robot_id], self.agent, robot_id=robot_id)
            # self._pub.publish(action_msg)
            # time.sleep(self.agent['slowness'])
            # # action_msg.points[0].positions = [np.random.uniform(low=-3.14159, high=3.14159) for i in range(3)]
            # # print(action_msg)
            #
            # # print(action_msg.points[0].positions)
            #
            # a = np.zeros(3, dtype=np.float) + np.random.uniform(low=-3.14159, high=3.14159, size=3)
            # b = np.zeros(3, dtype=np.float)  + np.random.uniform(low=-3.14159, high=3.14159, size=3)
            # # print("action_msgs",action_msg.points[0].positions)
            # # print("np array",np.array([a, b]))
            # # print("uniform", [np.random.uniform(low=-180.005, high=180.005) for i in range(6)])
            #
            # # action_msg.points[0].positions = [np.random.uniform(low=-3.14159, high=3.14159) for i in range(3)]
            # # action_msg.points[0].positions = [0.0, 0.0, 0.0]
            # # print(action_msg.points[0].positions.shape)
            # a = np.squeeze(np.asarray(self.agent['end_effector_velocities']))
            # print("self.agent['end_effector_velocities']: ", a)
            # print("self.agent['end_effector_points']: ", np.reshape(self.agent['end_effector_points'], -1))
            # print("np.reshape(np.array(action_msg.points[0].positions), -1): ", np.reshape(np.array(action_msg.points[0].positions), -1))
            # print("action_msg.points[0].positions: ", action_msg.points[0].positions)

            # here we reset the position to 0 in each joint. Probably I need to check this function and compare it to mujoco openai
            reset = np.r_[np.reshape(np.array(action_msg.points[0].positions), -1),
                            np.reshape(np.squeeze(np.asarray(self.agent['end_effector_points'])), -1),
                            np.reshape(np.squeeze(np.asarray(self.agent['end_effector_velocities'])), -1)]
        # #     # c = action_msg.points[0].positions
        # #     print("reset model: ",reset)
        #     print("obs model: ",self.ob)
        # #     c = reset
        # # #
        # #
        # reset = self.ob
        return reset
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    # initialize the steps
    def _step(self, action, robot_id=0):
        time_step = 0
        # publish_frequencies = []
        # start = timer()
        # record_actions = [[] for i in range(6)]
        # sample_idx = self.condition_run_trial_times[condition]
        while time_step < self.agent['T']: #rclpy.ok():
        # if rclpy.ok(): #rclpy.ok():
            # print("ROS is ok moving on.")

            # print("Time step: ", time_step)
            # Only read and process ROS messages if they are fresh.
            if self._observations_stale[robot_id] is False:
                # # Acquire the lock to prevent the subscriber thread from
                # # updating times or observation messages.
                self._time_lock.acquire(True)
                obs_message = self._observation_msg

                # Make it so that subscriber's thread observation callback
                # must be called before publishing again.
                self._observations_stale[robot_id] = False

                # Collect the end effector points and velocities in
                # cartesian coordinates for the state.
                # Collect the present joint angles and velocities from ROS for the state.
                last_observations = self._process_observations(obs_message, self.agent)
                if last_observations is None:
                    print("last_observations is empty")
                else:
                    self._time_lock.acquire(True)
                    self._pub.publish(self._get_trajectory_message(action[:self.ur_chain.getNrOfJoints()], self.agent))#rclpy.ok():
                    self._time_lock.release()
                # # # Get Jacobians from present joint angles and KDL trees
                # # # The Jacobians consist of a 6x6 matrix getting its from from
                # # # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
                    ee_link_jacobians = self._get_jacobians(last_observations)
                    if self.agent['link_names'][-1] is None:
                        print("End link is empty!!")
                    else:
                        # print(self.agent['link_names'][-1])
                        trans, rot = forward_kinematics(self.ur_chain,
                                                    self.agent['link_names'],
                                                    last_observations,
                                                    base_link=self.agent['link_names'][0],
                                                    end_link=self.agent['link_names'][-1])
                        # #
                        rotation_matrix = np.eye(4)
                        rotation_matrix[:3, :3] = rot
                        rotation_matrix[:3, 3] = trans
                        # angle, dir, _ = rotation_from_matrix(rotation_matrix)
                        # #
                        # current_quaternion = np.array([angle]+dir.tolist())#

                        # I need this calculations for the new reward function, need to send them back to the run scara or calculate them here
                        current_quaternion = quaternion_from_matrix(rotation_matrix)

                        current_ee_tgt = np.ndarray.flatten(get_ee_points(self.agent['end_effector_points'],
                                                                          trans,
                                                                          rot).T)
                        ee_points = current_ee_tgt - self.agent['ee_points_tgt']

                        ee_points_jac_trans, _ = self._get_ee_points_jacobians(ee_link_jacobians,
                                                                               self.agent['end_effector_points'],
                                                                               rot)
                        ee_velocities = self._get_ee_points_velocities(ee_link_jacobians,
                                                                       self.agent['end_effector_points'],
                                                                       rot,
                                                                       last_observations)

                        #
                        # Concatenate the information that defines the robot state
                        # vector, typically denoted asrobot_id 'x'.
                        # state = np.r_[np.reshape(last_observations, -1),
                        #               np.reshape(ee_points, -1),
                        #               np.reshape(ee_velocities, -1),]

                        self.ob = np.r_[np.reshape(last_observations, -1),
                                      np.reshape(ee_points, -1),
                                      np.reshape(ee_velocities, -1),]
                        # change here actions if its not working, I need to figure out how to 1. Get current action, run some policy on it and then send it back to the robot to simulate.
                        # how do you generate actions in OpenAI
                        # change here actions if its not working, I need to figure out how to 1. Get current action, run some policy on it and then send it back to the robot to simulate.
                        # how do you generate actions in OpenAI
                        #if the error is less than 5 mm give good reward, if not give negative reward
                        if np.linalg.norm(ee_points) < 0.005:
                            self.reward_dist = 1000 * np.linalg.norm(ee_points)#- 10.0 * np.linalg.norm(ee_points)
                            # we do not use this and makes the convergence very bad. We need to remove it
                            # self.reward_ctrl = np.linalg.norm(action)#np.square(action).sum()
                            self.reward = 100
                            print("Eucledian dist (mm): ", self.reward_dist)
                        # if we are close to the goal in 1 cm give positive reward, converting the distance from meters to mm
                        elif np.linalg.norm(ee_points) < 0.01:
                            self.reward_dist = 1000 * np.linalg.norm(ee_points)
                            self.reward = self.reward_dist
                            # print("Eucledian dist (mm): ", self.reward_dist)
                        else:
                            self.reward_dist = - np.linalg.norm(ee_points)
                            #self.reward_ctrl = - np.linalg.norm(action)# np.square(action).sum()
                            self.reward = self.reward_dist
                        # self.reward = 2.0 * self.reward_dist + 0.01 * self.reward_ctrl
                        #removed the control reward, maybe we should add it later.
                        # TODO: this is something we need to figure out... Should we restart the enviroment to the initial observation every time we hit the target or when we went too far, or both?
                        # for now setting it to false all the time, meaning the enviroment will never restart.
                        done = False #bool(np.linalg.norm(ee_points) < 0.005)#False

                    self._time_lock.release()

                rclpy.spin_once(node)
                time_step += 1
                # print("time_step: ", time_step)
        return self.ob, self.reward, self.done, dict(reward_dist=self.reward_dist, reward_ctrl=self.reward_ctrl)

    def step(self, action, robot_id=0):
        time_step = 0
        publish_frequencies = []
        start = timer()
        record_actions = [[] for i in range(6)]
        # sample_idx = self.condition_run_trial_times[condition]
        #while time_step < self.agent['T']: #rclpy.ok():
        if rclpy.ok():
            self._time_lock.acquire(True)
            self._pub.publish(self._get_trajectory_message(action[:self.ur_chain.getNrOfJoints()], self.agent))#rclpy.ok():
            self._time_lock.release()

            # print("ROS is ok moving on.")

            # print("Time step: ", time_step)
            # Only read and process ROS messages if they are fresh.
            if self._observations_stale[robot_id] is False:
                # # Acquire the lock to prevent the subscriber thread from
                # # updating times or observation messages.
                self._time_lock.acquire(True)

                obs_message = self._observation_msg

                # Make it so that subscriber's thread observation callback
                # must be called before publishing again.
                self._observations_stale[robot_id] = False

                # Collect the end effector points and velocities in
                # cartesian coordinates for the state.
                # Collect the present joint angles and velocities from ROS for the state.
                last_observations = self._process_observations(obs_message, self.agent)
                if last_observations is None:
                    print("last_observations is empty")
                else:
                # # # Get Jacobians from present joint angles and KDL trees
                # # # The Jacobians consist of a 6x6 matrix getting its from from
                # # # (# joint angles) x (len[x, y, z] + len[roll, pitch, yaw])
                    ee_link_jacobians = self._get_jacobians(last_observations)
                    if self.agent['link_names'][-1] is None:
                        print("End link is empty!!")
                    else:
                        # print(self.agent['link_names'][-1])
                        trans, rot = forward_kinematics(self.ur_chain,
                                                    self.agent['link_names'],
                                                    last_observations[:3],
                                                    base_link=self.agent['link_names'][0],
                                                    end_link=self.agent['link_names'][-1])
                        # #
                        rotation_matrix = np.eye(4)
                        rotation_matrix[:3, :3] = rot
                        rotation_matrix[:3, 3] = trans
                        # angle, dir, _ = rotation_from_matrix(rotation_matrix)
                        # #
                        # current_quaternion = np.array([angle]+dir.tolist())#

                        # I need this calculations for the new reward function, need to send them back to the run scara or calculate them here
                        current_quaternion = quaternion_from_matrix(rotation_matrix)

                        current_ee_tgt = np.ndarray.flatten(get_ee_points(self.agent['end_effector_points'],
                                                                          trans,
                                                                          rot).T)
                        ee_points = current_ee_tgt - self.agent['ee_points_tgt']

                        ee_points_jac_trans, _ = self._get_ee_points_jacobians(ee_link_jacobians,
                                                                               self.agent['end_effector_points'],
                                                                               rot)
                        ee_velocities = self._get_ee_points_velocities(ee_link_jacobians,
                                                                       self.agent['end_effector_points'],
                                                                       rot,
                                                                       last_observations)

                        #
                        # Concatenate the information that defines the robot state
                        # vector, typically denoted asrobot_id 'x'.
                        state = np.r_[np.reshape(last_observations, -1),
                                      np.reshape(ee_points, -1),
                                      np.reshape(ee_velocities, -1),]

                        self.ob = np.r_[np.reshape(last_observations, -1),
                                      np.reshape(ee_points, -1),
                                      np.reshape(ee_velocities, -1),]
                        # TODO: remove garbage code.
                        #if the error is less than 5 mm give good reward, if not give negative reward
                        # if np.linalg.norm(ee_points) < 0.005:
                        #     self.reward_dist = 1000 * np.linalg.norm(ee_points)#- 10.0 * np.linalg.norm(ee_points)
                        #     # we do not use this and makes the convergence very bad. We need to remove it
                        #     # self.reward_ctrl = np.linalg.norm(action)#np.square(action).sum()
                        #     self.reward = 100
                        #     print("Eucledian dist (mm): ", self.reward_dist)
                        # # if we are close to the goal in 1 cm give positive reward, converting the distance from meters to mm
                        # elif np.linalg.norm(ee_points) < 0.01:
                        #     self.reward_dist = 1000 * np.linalg.norm(ee_points)
                        #     self.reward = self.reward_dist
                        #     # print("Eucledian dist (mm): ", self.reward_dist)
                        # else:
                        #     self.reward_dist = - np.linalg.norm(ee_points)
                        #     #self.reward_ctrl = - np.linalg.norm(action)# np.square(action).sum()
                        #     self.reward = self.reward_dist

                        # TODO: thinking about it it really makes sense to use negative distance as the reward function, because the closer to zero the better you are.
                        # setting the reward to positive has negative effect of the overall performance: 1) further you are the better the reward is.
                        # If you set to static reward you do not give a chance to the algorithm to converge (since everything that is smaller than 5 mm is going to have same reward)
                        self.reward_dist = - self.rmse_func(ee_points)
                        #self.reward_ctrl = - np.linalg.norm(action)# np.square(action).sum()
                        self.reward = self.reward_dist
                        if abs(self.reward) < 0.005:
                            print("Eucledian dist (mm): ", -1000 * self.reward_dist)

                        done = False #bool(np.linalg.norm(ee_points) < 0.005)#False
                    self._time_lock.release()

                rclpy.spin_once(node)

        return self.ob, self.reward, self.done, dict(reward_dist=self.reward_dist, reward_ctrl=self.reward_ctrl)

    def rmse_func(self, ee_points):
      """
        Computes the Residual Mean Square Error of the difference between current and desired end-effector position
      """
      rmse = np.sqrt(np.mean(np.square(ee_points), dtype=np.float32))
      return rmse

    def _get_jacobians(self, state, robot_id=0):
        """Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """

        # Initialize a Jacobian for n joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(self.ur_chain.getNrOfJoints())

        # Initialize a joint array for the present n joint angles.
        angles = JntArray(self.ur_chain.getNrOfJoints())

        # Construct the joint array from the most recent joint angles.
        for i in range(self.ur_chain.getNrOfJoints()):
            angles[i] = state[i]

        # Update the jacobian by solving for the given angles.
        self.jac_solver.JntToJac(angles, jacobian)

        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])

        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J
        return ee_jacobians


    def _process_observations(self, message, agent, robot_id=0):
        """Helper fuinction only called by _run_trial to convert a ROS message
        to joint angles and velocities.
        Check for and handle the case where a message is either malformed
        or contains joint values in an order different from that expected
        in hyperparams['joint_order']"""
        # print(message)
        # len(message)
        if not message:
            print("Message is empty");
        else:
            # # Check if joint values are in the expected order and size.
            if message.joint_names != agent['joint_order']:
                # Check that the message is of same size as the expected message.
                if len(message.joint_names) != len(agent['joint_order']):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER

                # Check that all the expected joint values are present in a message.
                if not all(map(lambda x,y: x in y, message.joint_names,
                    [self._valid_joint_set[robot_id] for _ in range(len(message.joint_names))])):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
                    print("Joints differ")

            return np.array(message.actual.positions) # + message.actual.velocities

                # # If necessary, reorder the joint values to conform to the order
                # # expected in hyperparams['joint_order'].
                # new_message = [None for _ in range(len(message))]
                # print(new_message)
                # for joint, index in message.joint_names.enumerate():
                #     for state_type in self._hyperparams['state_types']:
                #         new_message[self._valid_joint_index[robot_id][joint]] = message[state_type][index]
                #
                # message = new_message
                # #
                # # # Package the positions, velocities, amd accellerations of the joint angles.
                # # for (state_type, state_category), state_value_vector in zip(
                # #     # self.agent['state_types'].items(),
                # #     [message.actual.positions, message.actual.velocities,
                # #     message.actual.accelerations]):
                # #
                # #     # Assert that the length of the value vector matches the corresponding
                # #     # number of dimensions from the hyperparameters file
                # #     # assert len(state_value_vector) == self._hyperparams['sensor_dims'][state_category]
                # #
                # #     # Write the state value vector into the results dictionary keyed by its
                # #     # state category
                # #     result[state_category].append(state_value_vector)
                # #     print(result)
                # #


    def _get_trajectory_message(self, action, agent, robot_id=0):
        """Helper function only called by reset() and run_trial().
        Wraps an action vector of joint angles into a JointTrajectory message.
        The velocities, accelerations, and effort do not control the arm motion"""

        # Set up a trajectory message to publish.
        action_msg = JointTrajectory()
        action_msg.joint_names = agent['joint_order']

        # Create a point to tell the robot to move to.
        target = JointTrajectoryPoint()
        action_float = [float(i) for i in action]
        target.positions = action_float

        # These times determine the speed at which the robot moves:
        # it tries to reach the specified target position in 'slowness' time.
        target.time_from_start.sec = agent['slowness']

        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        return action_msg

    def _get_ee_points_jacobians(self, ref_jacobian, ee_points, ref_rot):
        """
        Get the jacobians of the points on a link given the jacobian for that link's origin
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :return: 3N x 6 Jac_trans, each 3 x 6 numpy array is the Jacobian[:3, :] for that point
                 3N x 6 Jac_rot, each 3 x 6 numpy array is the Jacobian[3:, :] for that point
        """
        ee_points = np.asarray(ee_points)
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
        ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                        np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                            (0, 2, 1)).reshape(-1, self.ur_chain.getNrOfJoints())
        ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
        return ee_points_jac_trans, ee_points_jac_rot

    def _get_ee_points_velocities(self, ref_jacobian, ee_points, ref_rot, joint_velocities):
        """
        Get the velocities of the points on a link
        :param ref_jacobian: 6 x 6 numpy array, jacobian for the link's origin
        :param ee_points: N x 3 numpy array, points' coordinates on the link's coordinate system
        :param ref_rot: 3 x 3 numpy array, rotational matrix for the link's coordinate system
        :param joint_velocities: 1 x 6 numpy array, joint velocities
        :return: 3N numpy array, velocities of each point
        """
        ref_jacobians_trans = ref_jacobian[:3, :]
        ref_jacobians_rot = ref_jacobian[3:, :]
        ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
        ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
        ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                       ref_rot.dot(ee_points.T).T)
        return ee_velocities.reshape(-1)
