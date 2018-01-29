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

# class ROBOT_MADE_CONTACT_WITH_GAZEBO_GROUND_SO_RESTART_ROSLAUNCH(Exception):
#     """Error object exclusively raised by reset."""
#     pass


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

        # Setup the main node.
        print("Init ros node")
        rclpy.init(args=None)
        self.node = rclpy.create_node('robot_ai_node')
        self._pub = self.node.create_publisher(JointTrajectory,self.agent['joint_publisher'])
        self._sub = self.node.create_subscription(JointTrajectoryControllerState,
                                            self.agent['joint_subscriber'],
                                            self._observation_callback,
                                            qos_profile=qos_profile_sensor_data)
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

        # Initialize a tree structure from the robot urdf.
        # Note that the xacro of the urdf is updated by hand.
        # Then the urdf must be compiled.
        _, self.scara_tree = treeFromFile(self.agent['tree_path'])
        # Retrieve a chain structure between the base and the start of the end effector.
        self.scara_chain = self.scara_tree.getChain(self.agent['link_names'][0], self.agent['link_names'][-1])
        print("Nr. of jnts: ", self.scara_chain.getNrOfJoints())

    #     # Initialize a KDL Jacobian solver from the chain.
        self.jac_solver = ChainJntToJacSolver(self.scara_chain)
        print(self.jac_solver)
        self._observations_stale = False
        print("after observations stale")

    #
        self._currently_resetting = [False for _ in range(1)]
        self.reset_joint_angles = [None for _ in range(1)]

        # taken from mujoco in OpenAi how to initialize observation space and action space.
        observation, _reward, done, _info = self._step(np.zeros(self.scara_chain.getNrOfJoints()))
        assert not done
        self.obs_dim = observation.size
        print(self.obs_dim)
        # print(observation, _reward)
        # Here idially we should find the control range of the robot. Unfortunatelly in ROS/KDL there is nothing like this.
        # I have tested this with the mujoco enviroment and the output is always same low[-1.,-1.], high[1.,1.]
        # bounds = self.model.actuator_ctrlrange.copy()
        low = -np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints()) #bounds[:, 0]
        high = np.pi/2.0 * np.ones(self.scara_chain.getNrOfJoints()) #bounds[:, 1]
        # print("Action Spaces:")
        # print("low: ", low, "high: ", high)
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        print(self.observation_space)

        self.goal_cmd = [0]*self.scara_chain.getNrOfJoints()
        self.goal_vel = [ self.agent['goal_vel'] ]*self.scara_chain.getNrOfJoints()

        self.goal_vel_value = [self.agent['goal_vel']]*self.scara_chain.getNrOfJoints()
        self.goal_effort_value = [float('nan')]*self.scara_chain.getNrOfJoints()
        self.joint_order = self.agent['joint_order']


        # self.seed()
    def _observation_callback(self, message):
        # message: observation from the robot to store each listen."""
        # self._time_lock.acquire(True, -1)
        # with self._time_lock:
        self._observations_stale = False
        self._observation_msg = message
            # if self._currently_resetting:
            #     epsilon = 1e-3
            #     reset_action = self.agent['reset_conditions']['initial_positions']
            #     now_action = self._observation_msg.actual.positions
            #     du = np.linalg.norm(reset_action-now_action, float(np.inf))
            #     if du < epsilon:
            #         self._currently_resetting = False

        # self._time_lock.release()


    def reset(self):
        """Reset function should call reset_model.
           In OpenAI in reset_model we are setting the robot to initial pose + some random number.
           This function returns the observations which are then used again for the policy calculation.
           In our case we are going to set the robot to initial pose, for now given by the user."""
        self.obs = None
        # self._currently_resetting = True

        # Set the reset position as the initial position from agent hyperparams.
        self.reset_joint_angles = self.agent['reset_conditions']['initial_positions']
        # Prepare the present positions to see how far off we are.
        now_position = np.asarray(self._observation_msg.actual.positions)

        # Wait until the arm is within epsilon of reset configuration.
        # action_msg = JointTrajectory()
        # self._time_lock.acquire(True, -1)
        # with self._time_lock:
        #     self._currently_resetting = True


        # if self._currently_resetting:
        #     epsilon = 1e-3
        #     reset_action = self.agent['reset_conditions']['initial_positions']
        #     now_action = self._observation_msg.actual.positions
        #     du = np.linalg.norm(reset_action-now_action, float(np.inf))
        #     if du > epsilon:
        #         self._pub.publish(self._get_trajectory_message(self.agent['reset_conditions']['initial_positions'], self.agent))
        #         time.sleep(self.agent['slowness'])
        #         self._currently_resetting = True
        #         print("reset is false.")
        #     else:
        #         self._currently_resetting = False
        #         print("reset is true.")

        self.ob, ee_points = self._get_obs()
        # self._time_lock.release()

        # print("resetting: ", self.ob)
        self.last_time = time.time()

        return self.ob
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _get_obs(self):
        observations = None
        ee_points = None
        if self._observations_stale is False:
            # # Acquire the lock to prevent the subscriber thread from
            # # updating times or observation messages.
            # self._time_lock.acquire(True)

            obs_message = self._observation_msg
            # if obs_message is None:
            #     obs_message = self._observation_msg
            #     # time.sleep(self.agent['slowness'])
            #     # self._observations_stale = True
            # self._time_lock.release()



            # Make it so that subscriber's thread observation callback
            # must be called before publishing again.

            # Collect the end effector points and velocities in
            # cartesian coordinates for the state.
            # Collect the present joint angles and velocities from ROS for the state.
            last_observations = self._process_observations(obs_message, self.agent)

            # self._observations_stale = True


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
                    self._observations_stale = False
                    # print(self.agent['link_names'][-1])
                    trans, rot = forward_kinematics(self.scara_chain,
                                                self.agent['link_names'],
                                                last_observations[:self.scara_chain.getNrOfJoints()],
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

                    observations = np.r_[np.reshape(last_observations, -1),
                                  np.reshape(ee_points, -1),
                                  np.reshape(ee_velocities, -1),]
                    if observations is None:
                        #need to handle this properly
                        print("Observations are none!!!")
                    # else:
                    #     print("We have observations")


            return observations, ee_points
    # initialize the steps
    def _step(self, action):
        time_step = 0
        ee_points = None
        self.ob = None
        while time_step < self.agent['T']: #rclpy.ok():
            """
            How they do in OpenAI:
              1. Calculate the reward
              2. Perform action (do_simulation)
              3. Get the Observations
            """
            _, ee_points  = self._get_obs()

            if ee_points is None:
                print("self.ob: ", self.ob)
                print("ee_points: ", ee_points)
                done = False
                rclpy.spin_once(self.node)
                # time.sleep(self.agent['slowness'])
            else:
                # self.reward_dist = - self.rmse_func(ee_points) - 0.5 * abs(np.sum(self.log_dist_func(ee_points)))
                # print("reward: ", self.reward_dist)

                self.reward_dist = - self.rmse_func(ee_points)
                # print("reward_dist :", self.reward_dist)
                if(self.reward_dist<0.005):
                    self.reward = 1 - self.rmse_func(ee_points) # Make the reward increase as the distance decreases
                    print("Reward is: ", self.reward)
                    print("Eucledian distance is: ", np.linalg.norm(ee_points))
                else:
                    self.reward = self.reward_dist

                print("reward: ", self.reward)

                # Calculate if the env has been solved
                done = bool(abs(self.reward_dist) < 0.005)

                # self._time_lock.acquire(True)
                # self._pub.publish(self._get_trajectory_message(action[:self.scara_chain.getNrOfJoints()], self.agent))#rclpy.ok():

                while self.ob is None or ee_points is None:
                    self.ob, ee_points  = self._get_obs()
                    rclpy.spin_once(self.node)
                    # time.sleep(self.agent['slowness'])

                # self._time_lock.release()

                    # self._time_lock.release()
                rclpy.spin_once(self.node)

                time_step += 1

                # print("time_step: ", time_step)
                return self.ob, self.reward, done, {}

    def step(self, action):
        """
        Dont know if we need this but just in case set the obs to None, so always takes fresh values
        """
        self.obs = None
        observations = None

        print("Action: ", action)
        """
        How they do in OpenAI:
             1. Calculate the reward
             2. Perform action (do_simulation)
             3. Get the Observations
        """
        start = timer()
        # Check if ROS2 is ok
        if rclpy.ok():
            observations, ee_points  = self._get_obs()
            if self.ob is None or ee_points is None:
                print("self.ob: ", self.ob)
                print("ee_points: ", ee_points)
                done = False
                rclpy.spin_once(self.node)
                time_step += 1
            else:
                # self.reward_dist = - abs(self.rmse_func(ee_points)) #np.linalg.norm(ee_points) #self.rmse_func(ee_points) #- 0.5 * abs(np.sum(self.log_dist_func(ee_points)))
                # # print("reward: ",self.reward_dist)

                self.reward_dist = -self.rmse_func(ee_points)
                # print("reward_dist :", self.reward_dist)
                if(self.rmse_func(ee_points)<0.005):
                    self.reward = 1 - self.rmse_func(ee_points) # Make the reward increase as the distance decreases
                    print("Reward is: ", self.reward)
                    print("Eucledian distance is: ", np.linalg.norm(ee_points))
                else:
                    self.reward = self.reward_dist
                    # print("Eucledian distance is: ", np.linalg.norm(ee_points))

                # print("reward: ", self.reward)
                # print("rmse_func: ", self.rmse_func(ee_points))

                # Calculate if the env has been solved
                done = bool(abs(self.reward_dist) < 0.005)

                self.ob, ee_points  = self._get_obs()

                dt = time.time() - self.last_time
                for i in range(self.scara_chain.getNrOfJoints()):
                    if(self._observation_msg.actual.positions[i] > action[i]):
                        self.goal_vel[i] = -self.goal_vel_value[0]
                    else:
                        self.goal_vel[i] =  self.goal_vel_value[0]
                    self.goal_cmd[i] += dt*self.goal_vel[i]

                self.last_time = time.time()

                action_msg = JointTrajectory()
                action_msg.joint_names = self.joint_order

                # Create a point to tell the robot to move to.
                target = JointTrajectoryPoint()
                target.positions  = self.goal_cmd

                target.velocities = self.goal_vel
                target.effort = self.goal_effort_value

                # Package the single point into a trajectory of points with length 1.
                action_msg.points = [target]

                self._pub.publish(action_msg)
                rclpy.spin_once(self.node)


        return self.ob, self.reward, done, {}

    def rmse_func(self, ee_points):
      """
        Computes the Residual Mean Square Error of the difference between current and desired end-effector position
      """
      rmse = np.sqrt(np.mean(np.square(ee_points), dtype=np.float32))
      return rmse
    def log_dist_func(self, ee_points):
        """
        Computes the Log of the Eucledian Distance Error between current and desired end-effector position
        """
        log_dist = np.log(ee_points, dtype=np.float32) # [:(self.scara_chain.getNrOfJoints())]
        log_dist[log_dist == -np.inf] = 0.0
        log_dist = np.nan_to_num(log_dist)
        # print("log_distance: ", log_dist)
        return log_dist

    def _get_jacobians(self, state):
        """Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
        This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
        The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
        Returns a repackaged Jacobian that is 3x6.
        """

        # Initialize a Jacobian for n joint angles by 3 cartesian coords and 3 orientation angles
        jacobian = Jacobian(self.scara_chain.getNrOfJoints())

        # Initialize a joint array for the present n joint angles.
        angles = JntArray(self.scara_chain.getNrOfJoints())

        # Construct the joint array from the most recent joint angles.
        for i in range(self.scara_chain.getNrOfJoints()):
            angles[i] = state[i]

        # Update the jacobian by solving for the given angles.
        self.jac_solver.JntToJac(angles, jacobian)

        # Initialize a numpy array to store the Jacobian.
        J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])

        # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
        ee_jacobians = J
        return ee_jacobians


    def _process_observations(self, message, agent):
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
                    [self._valid_joint_set for _ in range(len(message.joint_names))])):
                    raise MSG_INVALID_JOINT_NAMES_DIFFER
                    print("Joints differ")

            return np.array(message.actual.positions) # + message.actual.velocities


    def _get_trajectory_message(self, action, agent):
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
        # target.time_from_start.nanosec = agent['slowness']
        target.time_from_start.sec = agent['slowness']
        # Package the single point into a trajectory of points with length 1.
        action_msg.points = [target]

        print(action_msg)

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
                                            (0, 2, 1)).reshape(-1, self.scara_chain.getNrOfJoints())
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
