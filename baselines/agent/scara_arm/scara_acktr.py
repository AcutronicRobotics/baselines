#!/usr/bin/python
import sys

sys.path.append('/home/rkojcev/devel/gps/python')

from time import sleep

import rospy

import copy

from control_msgs.msg import JointTrajectoryControllerState # Used for subscribing to the UR.
from sensor_msgs.msg import JointState # Used for subscribing to the UR.
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint # Used for publishing UR joint angles.
from gps.agent.ur_ros.tree_urdf import treeFromFile

# from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
#         END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION, \
#         TRIAL_ARM, JOINT_SPACE, END_EFFECTOR_POINT_JACOBIANS

from gps.utility.general_utils import forward_kinematics, get_ee_points, rotation_from_matrix, \
    get_rotation_matrix,quaternion_from_matrix, inverse_kinematics# For getting points and velocities.

from PyKDL import Jacobian, Chain, ChainJntToJacSolver, JntArray # For KDL Jacobians

import tensorflow as tf
import numpy as np
import time
import math

_observation_msg = None
jac_solver = None

# sess=tf.Session()
#
# #First let's load meta graph and restore weights
# # saver = tf.train.import_meta_graph('/home/erle/github/neural_models/ur3_simple/tf_train2/policy_model.ckpt-210000.meta')
# # saver.restore(sess,'/home/erle/github/neural_models/ur3_simple/tf_train2/policy_model.ckpt-210000')
#
# filename = "/home/rkojcev/scara_e1_3joint/0_center/3/tf_train/policy_model.ckpt"
#
# # filename = "/home/rkojcev/devel/gps/experiments/scara_pose_3joints/saved/policy_model.ckpt"
#
#
# iteration = sys.argv[1]#'-60000'
#
# saver = tf.train.import_meta_graph(filename + iteration + '.meta')
# saver.restore(sess,filename + iteration)
#
# graph = tf.get_default_graph()


# import pickle
# pol_dict = pickle.load(open(filename + iteration + '_pol', "rb"))
# scale = pol_dict['scale']
# bias = pol_dict['bias']
#
# print(scale)
# print(bias)
#
# print(tf.global_variables())
#
# nn_input = graph.get_tensor_by_name("nn_input:0")
# out = graph.get_tensor_by_name("action_output/BiasAdd:0")
# print(nn_input)



# MOTOR1_JOINT = 'motor1'
# MOTOR2_JOINT = 'motor2'
# MOTOR3_JOINT = 'motor4'

# MOTOR1_JOINT = 'motor1'
# MOTOR2_JOINT = 'motor2'
# MOTOR3_JOINT = 'motor3'

# MOTOR1_JOINT = 'motor1'
# MOTOR2_JOINT = 'motor2'
# MOTOR3_JOINT = 'motor3'
#
# # Set constants for links
# WORLD = "world"
# BASE = 'scara_e1_base_link'
# BASE_MOTOR = 'scara_e1_base_motor'
#
# SCARA_MOTOR1 = 'scara_e1_motor1'
# SCARA_INSIDE_MOTOR1 = 'scara_e1_motor1_inside'
# SCARA_SUPPORT_MOTOR1 = 'scara_e1_motor1_support'
# SCARA_BAR_MOTOR1 = 'scara_e1_bar1'
# SCARA_FIXBAR_MOTOR1 = 'scara_e1_fixbar1'
#
# SCARA_MOTOR2 = 'scara_e1_motor2'
# SCARA_INSIDE_MOTOR2 = 'scara_e1_motor2_inside'
# SCARA_SUPPORT_MOTOR2 = 'scara_e1_motor2_support'
# SCARA_BAR_MOTOR2 = 'scara_e1_bar2'
# SCARA_FIXBAR_MOTOR2 = 'scara_e1_fixbar2'
#
# SCARA_MOTOR3 = 'scara_e1_motor3'
# SCARA_INSIDE_MOTOR3 = 'scara_e1_motor3_inside'
# SCARA_SUPPORT_MOTOR3 = 'scara_e1_motor3_support'
# SCARA_BAR_MOTOR3 = 'scara_e1_bar3'
# SCARA_FIXBAR_MOTOR3 = 'scara_e1_fixbar3'
#
# SCARA_RANGEFINDER = 'scara_e1_rangefinder'
#
# EE_LINK = 'ee_link'
#
# JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT]
# LINK_NAMES = [BASE, BASE_MOTOR,
#               SCARA_MOTOR1, SCARA_INSIDE_MOTOR1, SCARA_SUPPORT_MOTOR1, SCARA_BAR_MOTOR1, SCARA_FIXBAR_MOTOR1,
#               SCARA_MOTOR2, SCARA_INSIDE_MOTOR2, SCARA_SUPPORT_MOTOR2, SCARA_BAR_MOTOR2, SCARA_FIXBAR_MOTOR2,
#               SCARA_MOTOR3, SCARA_INSIDE_MOTOR3, SCARA_SUPPORT_MOTOR3,
#               EE_LINK]
#
# # States to check in agent._process_observations.
# STATE_TYPES = {'positions': JOINT_ANGLES,
#                'velocities': JOINT_VELOCITIES}
# Set end effector constants
INITIAL_JOINTS = np.array([0, 0, 0])

# Set the number of goal points. 1 by default for a single end effector tip.
EE_POINTS = np.asmatrix([[0, 0, 0]])

EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

tgt_quaternion = quaternion_from_matrix(EE_ROT_TGT)
# EE_POS_TGT = np.asmatrix([-0.05, 0.5, -0.120])
# EE_POS_TGT = np.asmatrix([0.7, 0.7, 0.5])
EE_POS_TGT = np.asmatrix([ 0.1195207, -0.3569719, 0.3745999932289124])

SCARA_GAINS = np.array([2.195, 0.922, 0.582])
# EE_POS_TGT = np.asmatrix([0.0, 0.419, 0.5-0.207])
# SENSOR_DIMS = {
#     JOINT_ANGLES: len(JOINT_ORDER),
#     JOINT_VELOCITIES: len(JOINT_ORDER),
#     END_EFFECTOR_POINTS: EE_POINTS.size,
#     END_EFFECTOR_POINT_VELOCITIES: EE_POINTS.size,
#     ACTION: len(SCARA_GAINS),
# }

def _observation_callback(message):
    global _observation_msg
    _observation_msg =  message

def _get_ur_trajectory_message( action, slowness):
    """Helper function only called by reset() and run_trial().
    Wraps an action vector of joint angles into a JointTrajectory message.
    The velocities, accelerations, and effort do not control the arm motion"""

    # Set up a trajectory message to publish.
    action_msg = JointTrajectory()

    global _observation_msg

    action_msg.joint_names = JOINT_ORDER

    target_current_pose = JointTrajectoryPoint()
    target_current_pose.positions = _observation_msg.actual.positions

    # Create a point to tell the robot to move to.
    target = JointTrajectoryPoint()
    target.positions = action[0:3]

    # These times determine the speed at which the robot moves:
    # it tries to reach the specified target position in 'slowness' time.
    target.time_from_start = rospy.Duration(slowness)

    # Package the single point into a trajectory of points with length 1.
    action_msg.points = [target]

    return action_msg

def _process_observations(message, result):
    """Helper fuinction only called by _run_trial to convert a ROS message
    to joint angles and velocities.
    Check for and handle the case where a message is either malformed
    or contains joint values in an order different from that expected
    in hyperparams['joint_order']"""


    # Check if joint values are in the expected order and size.
    if message.joint_names != JOINT_ORDER:

        # Check that the message is of same size as the expected message.
        if len(message.joint_names) != len(JOINT_ORDER):
            raise "MSG_INVALID_JOINT_NAMES_DIFFER"

        # Check that all the expected joint values are present in a message.
        if not all(map(lambda x,y: x in y, message.joint_names,
            [JOINT_ORDER for _ in range(len(message.joint_names))])):

            raise "MSG_INVALID_JOINT_NAMES_DIFFER"

        # If necessary, reorder the joint values to conform to the order
        # expected in hyperparams['joint_order'].
        new_message = [None for _ in range(len(message))]
        for joint, index in message.joint_names.enumerate():
            for state_type in STATE_TYPES:
                new_message[JOINT_ORDER[joint]] = message[state_type][index]

        message = new_message

    # Package the positions, velocities, amd accellerations of the joint angles.
    for (state_type, state_category), state_value_vector in zip(
        STATE_TYPES.items(),
        [message.actual.positions, message.actual.velocities,
        message.actual.accelerations]):

        # Assert that the length of the value vector matches the corresponding
        # number of dimensions from the hyperparameters file
        assert len(state_value_vector) == SENSOR_DIMS[state_category]

        # Write the state value vector into the results dictionary keyed by its
        # state category
        result[state_category].append(state_value_vector)

    return np.array(result[JOINT_ANGLES][-1] + result[JOINT_VELOCITIES][-1])

def _get_ee_points_jacobians(ref_jacobian, ee_points, ref_rot):
    """
    Get the jacobians of the points on a link given the jacobian for that link's origin
    :param ref_jacobian: 6 x 6 np array, jacobian for the link's origin
    :param ee_points: N x 3 np array, points' coordinates on the link's coordinate system
    :param ref_rot: 3 x 3 np array, rotational matrix for the link's coordinate system
    :return: 3N x 6 Jac_trans, each 3 x 6 np array is the Jacobian[:3, :] for that point
             3N x 6 Jac_rot, each 3 x 6 np array is the Jacobian[3:, :] for that point
    """
    ee_points = np.asarray(ee_points)
    ref_jacobians_trans = ref_jacobian[:3, :]
    ref_jacobians_rot = ref_jacobian[3:, :]
    end_effector_points_rot = np.expand_dims(ref_rot.dot(ee_points.T).T, axis=1)
    ee_points_jac_trans = np.tile(ref_jacobians_trans, (ee_points.shape[0], 1)) + \
                                    np.cross(ref_jacobians_rot.T, end_effector_points_rot).transpose(
                                        (0, 2, 1)).reshape(-1, 3)
    ee_points_jac_rot = np.tile(ref_jacobians_rot, (ee_points.shape[0], 1))
    return ee_points_jac_trans, ee_points_jac_rot

def _get_ee_points_velocities(ref_jacobian, ee_points, ref_rot, joint_velocities):
    """
    Get the velocities of the points on a link
    :param ref_jacobian: 6 x 6 np array, jacobian for the link's origin
    :param ee_points: N x 3 np array, points' coordinates on the link's coordinate system
    :param ref_rot: 3 x 3 np array, rotational matrix for the link's coordinate system
    :param joint_velocities: 1 x 6 np array, joint velocities
    :return: 3N np array, velocities of each point
    """
    ref_jacobians_trans = ref_jacobian[:3, :]
    ref_jacobians_rot = ref_jacobian[3:, :]
    ee_velocities_trans = np.dot(ref_jacobians_trans, joint_velocities)
    ee_velocities_rot = np.dot(ref_jacobians_rot, joint_velocities)
    ee_velocities = ee_velocities_trans + np.cross(ee_velocities_rot.reshape(1, 3),
                                                   ref_rot.dot(ee_points.T).T)
    return ee_velocities.reshape(-1)

def _get_jacobians(state, jac_solver):
    """Produce a Jacobian from the urdf that maps from joint angles to x, y, z.
    This makes a 6x6 matrix from 6 joint angles to x, y, z and 3 angles.
    The angles are roll, pitch, and yaw (not Euler angles) and are not needed.
    Returns a repackaged Jacobian that is 3x6.
    """

    # Initialize a Jacobian for 6 joint angles by 3 cartesian coords and 3 orientation angles
    jacobian = Jacobian(3)

    # Initialize a joint array for the present 6 joint angles.
    angles = JntArray(3)

    # Construct the joint array from the most recent joint angles.
    for i in range(3):
        angles[i] = state[i]

    # Update the jacobian by solving for the given angles.
    jac_solver.JntToJac(angles, jacobian)

    # Initialize a np array to store the Jacobian.
    J = np.array([[jacobian[i, j] for j in range(jacobian.columns())] for i in range(jacobian.rows())])

    # Only want the cartesian position, not Roll, Pitch, Yaw (RPY) Angles
    ee_jacobians = J
    return ee_jacobians, jac_solver


rospy.init_node('scara_e1_node')


INITIAL_JOINTS = np.array([0, -np.pi / 2, np.pi / 2, 0, 0, 0])

reset_joint_angles = INITIAL_JOINTS

# now_position = np.asarray(self._observation_msg[robot_id].actual.positions[:len(self.reset_joint_angles[robot_id])])

JOINT_PUBLISHER = '/scara_controller/command'
JOINT_SUBSCRIBER = '/scara_controller/state'
RESET_SLOWNESS = 10.0

_sub = rospy.Subscriber(JOINT_SUBSCRIBER, JointTrajectoryControllerState, _observation_callback)
_pub = rospy.Publisher(JOINT_PUBLISHER, JointTrajectory, queue_size=5)

r = rospy.Rate(20)

now_position = None
while True:
    now_position = _observation_msg
    r.sleep()
    if(now_position!=None):
        break
print(now_position)

TREE_PATH = '/home/rkojcev/catkin_ws/src/scara_e1/scara_e1_description/urdf/scara_e1_3joints.urdf'

SCARA_PREFIXES = ['']
#UR_PREFIXES = ['ur0', 'ur1', 'ur2', 'ur3', 'ur4', 'ur5', 'ur6', 'ur7', 'ur8', 'ur9' ]
m_joint_order = [copy.deepcopy(JOINT_ORDER) for prefix in SCARA_PREFIXES]
m_link_names = [copy.deepcopy(LINK_NAMES) for prefix in SCARA_PREFIXES]

_, ur_tree = treeFromFile(TREE_PATH)
# Retrieve a chain structure between the base and the start of the end effector.
# Retrieve a chain structure between the base and the start of the end effector.
print("------------> ", m_link_names[0][0])
print("------------> ", m_link_names[0][-1])
print("m_link_names:", m_link_names[0])
ur_chain = ur_tree.getChain(m_link_names[0][0], m_link_names[0][-1])
jac_solver = ChainJntToJacSolver(ur_chain)

ee_pos_tgt = EE_POS_TGT
ee_rot_tgt = EE_ROT_TGT

ee_tgt = np.ndarray.flatten(
    get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T
)
#
# cont = 1
# while cont < 90:
#     _pub.publish(_get_ur_trajectory_message(reset_joint_angles, 0.1))
#     r.sleep()
#     cont = cont +1

file = open("testfile.txt","w")

# file.write(str(ee_tgt[0]) + ", " + str(ee_tgt[1]) + ", " + str(ee_tgt[2]) + str("\n"))

# def fit_number(n):
#     while n > np.pi*2:
#         n = n - np.pi*2
#     while n < -np.pi*2:
#         n = n + np.pi*2
#     return n

while True:
    r.sleep()
    # print now_position.actual.positions
    # print now_position.actual.velocities

    # Initialize the data structure to be passed to GPS.
    result = {param: [] for param in
                       [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]
                     + [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES]
                     + []
                     + [END_EFFECTOR_POINT_JACOBIANS, ACTION]}

    print("******************************************************************")

    obs_message = _observation_msg

    # print(obs_message)

    last_observations = _process_observations(obs_message, result)

    # print(last_observations)

    ee_link_jacobians, jac_solver = _get_jacobians(last_observations[:3], jac_solver)

    # print "ee_link_jacobians: ", ee_link_jacobians
    # print "ur_chain: ", ur_chain

    trans, rot = forward_kinematics(ur_chain,
                                    LINK_NAMES,
                                    last_observations[:3],
                                    base_link=LINK_NAMES[0],
                                    end_link=LINK_NAMES[-1])
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rot
    rotation_matrix[:3, 3] = trans

    current_quaternion = quaternion_from_matrix(rotation_matrix)

    current_ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS,
                                                      trans,
                                                      rot).T)
    # file.write(str(current_ee_tgt[0]) + ", " + str(current_ee_tgt[1]) + ", " + str(current_ee_tgt[2]) + str("\n"))
    # print current_ee_tgt
    ee_points = current_ee_tgt - ee_tgt

    ee_points_jac_trans, _ = _get_ee_points_jacobians(ee_link_jacobians,
                                                           EE_POINTS,
                                                           rot)

    ee_velocities = _get_ee_points_velocities(ee_link_jacobians,
                                                   EE_POINTS,
                                                   rot,
                                                   last_observations[3:])
    euc_distance = np.linalg.norm(ee_points.reshape(-1, 3), axis=1)
    print(euc_distance)

    observation = np.array([np.r_[np.reshape(last_observations, -1),
                  np.reshape(ee_points, -1),
                  np.reshape(ee_velocities, -1),]])

    feed_dict = {nn_input: observation}
    output = sess.run(out, feed_dict=feed_dict)

    # SHOULDER_PAN = output[0][0]
    # SHOULDER_LIFT = output[0][1]
    # ELBOW = output[0][2]


    msg = _get_ur_trajectory_message(output[0], RESET_SLOWNESS)

    print(msg)

    _pub.publish(msg)
file.close()
