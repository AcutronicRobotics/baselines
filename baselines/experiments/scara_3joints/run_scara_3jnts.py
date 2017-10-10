import numpy as np
import sys

import argparse
import copy

sys.path.append('/home/rkojcev/devel/baselines')
from baselines.agent.scara_arm.agent_scara import AgentSCARAROS
from baselines import logger
from baselines.common import set_global_seeds

from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction
from baselines.agent.utility.general_utils import get_ee_points, get_position


# from gym import utils
# from gym.envs.mujoco import mujoco_env

class ScaraJntsEnv(AgentSCARAROS):

    # agent_scara.AgentSCARAROS.__init__(self, 'tests.xml')

    def __init__(self):
        print("I am in init function")
        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'
        # where should the agent reach
        EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.7112])
        EE_ROT_TGT = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        EE_POINTS = np.asmatrix([[0, 0, 0]])

        #add here the joint names:
        MOTOR1_JOINT = 'motor1'
        MOTOR2_JOINT = 'motor2'
        MOTOR3_JOINT = 'motor3'

        # Set constants for links
        WORLD = "world"
        BASE = 'scara_e1_base_link'
        BASE_MOTOR = 'scara_e1_base_motor'

        SCARA_MOTOR1 = 'scara_e1_motor1'
        SCARA_INSIDE_MOTOR1 = 'scara_e1_motor1_inside'
        SCARA_SUPPORT_MOTOR1 = 'scara_e1_motor1_support'
        SCARA_BAR_MOTOR1 = 'scara_e1_bar1'
        SCARA_FIXBAR_MOTOR1 = 'scara_e1_fixbar1'

        SCARA_MOTOR2 = 'scara_e1_motor2'
        SCARA_INSIDE_MOTOR2 = 'scara_e1_motor2_inside'
        SCARA_SUPPORT_MOTOR2 = 'scara_e1_motor2_support'
        SCARA_BAR_MOTOR2 = 'scara_e1_bar2'
        SCARA_FIXBAR_MOTOR2 = 'scara_e1_fixbar2'

        SCARA_MOTOR3 = 'scara_e1_motor3'
        SCARA_INSIDE_MOTOR3 = 'scara_e1_motor3_inside'
        SCARA_SUPPORT_MOTOR3 = 'scara_e1_motor3_support'
        SCARA_BAR_MOTOR3 = 'scara_e1_bar3'
        SCARA_FIXBAR_MOTOR3 = 'scara_e1_fixbar3'

        SCARA_RANGEFINDER = 'scara_e1_rangefinder'

        EE_LINK = 'ee_link'

        JOINT_ORDER = [MOTOR1_JOINT, MOTOR2_JOINT, MOTOR3_JOINT]
        LINK_NAMES = [BASE, BASE_MOTOR,
                      SCARA_MOTOR1, SCARA_INSIDE_MOTOR1, SCARA_SUPPORT_MOTOR1, SCARA_BAR_MOTOR1, SCARA_FIXBAR_MOTOR1,
                      SCARA_MOTOR2, SCARA_INSIDE_MOTOR2, SCARA_SUPPORT_MOTOR2, SCARA_BAR_MOTOR2, SCARA_FIXBAR_MOTOR2,
                      SCARA_MOTOR3, SCARA_INSIDE_MOTOR3, SCARA_SUPPORT_MOTOR3,
                      EE_LINK]
        # Set end effector constants
        INITIAL_JOINTS = np.array([0, 0, 0, 0, 0, 0])
        # where is your urdf?
        TREE_PATH = '/home/rkojcev/catkin_ws/src/scara_e1/scara_e1_description/urdf/scara_e1_3joints.urdf'

        STEP_COUNT = 100  # Typically 100.

        # Set the number of seconds per step of a sample.
        TIMESTEP = 0.01  # Typically 0.01.
        # Set the number of timesteps per sample.
        STEP_COUNT = 100  # Typically 100.
        # Set the number of samples per condition.
        SAMPLE_COUNT = 5  # Typically 5.
        # set the number of conditions per iteration.
        CONDITIONS = 1  # Typically 2 for Caffe and 1 for LQR.
        # Set the number of trajectory iterations to collect.
        ITERATIONS = 20  # Typically 10.

        m_joint_order = copy.deepcopy(JOINT_ORDER)
        m_link_names = copy.deepcopy(LINK_NAMES)
        m_joint_publishers = copy.deepcopy(JOINT_PUBLISHER)
        m_joint_subscribers = copy.deepcopy(JOINT_SUBSCRIBER)

        ee_pos_tgt = EE_POS_TGT
        ee_rot_tgt = EE_ROT_TGT

            # Initialize target end effector position
        ee_tgt = np.ndarray.flatten(get_ee_points(EE_POINTS, ee_pos_tgt, ee_rot_tgt).T)
        # States to check in agent._process_observations.
        # STATE_TYPES = {'positions': JOINT_ANGLES,
        #        'velocities': JOINT_VELOCITIES}

        agent = {
            'type': AgentSCARAROS,
            'dt': TIMESTEP,
            # 'dU': SENSOR_DIMS[ACTION],
            # 'conditions': common['conditions'],
            'T': STEP_COUNT,
            # 'x0': x0s,
            'ee_points_tgt': ee_tgt,
            # 'reset_conditions': reset_conditions,
            # 'sensor_dims': SENSOR_DIMS,
            'joint_order': m_joint_order,
            'link_names': m_link_names,
            # 'state_types': STATE_TYPES,
            'tree_path': TREE_PATH,
            'joint_publisher': m_joint_publishers,
            'joint_subscriber': m_joint_subscribers,
            # 'state_include': [JOINT_ANGLES, JOINT_VELOCITIES,
            #                   END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            'end_effector_points': EE_POINTS,
            # 'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
            # 'node_suffix': ROS_NODE_SUFFIX,
            'num_samples': SAMPLE_COUNT,
        }
        # utils.EzPickle.__init__(self)
        AgentSCARAROS.__init__(self, agent)
        AgentSCARAROS._run_trial(self, agent)

    # def _step(self, a):
    #     vec = self.get_body_com("fingertip")-self.get_body_com("target")
    #     reward_dist = - np.linalg.norm(vec)
    #     reward_ctrl = - np.square(a).sum()
    #     reward = reward_dist + reward_ctrl
    #     self.do_simulation(a, self.frame_skip)
    #     ob = self._get_obs()
    #     done = False
    #     return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
    #
    # def viewer_setup(self):
    #     self.viewer.cam.trackbodyid = 0
    #
    # def reset_model(self):
    #     qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
    #     while True:
    #         self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
    #         if np.linalg.norm(self.goal) < 2:
    #             break
    #     qpos[-2:] = self.goal
    #     qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
    #     qvel[-2:] = 0
    #     self.set_state(qpos, qvel)
    #     return self._get_obs()
    #
    # def _get_obs(self):
    #     theta = self.model.data.qpos.flat[:2]
    #     return np.concatenate([
    #         np.cos(theta),
    #         np.sin(theta),
    #         self.model.data.qpos.flat[2:],
    #         self.model.data.qvel.flat[:2],
    #         self.get_body_com("fingertip") - self.get_body_com("target")
    #     ])

if __name__ == "__main__":
    ScaraJntsEnv()
    parser = argparse.ArgumentParser(description='Run Gazebo benchmark.')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default="Reacher-v1")
    args = parser.parse_args()
