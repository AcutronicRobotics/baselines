import numpy as np
import sys

import argparse

sys.path.append('/home/rkojcev/devel/baselines')
from baselines.agent.scara_arm import agent_scara
from baselines import logger
from baselines.common import set_global_seeds

from baselines.acktr.acktr_cont import learn
from baselines.acktr.policies import GaussianMlpPolicy
from baselines.acktr.value_functions import NeuralNetValueFunction


# from gym import utils
# from gym.envs.mujoco import mujoco_env

class ScaraJntsEnv(agent_scara.AgentSCARAROS):

    # agent_scara.AgentSCARAROS.__init__(self, 'tests.xml')

    def __init__(self):
        print("I am in init function")
        # Topics for the robot publisher and subscriber.
        JOINT_PUBLISHER = '/scara_controller/command'
        JOINT_SUBSCRIBER = '/scara_controller/state'
        # where should the agent reach
        EE_POS_TGT = np.asmatrix([0.3325683, 0.0657366, 0.7112])

        #add here the joint names:
        # Set constants for joints
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
        TREE_PATH = '/home/rkojcev/catkin_ws/src/scara_e1/scara_e1_description/urdf/scara_e1_3joints.urdf'

        STEP_COUNT = 100  # Typically 100.
        # utils.EzPickle.__init__(self)
        agent_scara.AgentSCARAROS.__init__(self, TREE_PATH)
        agent_scara.AgentSCARAROS._run_trial(self)

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
