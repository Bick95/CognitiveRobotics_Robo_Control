import os, inspect, random, sys


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from pybullet_envs.bullet import kuka
import random
import pybullet_data
from pkg_resources import parse_version
from scipy.spatial import distance

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class CustomRobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=1000):
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._last_dist_to_obj = 0.0
        self._last_dev_from_goal_vec = 0.0

        self._p = p
        #self._robo_path = 'RobotModels/Panda/deps/Panda/panda.urdf'
        self._robo_path = 'RobotModels/Pybullet_Robots/data/franka_panda/panda.urdf'

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self.seed()
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0, 0],
                                 useFixedBase=1)
        self._num_joints = 7    # nr of controlled joints
        self._gripperIndex = 9  #self._p.getNumJoints(self._robot)
        #print('Num joints: ' + str(self._num_joints))
        #print('Gripper idx: ' + str(self._gripperIndex))
        self.reset()

        # CHECK FOR JOINT NAMES AND THEIR ASSOCIATED INDICES
        #_link_name_to_index = {p.getBodyInfo(self._robot)[0].decode('UTF-8'): -1, }
        #for _id in range(p.getNumJoints(self._robot)):
        #    _name = p.getJointInfo(self._robot, _id)[12].decode('UTF-8')
        #    _link_name_to_index[_name] = _id
        #print(_link_name_to_index)
        #time.sleep(100)

        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)
        if self._isDiscrete:
            self.action_space = spaces.Discrete(3+1)
        else:
            action_dim = self._num_joints+1
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def get_random_joint_config(self, nr_joints=None):
        if nr_joints is None:
            nr_joints = self._num_joints
        return [random.uniform(-1, 1) for _ in range(nr_joints)]

    def apply_actions(self, config):
        for i in range(self._num_joints):
            self._p.resetJointState(self._robot, i, config[i])
        time.sleep(1)

    def reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                   0.000000, 0.000000, 0.0, 1.0)

        self._trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.640000,
                                   0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                                   orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0, 0],
                                 useFixedBase=1)
        self._envStepCounter = 0

        self.apply_actions(self.get_random_joint_config())

        p.stepSimulation()
        self._observation = self.getExtendedObservation()

        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ##################################
    # Further added helper functions #
    ##################################

    def get_normalized_vector_from_a_to_b(self, a, b):
        vec = []
        len_of_vec = 0
        for i in range(len(a)):
            entry = b[i] - a[i]
            vec.append(entry)
            len_of_vec += abs(entry)
        for i in range(len(vec)):
            vec[i] /= len_of_vec
        return vec

    def euler_to_gripper_orientation(self, yaw, pitch=None, roll=None):
        """
           Returns the orientation of the z-axis of the gripper with respect to the world/universe-reference
           coordinate system. The z-axis of the gripper is the one pointing along the direction of the fingers
           of the gripper.
        """
        if isinstance(yaw, tuple):
            yaw = list(yaw)
        if isinstance(yaw, list):
            # list of angles is provided
            pitch, roll, yaw = yaw[1], yaw[2], yaw[0]

        # Construct rotation matrices
        sin = math.sin
        cos = math.cos
        Rx_yaw = np.array([[1, 0, 0], [0, cos(yaw), -sin(yaw)], [0, sin(yaw), cos(yaw)]])
        Rz_rol = np.array([[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]])
        Ry_pit = np.array([[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]])

        # Rotate a coord system initially coincident with ref frame in exact same way as roll/pitch/yaw are applied in
        # simulation. To get the orientation of the axes defining the coord system attached to COM (=Center of mass) of
        # end-effector (=gripper) expressed with respect to ref system.
        R = Rz_rol.dot(Ry_pit.dot(Rx_yaw))

        ee_z_axis = R[:, 2]  # Direction of z-axis of coord system expressing gripper orientation wrt reference frame

        return ee_z_axis

    def normalize_vector(self, vec):
        len_of_vec = 0
        for i in range(len(vec)):
            len_of_vec += abs(vec[i])
        for i in range(len(vec)):
            vec[i] /= len_of_vec
        return vec

    ##############################
    # End added helper functions #
    ##############################

    def getExtendedObservation(self):

        # TODO: introduce standard for variable names! Refactor code...

        self._observation = []

        # Robot's joint space pos
        jointPos = [j[0] for j in self._p.getJointStates(self._robot, range(self._num_joints))]

        # Robot's Cartesian end-effector pos
        world_position, world_ori_quat = self._p.getLinkState(self._robot, self._gripperIndex)[0:2]
        cartePos = world_position[:3]
        gripperOrn_euler = self._p.getEulerFromQuaternion(world_ori_quat)

        # Block's Cartesian position
        blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)

        # Robot's gripper's orientation expressed in directional vector
        gripperOrn_vec = self.euler_to_gripper_orientation(gripperOrn_euler)
        gripperOrn_vec = self.normalize_vector(gripperOrn_vec)

        # Direction from gripper to goal
        gripper_to_goal_vec = self.get_normalized_vector_from_a_to_b(cartePos, blockPos)

        # Construct observation
        self._observation.extend(blockPos)              # Cartesian position of goal
        self._observation.extend(cartePos)              # Cartesian position of end-effector
        self._observation.extend(gripper_to_goal_vec)   # Normalized vector from gripper to goal
        self._observation.extend(gripperOrn_vec)        # Orientation of gripper's z-axis wrt reference frame
        self._observation.extend(jointPos)              # Joint angles

        return self._observation

    def step(self, action):
        #repetitions = int(abs(action[-1]*10))
        repetitions = 5
        print('Actions: ', end='\t\t')
        print(action)
        print('Repetitions: ', end='\t')
        print(repetitions)

        jointPos = [state[0] for state in self._p.getJointStates(self._robot, range(self._num_joints))]
        jointPos = [jointPos[i] + np.clip(action[i], -0.5, 0.5) for i in range(self._num_joints)]


        self._p.setJointMotorControlArray(bodyUniqueId=self._robot,
                                          jointIndices=range(self._num_joints),
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=jointPos,
                                          forces=[50]*self._num_joints)

        for i in range(repetitions):  # self._actionRepeat

            p.stepSimulation()

            if self._termination:
                break
            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        done = self._termination

        reward = self._reward()

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._robot)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=0,
                                                                upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    @property
    def _termination(self):

        if self.terminated or self._envStepCounter > self._maxSteps:
            self._observation = self.getExtendedObservation()
            return True

        # Obtain measurements
        blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)
        gripperPos, gripperOrn_quat = p.getLinkState(self._robot, self._gripperIndex)[0:2]
        gripperOrn_euler = self._p.getEulerFromQuaternion(gripperOrn_quat)

        # Cartesian distance calculation from gripper to goal:
        cartDistance = distance.euclidean(blockPos, gripperPos)

        # Directional vector deviation calculation
        # (Deviation of direction of z-axis of gripper from normalized directional vector from gripper towards goal)
        gripperToGoal_vec = self.get_normalized_vector_from_a_to_b(gripperPos, blockPos)
        gripperOrn_vec = self.euler_to_gripper_orientation(gripperOrn_euler)
        gripperOrn_vec = self.normalize_vector(gripperOrn_vec)
        directionalDeviation = distance.euclidean(gripperToGoal_vec, gripperOrn_vec)


        maxDist, maxDeviation = 0.25, 0.25
        if cartDistance < maxDist and directionalDeviation < maxDeviation:
            self.terminated = 1
            self._observation = self.getExtendedObservation()
            return True

        return False

    def _reward(self):

        # Obtain measurements
        blockPos, _ = p.getBasePositionAndOrientation(self.blockUid)
        gripperPos, gripperOrn_quat = p.getLinkState(self._robot, self._gripperIndex)[0:2]
        gripperOrn_euler = self._p.getEulerFromQuaternion(gripperOrn_quat)

        # Cartesian distance calculation from gripper to goal:
        cartDistance = distance.euclidean(blockPos, gripperPos)

        # Directional vector deviation calculation
        # (Deviation of direction of z-axis of gripper from normalized directional vector from gripper towards goal)
        gripperToGoal_vec = self.get_normalized_vector_from_a_to_b(gripperPos, blockPos)
        gripperOrn_vec = self.euler_to_gripper_orientation(gripperOrn_euler)
        gripperOrn_vec = self.normalize_vector(gripperOrn_vec)
        directionalDeviation = distance.euclidean(gripperToGoal_vec, gripperOrn_vec)

        # Reward computation:
        reward = 0

        # Distance reward
        maxDist = 0.25
        if cartDistance > maxDist:
            # Only when being too distant from goal, distance needs adjustment
            reward += self._last_dist_to_obj - cartDistance
        self._last_dist_to_obj = cartDistance

        # Orientation reward
        maxDeviation = 0.25
        if directionalDeviation > maxDeviation:
            # Only when being too far from desired orientation, orientation needs adjustment
            reward += self._last_dev_from_goal_vec - directionalDeviation
        self._last_dev_from_goal_vec = directionalDeviation

        # Terminal state rewards
        if cartDistance < maxDist and directionalDeviation < maxDeviation:
            # Goal reached reward
            print('Cart dist: ' + str(cartDistance))
            print('Dir. devi: ' + str(directionalDeviation))
            goalReward = 2
            timeReward = 200 / self._envStepCounter  # if done after 200 time steps, then +1
            reward += (goalReward + timeReward)
            print(
                "#######\n#######\n#######\n#######\nsuccessfully grasped a block!!!\n#######\n#######\n#######\n#######")
            print('Current step-ctr: ' + str(self._envStepCounter))
            time.sleep(5)
        elif self._envStepCounter > self._maxSteps:
            # Goal not reached punishment
            reward -= 2

        print('Current step-ctr: ' + str(self._envStepCounter))

        print('Reward: ' + str(reward))
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
