import os, inspect

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

# TODO: explore setRealTimeSimulation
# TODO: put in franka model and free from Kuka stuff

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
        self._gripperIndex = 9

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self.seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)

        observation_high = np.array([largeValObservation] * observationDim)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(9+1)
        else:
            action_dim = 9+1
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                   0.000000, 0.000000, 0.0, 1.0)

        self.trayUid = p.loadURDF(os.path.join(self._urdfRoot, "tray/tray.urdf"), 0.640000,
                                  0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)

        xpos = 0.55 + 0.12 * random.random()
        ypos = 0 + 0.2 * random.random()
        ang = 3.14 * 0.5 + 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                                   orn[0], orn[1], orn[2], orn[3])

        p.setGravity(0, 0, -10)
        self._kuka = p.loadURDF('../../Robot/pybullet_robots/data/franka_panda/panda.urdf', basePosition=[0, 0, 0],
                              useFixedBase=1) #kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):

        self._observation = []

        # Robot's joint space pos
        jointPos = [j[0] for j in self._p.getJointStates(self._kuka, range(9))]
        #print('Joint pos: ', end=' ')
        #print(jointPos)
        world_position = self._p.getLinkState(self._kuka, 9)[0]
        cartePos = world_position[:3]
        #print('Cart pos: ', end=' ')
        #print(cartePos)

        # Block's Cartesian position
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        #print('Block pos: ', end=' ')
        #print(blockPos)

        self._observation.extend(blockPos)
        self._observation.extend(cartePos)
        self._observation.extend(jointPos)
        #print('Observaton:')
        #print(self._observation)
        return self._observation

    def step(self, action):
        repetitions = int(abs(action[-1]*10))
        print(action, repetitions)
        for i in range(repetitions):  # self._actionRepeat
            print(action[:-1])
            self._p.setJointMotorControlArray(self._kuka, range(9), self._p.POSITION_CONTROL,
                                              targetPositions=action[:-1])
            p.stepSimulation()
            if self._termination:
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()

        # print("self._envStepCounter")
        # print(self._envStepCounter)

        done = self._termination

        reward = self._reward()
        time.sleep(0.1)

        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka)
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
        # print (self._kuka.endEffectorPos[2])
        state = p.getLinkState(self._kuka, self._gripperIndex)
        actualEndEffectorPos = state[0]

        # print("self._envStepCounter")
        # print(self._envStepCounter)
        if self.terminated or self._envStepCounter > self._maxSteps:
            self._observation = self.getExtendedObservation()
            return True
        maxDist = 0.005
        closestPoints = p.getClosestPoints(self.trayUid, self._kuka, maxDist)

        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        gripperPos, gripperOrn = p.getLinkState(self._kuka, self._gripperIndex)[0:2]

        # Implements: numpy.sqrt(numpy.sum((numpy.array(a)-numpy.array(b))**2))
        cartDistance = distance.euclidean(blockPos, gripperPos)

        if cartDistance < 0.2:
            print('##########\n'*2 + 'GOAL REACHED!' + '##########\n'*2)
            self.terminated = 1
            return True
        return False

    def _reward(self):
        reward = 0

        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        gripperPos, gripperOrn = p.getLinkState(self._kuka, self._gripperIndex)[0:2]

        # Implements: numpy.sqrt(numpy.sum((numpy.array(a)-numpy.array(b))**2))
        cartDistance = distance.euclidean(blockPos, gripperPos)

        distChange = self._last_dist_to_obj - cartDistance
        self._last_dist_to_obj = cartDistance
        reward += -cartDistance/10 + distChange

        if blockPos[2] > 0.2:
            # According to provided implementation: Gripper has reached target!
            goalReward = 1
            timeReward = 1/self._envStepCounter
            reward += (goalReward + timeReward)
            print("#######\n#######\n#######\n#######\nsuccessfully grasped a block!!!\n#######\n#######\n#######\n#######")
            time.sleep(5)

        print('Current step-ctr: ' + str(self._envStepCounter))

        print('Reward: ' + str(reward))
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
