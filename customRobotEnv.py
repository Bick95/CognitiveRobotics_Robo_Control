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
import random
import pybullet_data
from pkg_resources import parse_version
from scipy.spatial import distance

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

# TODO 1: getting average nr grasps per 10 updates + times
# TODO 2: define actual reward measure/function for final evaluation

class PandaRobotEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=5,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=1000,
                 fixedActionRepetitions=False,
                 distSpecifications=None,
                 maxDist=0.25,
                 maxDeviation=0.25):

        if distSpecifications is None:
            distSpecifications = [0, 'A']  # 0 = Euclidean distance, A = Use improved distance metric

        # Parameter settings
        self._distance_measure_specifications = distSpecifications
        self._fixed_nr_action_repetitions = fixedActionRepetitions
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._maxDist = maxDist
        self._maxDeviation = maxDeviation

        # Renndering
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        # Observations & Measurements
        self._envStepCounter = 0
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []
        self._observation = []
        self._joint_pos = []
        self._goal_pos = []
        self._gripper_pos = []
        self._gripper_orn_vec = []
        self._gripper_to_goal_vec = []
        self._dist_to_obj_primary = 0.0             # Metric used for determination of terminal states
        self._dev_from_goal_vec_primary = 0.0       # Metric used for determination of terminal states
        self._dist_to_obj_secondary = 0.0           # Metric used for reward computation
        self._dev_from_goal_vec_secondary = 0.0     # Metric used for reward computation
        self._reward_dist = 0.0                     # self._dist_to_obj_secondary translated to a reward
        self._reward_goal_dir = 0.0                 # self._dev_from_goal_vec_secondary translated to a reward

        # Simulation
        self._urdfRoot = urdfRoot
        self._trayUid = None
        self.blockUid = None

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
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0, 0], useFixedBase=1)
        self._num_joints = 7    # Nr of controlled joints
        self._gripperIndex = 9  # Index of end-effector-link for Franka Emika Panda
        self.reset()

        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)

        # Network may have to predict number of repetitions how many times a joint config is to be applied in a row by
        # means of an additional action node:
        additional_action_node = 0 if self._fixed_nr_action_repetitions else 1

        if self._isDiscrete:
            self.action_space = spaces.Discrete(3+additional_action_node)
        else:
            action_dim = self._num_joints+additional_action_node
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    ##################################
    # Further added helper functions #
    ##################################

    def divergence(self, a, b):
        """
            Distance measure:
            Computed Divergence of two points. Formula (45) from following paper:
                http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.154.8446&rep=rep1&type=pdf

            :param a: Data point a
            :param b: Data point b (has to have same length as a)
        """

        # if not len(a) == len(b):
        #    raise AssertionError

        d = 0
        for i in range(len(a)):
            d += (((a[i] - b[i]) ** 2) / ((a[i] + b[i]) ** 2))
        d *= 2

        return d

    def get_random_joint_config(self, nr_joints=None):
        """
            Returns a random joint angle in range [-1,1] in radians. One random angle is generated for each controlled joint
            from joint 1 through nr_joints, where nr_joints is the number of joints for which a random config has to be
            returned.

            :param nr_joints: Random joint angles are generated for joints 1 through nr_joints. Optional.
            :return: List of joint angles. One random joint angle in radians per requested joint
        """
        if nr_joints is None:
            nr_joints = self._num_joints
        return [random.uniform(-1, 1) for _ in range(nr_joints)]

    def apply_joint_config(self, config):
        """
            Instantaneously apply a given joint configuration to a robotic arm.

            :param config: list of joint angles to be applied to joints 1 through n, respectively;
                           n=number of joint angles provided
            :return: -
        """
        for i in range(len(config)):
            self._p.resetJointState(self._robot, i, config[i])
        # time.sleep(1)  # Observe pose

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

    def euler_to_vec_gripper_orientation(self, yaw, pitch=None, roll=None):
        """
           Returns the orientation of the z-axis of the gripper with respect to the world/universe-reference
           coordinate system. The z-axis of the gripper is the one pointing along the direction of the fingers
           of the gripper.
        """
        if isinstance(yaw, tuple):
            yaw = list(yaw)
        if isinstance(yaw, list):
            # list of angles is provided in yaw-variable
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

    def obtain_measurements(self):
        """
            Obtain general purpose measurements like positions of goal object and gripper's center of mass as well as
            measures used for later reward computations.

            :return: -
        """

        ### Obtain general measurements:

        ## Obtain robot's joint space positions for all controlled joints in [joint{1}, joint{nr_controlled_joints}]
        self._joint_pos = [j[0] for j in self._p.getJointStates(self._robot, range(self._num_joints))]


        ## Calculate measure of how close the end-effector (=gripper's Center of Mass (COM)) is to the goal location:
        # Obtain world information
        self._goal_pos, _ = p.getBasePositionAndOrientation(self.blockUid)
        self._gripper_pos, gripperOrn_quat = p.getLinkState(self._robot, self._gripperIndex)[0:2]

        # Euclidean/Cartesian (straight-line) distance calculation from gripper's COM to goal's COM coordinates
        euclideanDistance = distance.euclidean(self._gripper_pos, self._goal_pos)


        ## Measure of how precisely end-effector (=gripper's fingers) points towards goal location:
        # Calculate vectors
        gripperOrn_eul = self._p.getEulerFromQuaternion(gripperOrn_quat)
        self._gripper_to_goal_vec = self.get_normalized_vector_from_a_to_b(self._gripper_pos, self._goal_pos)
        self._gripper_orn_vec = self.euler_to_vec_gripper_orientation(gripperOrn_eul)
        self._gripper_orn_vec = self.normalize_vector(self._gripper_orn_vec)

        # Euclidean distance between vector pointing straight from gripper's COM location towards goal's COM
        # location and vector containing direction of z-axis of coordinate system expressing the orientation of COM
        # of end-effector (expressed with respect to reference frame attached to base of robotic arm).
        # (Z-axis of end-effector frame points from end-effector's COM towards its fingers.)
        euclideanDeviation = distance.euclidean(self._gripper_to_goal_vec, self._gripper_orn_vec)


        ### Obtain reward measurements:

        ## Distance based reward signal

        # Compute distance measure underlying continuous distance-based reward signal
        if 0 in self._distance_measure_specifications:
            # Euclidean distance
            newRewardDistance = euclideanDistance
        elif 1 in self._distance_measure_specifications:
            # Divergence metric
            newRewardDistance = self.divergence(self._gripper_pos, self._goal_pos)
        else:
            newRewardDistance = 0.0

        # Translate variable distance measure computed above into a reward:
        if 'A' in self._distance_measure_specifications:
            # Reward the reduction of goal distance compared to that of previous time-step
            self._reward_dist = self._dist_to_obj_secondary - newRewardDistance
        elif 'B' in self._distance_measure_specifications:
            # Reward the absolute negative distance from gripper to goal
            self._reward_dist = -newRewardDistance
        elif 'C' in self._distance_measure_specifications:
            # No continuous distance based reward signal
            self._reward_dist = 0.0
        else:
            pass

        ## Deviation based reward signal

        # Compute deviation measure underlying continuous deviation-from-goal-direction-based reward signal
        if 0 in self._distance_measure_specifications:
            # Euclidean distance
            newRewardDeviation = euclideanDeviation
        elif 1 in self._distance_measure_specifications:
            # Divergence metric
            newRewardDeviation = self.divergence(self._gripper_to_goal_vec, self._gripper_orn_vec)
        else:
            newRewardDeviation = 0.0

        # Translate variable deviation-from-goal-direction measure computed above into a reward:
        if 'A' in self._distance_measure_specifications:
            # Reward reduction of deviance from gripper's orientation to vector pointing to goal compared to that of
            # previous time-step
            self._reward_goal_dir = self._dev_from_goal_vec_secondary - newRewardDeviation
        elif 'B' in self._distance_measure_specifications:
            # Reward the absolute negative deviation of gripper's orientation from vector pointing towards goal
            self._reward_goal_dir = -newRewardDeviation
        elif 'C' in self._distance_measure_specifications:
            # No continuous deviation based reward signal
            self._reward_goal_dir = 0.0
        else:
            pass


        self._dist_to_obj_primary = euclideanDistance
        self._dist_to_obj_secondary = newRewardDistance
        self._dev_from_goal_vec_primary = euclideanDeviation
        self._dev_from_goal_vec_secondary = newRewardDeviation


    def reset_logged_train_data(self):
        """
            For logging training progress. Called via PPO agent's callback.
            :return: -
        """
        self.grasps_per_update_interval = 0
        self.grasp_time_steps_needed_per_update_interval = []

    ##############################
    # End added helper functions #
    ##############################


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
        self._robot = p.loadURDF(self._robo_path, basePosition=[0, 0, 0], useFixedBase=1)
        self._envStepCounter = 0

        # Set robotic arm to random initial pose
        self.apply_joint_config(self.get_random_joint_config())

        p.stepSimulation()
        self.obtain_measurements()
        self._observation = self.getExtendedObservation()

        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        """
            Construct an observation that serves as input to the reinforcement learning agent.
            :return: Representation of state of robotic arm and its surrounding
        """
        self._observation = []
        self._observation.extend(self._goal_pos)               # Cartesian position of goal
        self._observation.extend(self._gripper_pos)            # Cartesian position of end-effector
        self._observation.extend(self._gripper_to_goal_vec)    # Normalized vector from gripper to goal
        self._observation.extend(self._gripper_orn_vec)        # Orientation of gripper's z-axis wrt reference frame
        self._observation.extend(self._joint_pos)              # Joint angles

        return self._observation

    def step(self, action):

        # Determine how many times in a row commanded actions are supposed to be executed. Min=1, Max=10 times.
        if self._fixed_nr_action_repetitions:
            repetitions = self._actionRepeat
        else:
            minRep, maxRep = 1, 10
            repetitions = min(maxRep, max(minRep, int(abs(action[-1]*10))))

        # Get joint angles for all controlled joints
        jointPos = self._joint_pos.copy()

        # Determine new joint configurations. Clip actions and increment current joint angles by clipped action commands
        jointPos = [jointPos[i] + np.clip(action[i], -0.5, 0.5) for i in range(self._num_joints)]

        # Set new, desired joint configurations
        self._p.setJointMotorControlArray(bodyUniqueId=self._robot,
                                          jointIndices=range(self._num_joints),
                                          controlMode=self._p.POSITION_CONTROL,
                                          targetPositions=jointPos,
                                          forces=[50]*self._num_joints)

        # Step a simulations given number of times to get old joint configurations closer to the new, desired ones
        for i in range(repetitions):

            p.stepSimulation()
            self.obtain_measurements()

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

        if self._dist_to_obj_primary < self._maxDist and self._dev_from_goal_vec_primary < self._maxDeviation:
            # Goal attained
            self.terminated = 1
            self._observation = self.getExtendedObservation()
            # Log training progress
            self.grasps_per_update_interval += 1
            self.grasp_time_steps_needed_per_update_interval.append(self._envStepCounter)
            return True

        return False

    def _reward(self):

        reward = 0

        # Distance reward
        # Distance needs adjustment only when being too distant from goal
        if self._dist_to_obj_primary > self._maxDist:
            reward += self._reward_dist

        # Orientation reward
        # Orientation needs adjustment only when being too far from desired orientation
        if self._dev_from_goal_vec_primary > self._maxDeviation:
            reward += self._reward_goal_dir

        # Terminal state rewards
        if self._dist_to_obj_primary < self._maxDist and self._dev_from_goal_vec_primary < self._maxDeviation:
            # Goal reached reward
            goalReward = 2
            timeReward = 200 / self._envStepCounter  # If successful done after 200 time steps, then another +1
            reward += (goalReward + timeReward)

            #print(
            #    "#######\n#######\n#######\n#######\nsuccessfully grasped a block!!!\n#######\n#######\n#######\n#######")
            #print('Current step-ctr: ' + str(self._envStepCounter))
            #print('Cart dist: ' + str(self._dist_to_obj))
            #print('Dir. devi: ' + str(self._dev_from_goal_vec))
            #time.sleep(1)

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
