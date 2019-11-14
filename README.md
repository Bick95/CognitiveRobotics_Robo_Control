# CognitiveRobotics_Robo_Control

## Introduction
Welcome to the **CognitiveRobotics_Robo_Control** repository!

This repository contains the code developed for the final project of the course **Cognitive Robotics** offered at the **University of Groningen** during the academic year 2019-2020.

The goal of the project can be summarized as follows:
<ul>
<li>Trying to come up with an innovative robotic arm trajectory generating controller.</li>
</ul>

More precisely, the attempt was made to to train an Reinforcement Learning (RL) agent to control a robotic arm in joint space without having to perform possibly expensive Inverse Kinematics operations.
The goal of the controller is to compute changes to be applied to the current joint angles of all controlled joints in order to make the robotic arm attain a given goal location and to have the fingers of the arm's end#effector's gripper point towards the goal location.
The RL agent was only given the following information:
<ul>
<li>The goal position in Cartesian space</li>
<li>Its end-effector's Center of Mass'es (COM) Cartesian position</li>
<li>The normalized vector expressing the orientation of the end-effector's fingers</li>
<li>The normalized vector pointing from the end-effector's COM towards the goal location</li>
<li>The set of the robot's current joint angles in radians</li>
</ul>
All measurements were taken with respect to a universal coordinate system.


In order to achieve this goal, the physics simulation software [Pybullet](https://pybullet.org/wordpress/) is employed, in which a [Franka Emika Panda](https://www.franka.de/technology), i.e. a robotic arm, is simulated.
The simulated arm is controlled by a [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) Reinforcement Learning agent, where the implementation of the PPO algorithm is provided by [Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/index.html).
A customized [Gym environment](https://gym.openai.com/), called **PandaRobotEnv** and defined in '''customRobotEnv.py''', acts as a bridge between the Pybullet simulation and the PPO agent.
The model of the robotic arm is provided by [pybullet_robots](https://github.com/erwincoumans/pybullet_robots).
For designing the aforementioned PandaRobotEnv, the **KukaGymEnv**, included in this repository as a reference environment and originally shipped with the Pybullet installation, served as inspiration for designing some core functions.
However, except for the rendering functionality, where just a few parameters were adapted, and some variable names, the used Gym environment's functionality has been thoroughly redesigned and augmented in order to meet our custom goals and to be compatible with both Stable-Baselines' PPO implementation and the Franka Emika Panda.

This repository contains functionality to train PPO agents on controlling a Franka Emika Panda in joint using different reward functions and modes as well as to both visually render and record the performance of trained PPO agents performing their assigned task.

Furthermore, drawing upon a separate repository devoted to the evaluation of this project, which is included as a Git-submodule, the repository contains a set of trained agents, the evaluation of their training outcomes, and the functionality used to perform the evaluation.

An example video showing the evolution of the training progress of one trained PPO agent can be found on [YouTube]().

## Using the repository
### Installation/Setup
Software needed for running the code used in this project can be installed using pip as follows:

'''pip install tensorflow'''
'''pip install pybullet'''
'''pip install stable-baselines'''
'''pip install argparse'''

For recording videos of trained agents, an extra software is needed. Under Ubuntu, it can be installed by the following command:

'''sudo apt-get install ffmpeg'''

### Instructions

