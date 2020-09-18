# CognitiveRobotics_Robo_Control

## Introduction
Welcome to the **CognitiveRobotics_Robo_Control** repository!

This repository contains the code developed for the final project of the course **Cognitive Robotics** offered at the **University of Groningen** during the academic year 2019-2020.

The goal of the project can be summarized as follows:
<ul>
<li>Trying to come up with an innovative robotic arm trajectory generating controller.</li>
</ul>

More precisely, the attempt was made to to train an Reinforcement Learning (RL) agent to control a robotic arm in joint space without having to perform possibly expensive Inverse Kinematics operations.
The goal of the controller is to compute changes to be applied to the current joint angles of all controlled joints in order to make the robotic arm attain a given goal location and to have the fingers of the arm's end-effector's gripper point towards the goal location.
The RL agent was only given the following information:
<ul>
<li>The goal position in Cartesian space</li>
<li>Its end-effector's Center of Mass (COM) Cartesian position</li>
<li>The normalized vector expressing the orientation of the end-effector's fingers</li>
<li>The normalized vector pointing from the end-effector's COM towards the goal location</li>
<li>The set of the robot's current joint angles in radians</li>
</ul>
All measurements were taken with respect to a universal coordinate system.
As a reward signal, different reward functions were tested against each other.
More about the theoretical background can be found in the attached report (which is currently not yet available due to work in progress behind the scenes).


In order to achieve this goal, the physics simulation software [Pybullet](https://pybullet.org/wordpress/) is employed, in which a [Franka Emika Panda](https://www.franka.de/technology), i.e. a robotic arm, is simulated.
The simulated arm is controlled by a [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) Reinforcement Learning agent, where the implementation of the PPO algorithm is provided by [Stable-Baselines](https://stable-baselines.readthedocs.io/en/master/index.html).
A customized [Gym environment](https://gym.openai.com/), called **PandaRobotEnv** and defined in <code>customRobotEnv.py</code>, acts as a bridge between the Pybullet simulation and the PPO agent.
The model of the robotic arm is provided by [pybullet_robots](https://github.com/erwincoumans/pybullet_robots).
For designing the aforementioned PandaRobotEnv, the **KukaGymEnv**, included in this repository as a reference environment and originally shipped with the Pybullet installation, served as inspiration for designing some core functions.
However, the used Gym environment's functionality has been thoroughly redesigned and augmented in order to meet our custom goals and to be compatible with both Stable-Baselines' PPO implementation and the Franka Emika Panda. 

This repository contains functionality to train PPO agents on controlling a Franka Emika Panda in joint using different reward functions and modes as well as to both visually render and record the performance of trained PPO agents performing their assigned task.

Furthermore, drawing upon a separate repository devoted to the evaluation of this project, which is included as a Git-submodule, the repository contains a set of trained agents, the evaluation of their training outcomes, and the functionality used to perform the evaluation.

An example video showing the evolution of the training progress of one trained PPO agent can be found on [YouTube](https://youtu.be/MSssx5jxxAI).

## Using the repository
Note: The repository has been set up using Python 3.

### Installation/Setup
Software needed for running the code used in this project can be installed using pip as follows:

<code>pip install tensorflow</code>
<code>pip install pybullet</code>
<code>pip install stable-baselines</code>
<code>pip install argparse</code>

For recording videos of trained agents, an extra software is needed. Under Ubuntu, it can be installed by the following command:

<code>sudo apt-get install ffmpeg</code>

To load the included submodules containing the robot models, trained models, and evaluation data, one has to manually load them by executing the following command when loading them for the first time:

<code>git submodule update --init --recursive</code>

To get updated versions of the submodules at some later point, call:

<code>git submodule update --recursive --remote</code>

**Note**: In case of problems, check out [StackOverflow](https://stackoverflow.com/questions/1030169/easy-way-to-pull-latest-of-all-git-submodules)

### Instructions
In the following, the separate functionalities are are quickly introduced.
#### Training
For training a new or existing PPO agent, the <code>main.py</code> file can be used.
The file takes two optional arguments when being started:
<ul>
<li><code>-p</code>: A path to a json-file containing parameter specifications to be used for training.</li>
<li><code>-r</code>: A path to a trained model which is supposed to be loaded for the continuation of its training.
When continuing training, a new folder will be created and counting of weight updates starts at 0 again.
However, the trained model is used and the path to the read-in model will be recoded in the documentations of used parameters,
which are stored in both <code>params.csv</code> and <code>params.json</code></li>
</ul>

For the training process, a folder <code>Results/models_unique_folder</code> is created in the repo, where <code>models_unique_folder</code> is a unique identifier for each model and
the folder contains all data associated with the training process of the model. Checkpoints will be saved there, as well as documentation files etc.

Example: Starting training a new agent with parameter settings specified in the file <code>params_6.json</code>:
<code>python3 main.py -p ParameterSettings/params_6.json</code>

Example: Starting training a new agent with default parameter settings:
<code>python3 main.py</code>

#### Replaying
A trained model can be visually inspected using the <code>run_trained_model.py</code> file.
Starting the file, 0, 1 or 2 arguments can be provided.

Example: Observe how a given trained default model performs:

<code>python3 run_trained_model.py</code>

Example: Run a specific model provided to the code as an argument:

<code>python3 run_trained_model.py Evaluation_CognitiveRobotics_Robo_Control/Results/PPO2/PandaController_2019_08_11__15_41_05__262730fzyxnprhgl/final_model.zip</code>

Example: Run a specific model provided to the code as a first(!) argument for 1000 time steps given as a second(!) argument:

<code>python3 run_trained_model.py Evaluation_CognitiveRobotics_Robo_Control/Results/PPO2/PandaController_2019_08_11__15_41_05__262730fzyxnprhgl/final_model.zip 1000</code>

**Note**: In case that the data cannot be found, make sure to load the submodules (as described above).

#### Recording video sequence
<code>record_video_of_performing_trained_model.py</code> is the file to record video sequences of a trained agent.
It will create the file structure <code>VideoRecordings/model_name/Recording_date_some_info.mp4</code>.

It can be called without any arguments to record videos of a default model.
Alternatively, it can also be called given an argument, which is supposed to be a path to a trained model.

Example: Record a video sequence of a default model:

<code>python3 record_video_of_performing_trained_model.py</code>

Example: Record a video sequence of a specific model:

<code>python3 record_video_of_performing_trained_model.py Evaluation_CognitiveRobotics_Robo_Control/Results/PPO2/PandaController_2019_08_11__15_41_05__262730fzyxnprhgl/final_model.zip</code>

**Note**: By default, all video sequences are supposed to last 1000 time steps of the simulation.
To change this, adjust the value <code>VIDEO_LENGTH</code> in the said file.
However, due to technical issues, there is still a tendency for videos to encompass more time steps that the provided number.

#### Evaluation
The included git-submodule <code>Evaluation_CognitiveRobotics_Robo_Control</code> contains a set of trained models, the evaluations of both the training and the final training outcome,
and the tools used for the evaluation.

The tools contain a lot of inline-code and extensive class definitions explaining how the evaluation is done. Feel free to consult the attached project report for an overview.

All trained models are saved in separate folders. Their folders contain Training-check-points, files describing which parameters were used for training, and a documentation of the training process.

#### Further files:

##### callback.py
<code>callback.py</code> is used by the PPO agent to log training progress and to save checkpoints.

##### start.sh
<code>start.sh</code> is not particularly important to the project, but is the script for running the training process on the [University's Peregrine cluster](https://www.rug.nl/society-business/centre-for-information-technology/research/services/hpc/facilities/peregrine-hpc-cluster).
It has been attached and kept for convenience of the developers.

##### kukaGymEnv.py
<code>kukaGymEnv.py</code> served as inspiration for designing our own Gym environment. It is copied from the example environments shipped with the Pybullet installation and kept for comparison.

That's it. Have fun with the repository!
