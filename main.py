import os, time
import json, csv
import gym
import tensorflow as tf
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from callback import callback
from stable_baselines.common.vec_env import DummyVecEnv
from datetime import datetime
import argparse

import random
import string


def random_string(length=10):
    """
        Generate a random string of given length
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


now = datetime.now()

# Specify save-directories
ALGO = "PPO2"
ENV_NAME = "PandaController"
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = ENV_NAME + TIME_STAMP + random_string()
PATH = "Results/" + ALGO + "/" + MODEL_ID + "/"
SUFFIX = "final_model"
SAVE_MODEL_DESTINATION = PATH + SUFFIX          # For saving checkpoints and final model
TENSORBOARD_LOCATION = PATH + "tensorboard/"    # For tensorboard usage


# DEFAULTS:

RENDER = False
FIXED_NUM_REPETITIONS = True
CHECKPOINT_FREQUENCY = 10

params = dict(
    # Whether to render simulation during training, mainly true for debugging
    render=RENDER,
    # Specify custom NN architecture
    policy="MlpPolicy",
    act_fun='tf.nn.tanh',
    net_arch=[150, 150],
    # How much information shall be printed to terminal
    verbose=1,
    # Network's learning rate
    learning_rate=5e-4,
    # Number of training time steps
    total_timesteps=1000000,
    # Name of algo used
    algo=ALGO,
    # Identifier for simulation model used, containing name of env and time stamp
    model_id=MODEL_ID,
    # Whether number of repetitions per commanded action is going to be fixed or supposed to be learned as well
    fixed_action_repetitions=FIXED_NUM_REPETITIONS,
    # Every hor many time weight updated a checkpoint shall be saved
    checkpoint_frequency=CHECKPOINT_FREQUENCY,
    # Emission data path
    path=PATH,
    # Where data for tensorboard is saved to
    tensorboard_log=TENSORBOARD_LOCATION,
    # Which distance function specifictions are supposed to be used
    dist_specification=[0, 'A'],
    # Every how many weight updates training progress shall be logged
    log_train_progress_frequency=10,
    # Headings for training progress log file
    log_train_progress_data=['Update_nr',                # Current count of weight updates performed so far
                             'Grasps',                   # Nr of grasps over last X weight updates
                             'Avg_grasp_time_steps',     # Average of time steps needed in a simulation to get from init
                                                         # pose to attaining goal, averaged over the time steps recorded
                                                         # for all successful grasps over last X weight updates
                             'Std_grasp_time_steps',
                             'Max_grasp_time_steps',
                             'Min_grasp_time_steps',
                             'Total_time_steps'                # Total time steps simulated so far
                             ],
    # Measure of how close end-effector's center of mass (COM) must be to goal location in Euclidean (=straight-line)
    # distance in order for arm to have reached goal
    maxDist=0.25,
    # Measure of how close end-effector's fingers' orientation towards goal must be to vector pointing straight from
    # end-effector's COM towards goal location in Euclidean (=straight-line) distance in order for arm to have reached
    # goal
    maxDeviation=0.25,
)


def create_dir(direct=params['path']):
    """
        Ensure that a given path exists.
        :param direct: Directory to be created when necessary.
        :return: -
    """
    if not os.path.exists(direct):
        os.makedirs(direct)


def setup_train_log_file(direct=params['path']):
    create_dir(direct)
    with open(direct+"training_eval.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel', quoting=csv.QUOTE_ALL)
        wr.writerow(params['log_train_progress_data'])


def save_param_settings(direct=params['path']):
    """
        For convenience. Saves the parameter settings used to train a model to the directory to which the contents of
        the training run get saved. Facilitates later understanding of under which conditions a model was trained.
        :return: -
    """
    create_dir(direct)

    # Save as json (easier to read back in)
    js = json.dumps(params)
    f = open(direct+"params.json", "w")
    f.write(js)
    f.close()

    # Save as csv (nicer to read)
    with open(direct+"params.csv", "w") as f:
        w = csv.writer(f, dialect='excel', quoting=csv.QUOTE_ALL)
        for key, val in params.items():
            w.writerow([key, val])
    f.close()



def get_args():
    """
        Function for reading in command line arguments specified by flags.
        Call e.g.

            python3 main.py -r Results/.../checkpoint_7770.zip -p Params/param_setting_1.json

        for  retraining model saved at 'Results/.../checkpoint_7770.zip' using parameters
        specified in 'Params/param_setting_1.json'.

        :return: Dictionaly-like object containing a field per added argument.
                 If no value was provided for an argument, arg's value will be None
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--retrain", "--restore", help="Path to zip archive for continuing training", type=str)
    parser.add_argument("-p", "--params", "--parameters", help="Training parameter specifications", type=str)

    return parser.parse_args()


def read_in_input_params(file_name):
    """
        Reads in parameter settings specified in file found under path_file_name. Then updates params specified above,
        accordingly.

        :param file_name: Path to file containing parameter specifications
        :return: -
    """

    with open(file_name) as json_file:
        data = json.load(json_file)
    params.update(data)


if __name__ == '__main__':

    # Read in arguments provided via command line and specified via flags
    args = get_args()

    # Read in provided input parameters from file & update params
    if args.params is not None:
        params['provided_params_file'] = args.params
        read_in_input_params(params['provided_params_file'])

    create_dir(params['tensorboard_log'])
    setup_train_log_file()

    # Create and vectorize Environment
    env = PandaRobotEnv(renders=params['render'],
                        fixedActionRepetitions=params['fixed_action_repetitions'],
                        distSpecifications=params['dist_specification'],
                        maxDist=params['maxDist'],
                        maxDeviation=params['maxDeviation'])

    env = DummyVecEnv([lambda: env])   # The algorithms require a vectorized environment to run, hence vectorize

    # Check wor whether to continue training of a previously created & trained model
    if args.retrain is not None:
        # Reload model (PPO agent) for continuing training
        params['restored'] = args.retrain
        model = PPO2.load(params['restored'])
        model.env = env
    else:
        # Create new PPO agent
        model = PPO2(policy=params['policy'],
                     env=env,
                     policy_kwargs=dict(act_fun=eval(params['act_fun']),
                                        net_arch=params['net_arch']),
                     verbose=params['verbose'],
                     learning_rate=params['learning_rate'],
                     tensorboard_log=params['tensorboard_log'])

    # Save parameters to file
    save_param_settings()

    # Specify additional parameters for callback-method
    model.path = params['path']
    model.checkpoint_frequency = params['checkpoint_frequency']
    model.log_train_progress_frequency = params['log_train_progress_frequency']

    # Retrieve the environment
    env = model.get_env()

    # Train the agent
    model.learn(total_timesteps=params['total_timesteps'], callback=callback)

    # Save the agent
    create_dir()
    model.save(SAVE_MODEL_DESTINATION)

