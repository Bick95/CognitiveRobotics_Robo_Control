import os, sys
import json, csv
import gym
import tensorflow as tf
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from callback import callback
from stable_baselines.common.vec_env import DummyVecEnv
from datetime import datetime

now = datetime.now()

# Specify save-directories
ALGO = "PPO2"
ENV_NAME = "PandaController"
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = ENV_NAME + TIME_STAMP
PATH = "Results/" + ALGO + "/" + MODEL_ID + "/"
SUFFIX = "final_model"
SAVE_MODEL_DESTINATION = PATH + SUFFIX          # For saving checkpoints and final model
TENSORBOARD_LOCATION = PATH + "tensorboard/"    # For tensorboard usage


RENDER = True
FIXED_NUM_REPETITIONS = True
CHECKPOINT_FREQUENCY = 100

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
    total_timesteps=200000,
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
)


def create_dir(direct=PATH):
    if not os.path.exists(direct):
        os.makedirs(direct)


def save_param_settings():
    """
    For convenience. Saves the parameter settings used to train a model to the directory to which the contents of
    the training run get saved. Facilitates later understanding of under which conditions a model was trained.
    :return: -
    """
    create_dir()

    # Save as json (easier to read back in)
    js = json.dumps(params)
    f = open(PATH+"params.json", "w")
    f.write(js)
    f.close()

    # Save as csv (nicer to read)
    csv.register_dialect('customStyle',
                         delimiter=',',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)
    with open(PATH+"params.csv", "w") as f:
        w = csv.writer(f, dialect='customStyle')
        for key, val in params.items():
            w.writerow([key, val])
    f.close()



if __name__ == '__main__':

    create_dir(params['tensorboard_log'])

    # Create and vectorize Environment
    env = PandaRobotEnv(renders=params['render'],
                        fixedActionRepetitions=params['fixed_action_repetitions'])
    env = DummyVecEnv([lambda: env])   # The algorithms require a vectorized environment to run, hence vectorize

    if len(sys.argv) > 1:
        # Reload model for continuing training
        path = sys.argv[1]
        model = PPO2.load(path)
        model.env = env
        params['restored'] = path
    else:
        # Create the PPO agent
        model = PPO2(params['policy'], env,
                     policy_kwargs=dict(act_fun=eval(params['act_fun']),
                                        net_arch=params['net_arch']),
                     verbose=params['verbose'],
                     learning_rate=params['learning_rate'],
                     tensorboard_log=params['tensorboard_log'])

    # Save parameters to file
    save_param_settings()

    # Specify additional parameters for callback-method
    model.check_point_location = params['path']
    model.checkpoint_frequency = params['checkpoint_frequency']

    # Retrieve the environment
    env = model.get_env()

    # Train the agent
    model.learn(total_timesteps=params['total_timesteps'], callback=callback)

    # Save the agent
    create_dir()
    model.save(SAVE_MODEL_DESTINATION)

