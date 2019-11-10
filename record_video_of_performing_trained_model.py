import sys, os
import gym
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder

from datetime import datetime
import time

import random
import string

'''
    Program set up to record a visually rendered pybullet simulation, mainly to record the progress of models trained 
    on controlling robotic arms. Set up in a way that maximally one model is recorded at a time. Could be extended.
    
    Required package can be installed on Ubuntu via: sudo apt-get install ffmpeg 
'''


def random_string(length=5):
    """
        Generate a random string of given length
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


# Specify parameters
DEFAULT_MODEL = "PandaController_2019_08_11__15_41_05__262730fzyxnprhgl"
VIDEO_LENGTH = 300
NUMBER_OF_RECORDINGS = 1

# Specify save-directories
now = datetime.now()
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
RECORDING_ID = "Recording_" + TIME_STAMP + random_string()
PATH_BASE = "VideoRecordings/"


def create_dir(direct):
    """
        Ensure that a given path exists.
        :param direct: Directory to be created when necessary.
        :return: -
    """
    if not os.path.exists(direct):
        os.makedirs(direct)


def determine_save_path_components(model_path):
    path_parts = model_path.split("/")
    return path_parts[-2], path_parts[-1].replace('.zip', ''), path_parts[-3]


def recording_destination_and_name(model_path=None):
    model_path = model_path if model_path is not None else DEFAULT_MODEL
    model, train_state, env_name = determine_save_path_components(model_path)
    video_path = PATH_BASE + "/" + model + "/" + train_state + "/"

    video_id = RECORDING_ID

    return video_id, video_path, env_name


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]  # Provide the path to your custom model
    else:
        path = "Evaluation_CognitiveRobotics_Robo_Control/Results/PPO2/"+DEFAULT_MODEL+"/final_model.zip"


    env = PandaRobotEnv(renders=True, fixedActionRepetitions=True, evalFlag=True)  # Todo: could be adjusted to be dependent on parameters used during training.
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run, hence vectorize
    
    try:
        model = PPO2.load(path)
    except ValueError:
        print('\nError: Make sure to have the pre-trained models available or to provide a valid path to a custom model as an argument.\n')
        print('Provided path: ' + path + '\n')
        sys.exit()
    
    video_id, video_folder, env_id = recording_destination_and_name(path)
    create_dir(video_folder)
    video_length = VIDEO_LENGTH
    num_videos = NUMBER_OF_RECORDINGS

    # Record the video starting at the first step
    env = VecVideoRecorder(venv=env,
                           video_folder=video_folder,
                           record_video_trigger=lambda x: x == 0,
                           video_length=video_length,
                           name_prefix=video_id)

    
    # Enjoy trained agent
    obs = env.reset()
    time_step_counter = 0
    while time_step_counter <= (num_videos*(video_length+1)):
        env.envs[0].set_step_counter(time_step_counter)
        action, _ = model.predict(obs)
        obs, _, _, info = env.step(action)  # Assumption: eval conducted on single env only!

        reward, time_step_counter, done = info[0][:]

        # print(info)
        #time.sleep(0.1)

        if done:
            time.sleep(1)
            obs = env.reset()
