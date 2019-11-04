import os
import gym
import tensorflow as tf
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from callback import callback
from stable_baselines.common.vec_env import DummyVecEnv
from datetime import datetime

now = datetime.now()

RENDER = True
FIXED_NUM_REPETITIONS = True
UPDATE_FREQUENCY = 100

# Specify save-directory
ALGO = "PPO2"
ENV_NAME = "PandaController"
TIME_STAMP = now.strftime("_%Y_%d_%m__%H_%M_%S__%f")
MODEL_ID = ENV_NAME + TIME_STAMP
PATH = "Results/" + ALGO + "/" + MODEL_ID + "/"
SUFFIX = "final_model"
SAVE_MODEL_DESTINATION = PATH + SUFFIX

# Create and vectorize Environment
env = PandaRobotEnv(renders=True, fixedActionRepetitions=True)
env = DummyVecEnv([lambda: env])   # The algorithms require a vectorized environment to run, hence vectorize

# Create custom NN architecture
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[150, 150])

# Create the PPO agent
model = PPO2("MlpPolicy", env,
             policy_kwargs=policy_kwargs,
             verbose=1,
             learning_rate=5e-4,
             tensorboard_log=str("/tmp/"+TIME_STAMP+"ppo2/"))

# Specify additional parameters for callback-method
model.check_point_location = PATH
model.update_frequency = UPDATE_FREQUENCY

# Retrieve the environment
env = model.get_env()

# Train the agent
model.learn(total_timesteps=200000, callback=callback)

# Save the agent
if not os.path.exists(PATH):
    os.makedirs(PATH)
model.save(SAVE_MODEL_DESTINATION)

