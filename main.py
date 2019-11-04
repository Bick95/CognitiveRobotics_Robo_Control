import os
import gym
import tensorflow as tf
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv
from datetime import datetime

now = datetime.now()

RENDER = True
FIXED_NUM_REPETITIONS = True

PATH = "Results/"
ENV_NAME = "ppo2-" + "PandaRobotEnv"
TIME_STAMP = now.strftime("%m_%d_%Y__%H_%M_%S__%f")
SAVE_MODEL_DESTINATION = PATH + ENV_NAME + TIME_STAMP

# Create and vectorize Environment
env = PandaRobotEnv(renders=True, fixedActionRepetitions=True)
env = DummyVecEnv([lambda: env])   # The algorithms require a vectorized environment to run, hence vectorize

# Create custom NN architecture
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[150, 150])

# Create the PPO agent
model = PPO2("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=5e-4)

# Retrieve the environment
env = model.get_env()

# Train the agent
model.learn(total_timesteps=200000)

# Save the agent
if not os.path.exists(PATH):
    os.makedirs(PATH)
model.save(SAVE_MODEL_DESTINATION)

