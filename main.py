import gym
import tensorflow as tf
from stable_baselines import PPO2

# Environment related
#from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from customRobotEnv import CustomRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv

# Create and vectorize Environment
env_name = "CustomRobotEnv"
env = eval(env_name + '(renders=True)')  # <-- Allows for making the Env-name variable: returns: KukaGymEnv(renders=True) (as used in line below)
#env = KukaGymEnv(renders=True)          # was initially: env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])         # The algorithms require a vectorized environment to run, hence vectorize

# Create custom NN architecture
policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[40, 40])

# Create the PPO agent
model = PPO2("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=2.5e-4)  # env was before string: "CartPole-v1"

# Retrieve the environment
env = model.get_env()

# Train the agent
model.learn(total_timesteps=1000000)

# Save the agent
model.save("ppo2-" + env_name)

# Show 1000 test iterations to user after training is done
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# Demonstrate deletion of model
del model

# Demonstrating loading of model
model = PPO2.load("ppo2-" + env_name)
