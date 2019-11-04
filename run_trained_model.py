import sys
import gym
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv



if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "Results/PPO2/PandaController_2019_04_11__21_37_27__747400/checkpoint_300.zip"
    if len(sys.argv) > 2:
        iterations = sys.argv[2]
    else:
        iterations = 200000

    env = PandaRobotEnv(renders=True, fixedActionRepetitions=True)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run, hence vectorize

    model = PPO2.load(path)

    # Enjoy trained agent
    obs = env.reset()
    for _ in range(iterations):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
