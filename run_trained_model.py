import sys
import gym
from stable_baselines import PPO2
from customRobotEnv import PandaRobotEnv
from stable_baselines.common.vec_env import DummyVecEnv



if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]  # Provide the path to your custom model
    else:
    
        path = "Evaluation_CognitiveRobotics_Robo_Control/Results/PPO2/PandaController_2019_08_11__15_41_05__262730fzyxnprhgl/final_model.zip"
    
    if len(sys.argv) > 2:
        iterations = sys.argv[2]
    else:
        iterations = 200000

    env = PandaRobotEnv(renders=True, fixedActionRepetitions=True)  # Todo: could be adjusted to be dependent on parameters used during training.
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run, hence vectorize
    
    try:
        model = PPO2.load(path)
    except ValueError:
        print('\nError: Make sure to have the pre-trained models available or to provide a valid path to a custom model as an argument.\n')
        sys.exit()

    # Enjoy trained agent
    obs = env.reset()
    for _ in range(iterations):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
