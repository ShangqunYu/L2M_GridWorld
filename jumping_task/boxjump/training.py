import gym
import sys

sys.path.append('/home/simon/Downloads/stable-baselines3')
sys.path.append('/home/simon/Downloads/jumping-task/gym-jumping-task')
print(sys.path)
from stable_baselines3 import DQN, TD3, A2C, PPO
from gym_jumping_task.envs import JumpTaskEnv
from stable_baselines3.common.env_checker import *
import numpy as np
from stable_baselines3.ddpg.policies import MlpPolicy



env = gym.make('jumping-task-v0', rendering=True) #

obs = env.reset()
obs, r, done, info = env.step(0)
print(type(r))
print("obs:", obs.shape)
print(env.observation_space.shape)
print("length", len(env.observation_space.shape))

check_env(env)





training_steps = 2000000

learning_rate = 0.0001
# Initialize the model0


model = DQN("CnnPolicy", env, learning_rate = learning_rate, exploration_final_eps = 0.05, verbose=1) #
# Train the model
model.learn(training_steps, eval_env=env, eval_freq= 10000, n_eval_episodes=10, eval_log_path = "log")

model.save("./jumping_model")

env.close()
