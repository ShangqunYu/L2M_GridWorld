import pygame
import gym
import sys

sys.path.append('/Users/fionaaga/Desktop/stable-baselines3')
sys.path.append('/Users/fionaaga/Desktop/L2M_GridWorld/jumping_task/gym-jumping-task')
print(sys.path)
from stable_baselines3 import DQN, TD3, A2C, PPO
import gym_jumping_task
from gym_jumping_task.envs import JumpTaskEnv
from stable_baselines3.common.env_checker import *
import numpy as np
from stable_baselines3.ddpg.policies import MlpPolicy



#env = gym.make('jumping-task-v0', rendering=True) #

#obs = env.reset()
#obs, r, done, info = env.step(0)
#print(type(r))
#print("obs:", obs.shape)
#print(env.observation_space.shape)
#print("length", len(env.observation_space.shape))

#check_env(env)





#training_steps = 2000000

#learning_rate = 0.0001
## Initialize the model0


#model = DQN("CnnPolicy", env, learning_rate = learning_rate, exploration_final_eps = 0.05, verbose=1) #
## Train the model
#model.learn(training_steps, eval_env=env, eval_freq= 10000, n_eval_episodes=10, eval_log_path = "log")

#model.save("./jumping_model")

#env.close()

env = gym.make('jumping-task-v0', rendering=True)

training_steps = 2000000

learning_rate = 0.0001


model = DQN("MlpPolicy", env, learning_rate = learning_rate, buffer_size = 100000, exploration_final_eps = 0.05, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("./jumping_model")

del model

model = DQN.load("./jumping_model") 

obs = env.reset()

print("obs:", obs.shape)
print(env.observation_space.shape)
print("length", len(env.observation_space.shape))

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset() 
