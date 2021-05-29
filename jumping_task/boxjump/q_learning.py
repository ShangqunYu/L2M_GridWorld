# make some imports
import numpy as np
from tqdm import tqdm
import time

#sys.path.append(".")

from jumping_task import JumpTaskEnv
from QLAgent import QLAgent

# initialize the environment and the agent
agent_info = {
    "grid_width": 60,
    "grid_height": 16,
    "num_actions": 2,
    "epsilon": 0,
    "alpha": 1,
    "discount": 1
}

agent = QLAgent(agent_info)

#—————————————————————————————————————————————————————————————————————————————————————————————

# initialize some stuff
num_episodes = 60
SPEED = 0.03
LIFE = -10
EXIT = 10
RENDER = True
total_reward_hist = []
success_rate = []

# train the agent
# loop over number of episodes. change rendering to True above to visualize
for episode in tqdm(range(num_episodes)):
  #if episode > num_episodes-3:
    #RENDER = True
  env = JumpTaskEnv(obstacle_position=12, rendering=RENDER, slow_motion=True, finish_jump=True, speed=SPEED, life=LIFE, exit=EXIT)
  env.render()
  start_location = (int(env.agent_pos_x), int(env.agent_pos_y)-10)
  agent.agent_start(start_location)

  total_reward = 0

  # one episode
  while not env.done:
    # actions: 0 is right, 1 is up
    agent.last_state = agent.state
    agent.action = agent.select_action_egreedy()

    # agent will receive EXIT reward for success, LIFE for crashing
    _, reward, term, _ = env.step(agent.action)
    env.render()
    new_location = (int(env.agent_pos_x), int(env.agent_pos_y)-10)

    # update the agent with new information, penalize for taking a long time
    agent.agent_step(new_location, reward)
    agent.bellman_update()
    total_reward += reward - 1

  if env.agent_pos_x > 55:
    success_rate.append(1)
  else:
    success_rate.append(0)
  total_reward_hist.append(total_reward)

#—————————————————————————————————————————————————————————————————————————————————————————————

print()
print('——————————————————')
print('Training complete.')
print('——————————————————')
print()

np.save("./results/q_learning_rewards.npy", total_reward_hist)

print('Success rate: ', 100*sum(success_rate) / num_episodes, "%")
print('Success rate in first 10 episodes: ', 100*sum(success_rate[0:10]) / 10, "%")
print('Success rate in last 10 episodes: ', 100*sum(success_rate[num_episodes-9:num_episodes+1]) / 10, "%")

#—————————————————————————————————————————————————————————————————————————————————————————————

print()
print("Optimal Policy will now play.")
print()
time.sleep(1)

env = JumpTaskEnv(obstacle_position=12, rendering=RENDER, slow_motion=True, finish_jump=True, speed=SPEED, life=LIFE, exit=EXIT)
env.render()
start_location = (int(env.agent_pos_x), int(env.agent_pos_y)-10)
agent.agent_start(start_location)

while not env.done:
  agent.last_state = agent.state
  agent.action = agent.argmax(agent.Q[agent.state])
  _, reward, term, _ = env.step(agent.action)
  env.render()
  new_location = (int(env.agent_pos_x), int(env.agent_pos_y)-10)
  agent.agent_step(new_location, reward)