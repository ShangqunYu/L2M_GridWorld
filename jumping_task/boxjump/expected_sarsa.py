# make some imports
import numpy as np
from tqdm import tqdm

#sys.path.append(".")

from jumping_task import JumpTaskEnv

# initialize the environment JumpTaskEnv defined in jumping_task.py
env = JumpTaskEnv(scr_w=60, scr_h=60)

# create an argmax function that breaks ties arbitrarily
def argmax(q_values):
  top = float("-inf")
  ties = []
  for i in range(len(q_values)):
    if q_values[i] > top:
      top = q_values[i]
      ties = []
    if q_values[i] == top:
      ties.append(i)
  return np.random.choice(ties)

# create an epsilon greedy function for off-policy learning
def select_action_egreedy(q_values, epsilon):
  if np.random.uniform() < epsilon:
    return np.random.choice(len(q_values))
  else:
    return argmax(q_values)

def action_probs(q_values, epsilon):
  next_state_probs = [epsilon/len(q_values)]*len(q_values)
  best_action = select_action_egreedy(q_values, epsilon)
  next_state_probs[best_action] += (1.0 - epsilon)

  return next_state_probs

# define the bellman update for q-learning
def bellman_update(alpha, reward, discount, last_state, state):
  pi = action_probs(Q[state], epsilon)
  target = reward + discount*sum(pi*Q[state])

  Q[last_state, action] += alpha*(target - Q[last_state, action])

#—————————————————————————————————————————————————————————————————————————————————————————————

# initialize some stuff
num_episodes = 10000
num_states = 60
num_actions = 2
Q = np.zeros((num_states, num_actions))
epsilon = 0.2
alpha = 0.1
discount = 0.1
SPEED = 0.05
LIFE = -10
EXIT = 10
RENDER = False
total_reward_hist = []
success_rate = []

# train the agent
# loop over number of episodes. change rendering to True above to visualize
for episode in tqdm(range(num_episodes)):
  #if episode > num_episodes-15:
    #RENDER = True 
  env = JumpTaskEnv(obstacle_position=12, rendering=RENDER, slow_motion=True, 
                    finish_jump=False, speed=SPEED, life=LIFE, exit=EXIT)
  env.render()
  state = env.agent_pos_x
  total_reward = 0

  # one episode
  while not env.done:
    last_state = state
    action = select_action_egreedy(Q[last_state], epsilon)
    # actions: 0: right, 1: jump
    # agent will receive EXIT reward for success, LIFE for crashing
    _, reward, term, _ = env.step(action)
    env.render()
    state = env.agent_pos_x
    # reward the agent for moving forward
    if state > last_state:
      reward += state
    # penalize the agent for jumping
    if action == 1:
      reward -= 1
    bellman_update(alpha, reward, discount, last_state, state)
    total_reward += reward
    #print(state, last_state, reward, total_reward)
  if state > 50:
    success_rate.append(1)
  else:
    success_rate.append(0)
  total_reward_hist.append(total_reward)

#print('Q values: ', Q)
print()
print('——————————————————')
print('Training complete.')
print('——————————————————')
#print('Reward history: ')
#print(total_reward_hist)
print()

np.save("./results/expected_sarsa_rewards.npy", total_reward_hist)

print('Success rate: ', 100*sum(success_rate) / num_episodes, "%")
print('Success rate in first 1000 episodes: ', 100*sum(success_rate[0:1000]) / 1000, "%")
print('Success rate in last 1000 episodes: ', 100*sum(success_rate[num_episodes-999:num_episodes+1]) / 100, "%")

#—————————————————————————————————————————————————————————————————————————————————————————————
'''
print()
print('Optimal policy will play.')
print()

# visualize one loop of our optimal policy
env = JumpTaskEnv(obstacle_position=12, rendering=True, slow_motion=True, 
                    finish_jump=False, speed=SPEED, life=LIFE, exit=EXIT)
env.render()
state = env.agent_pos_x
total_reward = 0

# one episode
while not env.done:
  last_state = state
  action = argmax(Q[last_state])
  # actions: 0: right, 1: jump
  # agent will receive 1000 reward for success, -100 for crashing
  _, reward, term, _ = env.step(action)
  env.render()
  state = env.agent_pos_x
  # reward the agent for moving forward
  if state > last_state:
    reward += state
  # penalize the agent for jumping
  if action == 1:
    reward -= 30
  bellman_update(alpha, reward, discount, last_state, state)
  total_reward += reward
  #print(state, last_state, reward, total_reward)
  #print(action, argmax(Q[last_state]), Q[last_state])
'''