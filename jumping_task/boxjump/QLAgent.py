import numpy as np

class QLAgent:
  def __init__(self, init_info):
    self.grid_width = init_info["grid_width"]
    self.grid_height = init_info["grid_height"]
    self.num_actions = init_info["num_actions"]
    self.epsilon = init_info["epsilon"]
    self.alpha = init_info["alpha"]
    self.discount = init_info["discount"]

    self.Q = {index: [10]*self.num_actions for index in range(self.grid_width)}

  def agent_start(self, start_location):
    # procedure for determining state from location
    self.state = start_location[0]
  
    # create an epsilon greedy function for off-policy learning
  def select_action_egreedy(self):
    if np.random.uniform() < self.epsilon:
      return np.random.choice(range(self.num_actions))
    else:
      q_values = self.Q[self.state]
      return self.argmax(q_values)
      
  # define a step function for mapping location to state and updating reward
  def agent_step(self, new_location, reward):
    self.last_state = self.state
    self.state = new_location[0]
    self.reward = reward
  
  # create an argmax function that breaks ties arbitrarily
  def argmax(self, q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
      if q_values[i] > top:
        top = q_values[i]
        ties = []
      if q_values[i] == top:
        ties.append(i)
    return np.random.choice(ties)
  
  # define the bellman update for q-learning
  def bellman_update(self):
    # get a list of the next action values given the next state
    q_values = self.Q[self.state]
    # define the target as reward + discount*q_value for best next action
    target = self.reward + self.discount*np.max(q_values)
    self.Q[self.last_state][self.action] += self.alpha*(target - self.Q[self.last_state][self.action])