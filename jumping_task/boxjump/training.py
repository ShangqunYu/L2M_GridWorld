# import the class JumpTaskEnv defined in jumping_task.py
import numpy as np
import pygame
import time
import sys

sys.path.append(".")

from jumping_task import JumpTaskEnv

# initialize the environment
env = JumpTaskEnv(scr_w=60, scr_h=60)

def test():
  env = JumpTaskEnv(scr_w=60, scr_h=60, obstacle_position=0, rendering=True, slow_motion=True, finish_jump=False)
  env.render()
  score = 0
  while not env.done:
    action = None
    if env.jumping[0] and env.finish_jump:
      action = 3
    else:
      events = pygame.event.get()
      for event in events:
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_RIGHT:
            action = 0
          elif event.key == pygame.K_UP:
            action = 1
          elif event.key == pygame.K_e:
            env.exit()
          else:
            action = 'unknown'
    if action is None:
      continue
    elif action == 'unknown':
      print('We did not recognize that action. Please use the arrows to move the agent or the \'e\' key to exit.')
      continue
    _, reward, term, _ = env.step(action)
    env.render()
    score += reward
    print('Agent position: {:2d} | Reward: {:2d} | Terminal: {}'.format(env.agent_pos_x, reward, term))
  print('---------------')
  print('Final score: {:2d}'.format(int(score)))
  print('---------------')

test()