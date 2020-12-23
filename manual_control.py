#!/usr/bin/env python3
import random
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def step(action, index=0):
    obs, reward, done, info = env.step(action)
    print("current location:", env.agent_pos)
    print("current orientation", env.agent_dir)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    obs = np.append(obs, index)
    print('obs:', obs)
    if done:
        print('done!')
        reset()
    else:
        redraw(obs)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == 'pageup':
        step(env.actions.pickup)
        return
    if event.key == 'pagedown':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-MultiRoom-N6-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--num_envs',
    type=int,
    help="number of envs",
    default=5
)
parser.add_argument(
    '--num_episodes',
    type=int,
    help="number of episodes in value iteration",
    default=1000
)
parser.add_argument(
    '--num_steps',
    type=int,
    help="number of steps in each episodes in value iteration",
    default=1000
)

parser.add_argument(
    '--num_roomtypes',
    type=int,
    help="number of room types",
    default=5
)
parser.add_argument(
    '--num_rooms',
    type=int,
    help="number of rooms in each house",
    default=4
)
parser.add_argument(
    '--num_houses',
    type=int,
    help="number of houses",
    default=4
)


args = parser.parse_args()
env_set=[]

for j in range(args.num_envs):
    env = gym.make(args.env)
    env_set.append(env)
random.shuffle(env_set)

#table 1: Map from house id and room location to room type. (0 menas “I don’t know yet”):
 #houseToRoomtype[0][0] denotes room types of room 0 in house 0. if value is 0 then it means we don't know yet
houseRoomToType = np.zeros((args.num_houses, args.num_rooms))   
#table 2: Map from (nothing) to room type probability. experience we have collected after we wander inside houses
#initially we just assume that all rooms can be any kind of room. we will use Dirichlet distribution.
#we set alpha to be (1,1,1,....1). So any kinds of distribution will be uniformly possible 
RoomToTypeProb = np.ones((args.num_rooms, args.num_roomtypes)) 
#table 3: Map from current house location to object at that location (can include “empty” but also “I don’t know yet”)
#houseLocToObject[0][0] denotes location 0,0.	if value is 0 then it means we don't know yet
houseLocToObject = np.ones((env.grid.width, env.grid.height))
print("size of the env", env.grid.width, " X ", env.grid.height)




'''When an agent is born, it sets table 1 to “I don’t know” everywhere 
and it initializes the counts in table 2 and table 4 to be some small number (epsilon = 0.5?)
'''



for j in range(args.num_envs):
    env = env_set[j]
    index = j
    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window('gym_minigrid - ' + args.env)
    window.reg_key_handler(key_handler)

    reset()
    for i in range(30):
        a = np.random.choice(3)
        step(a, index)
    # Blocking event loop
    window.show(block=True)
