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

def reset2():
    obs = env.reset2()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)
    redraw(obs)

def step(action, index=0):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    obs['house'] = index
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
    "--render",
    help="whether we shall render the env or not",
    default=False
)
parser.add_argument(
    '--num_envs',
    type=int,
    help="number of envs",
    default=2
)
parser.add_argument(
    '--num_rooms',
    type=int,
    help="number of rooms",
    default=4
)
parser.add_argument(
    '--num_roomtypes',
    type=int,
    help="number of roomtypes",
    default=3
)
parser.add_argument(
    '--num_objects',
    type=int,
    help="number of objects",
    default=3
)
parser.add_argument(
    '--num_visitsPerHouse',
    type=int,
    help="number visits for each house",
    default=2
)
parser.add_argument(
    '--K',
    type=int,
    help="number of Mdps",
    default=5
)
args = parser.parse_args()
env_set=[]

for j in range(args.num_envs):
    env = gym.make(args.env)
    env_set.append(env)
random.shuffle(env_set)
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
    for i in range(200):
        a = np.random.choice(3)
        step(a, index)
        if i % 20 ==0:
            reset2()
    # Blocking event loop
    window.show(block=True)
'''
#table 1: Map from house id and room location to room type. (-1 menas “I don’t know yet”):
 #houseToRoomtype[0][0] denotes room types of room 0 in house 0. if value is -1 then it means we don't know yet
houseRoomToType = np.ones((args.num_envs, args.num_rooms)) * -1
#table 2: Map from (nothing) to room type probability. experience we have collected after we wander inside houses
#initially we just assume that all rooms can be any kind of room. we will use Dirichlet distribution.
#we set alpha to be (1,1,1,....1). So any kinds of distribution will be uniformly possible 
RoomToTypeProb = np.ones((args.num_rooms, args.num_roomtypes)) 
#table 3: Map from current house location to object at that location (can include “empty” but also “I don’t know yet”)
#houseLocToObject[0][0] denotes location 0,0.   if value is 0 then it means we don't know yet
houseLocToObject = np.zeros((env.grid.width, env.grid.height))
#Map from room type to object type probability.
roomtypeToObject = np.ones((args.num_roomtypes, args.num_objects ))

#test
RoomToTypeProb[0,0] = 10
RoomToTypeProb[1,1] =10
RoomToTypeProb[2,2] = 10

#each house we visit several times:
for ith_visit in range(args.num_visitsPerHouse):
    #loop through all the houses:
    for ith_house in range(args.num_envs):
        env = env_set[ith_house]
        if args.agent_view:
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)
        window = Window('gym_minigrid - ' + args.env +' house ' + str(ith_house))
        window.reg_key_handler(key_handler)
        #we reset the house environment, which doesn't change the room layout, some minor issues with object
        reset2()
        
        #When an agent arrives at a house, it initializes table 3 to “I don’t know” everywhere.
        houseLocToObject = np.zeros((env.grid.width, env.grid.height))
        
        #once we get into a house, we create 10 mdps, 10 imagined environment based on past experience, 
        for ith_mdp in range(args.K):
            #find out what we know about the room layout for this house, 0 means we don't know for a particular room
            rooms = np.copy(houseRoomToType[ith_house])

            for ith_room in range(len(rooms)):
                #if we don't know what kind of room the ith_room is, we take a guess based on the past experience
                if rooms[ith_room] == -1:
                    prob = np.random.dirichlet(alpha=RoomToTypeProb[ith_room])
                    rooms[ith_room] = np.random.choice(args.num_roomtypes, p=prob)
                    print("my guess for room ", ith_room, "is :", rooms[ith_room])




        for i in range(2):
            a = np.random.choice(3)
            step(a, ith_house)






# each house we visit 10 times:
#   loop through all the houses:
#  
#       once we get into a house, we create 10 mdps, 10 imagined environment based on past experience, 
#       Then we solve those 10 MDPs, and combine them into 1 qtable, select 1 action based on the q table
#       while not found the target: 
#           we check if we know something new based on the observation, if yes, we recreate those 10 mdps, and solve it
#           we choose an action to go

