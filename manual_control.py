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
    #redraw(obs)

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
    return obs, reward, done, info

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
    default=1
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
    default=1
)
parser.add_argument(
    '--K',
    type=int,
    help="number of Mdps",
    default=5
)
parser.add_argument(
    '--epsilon',
    type=float,
    help="initial epislon",
    default=1.0
)
parser.add_argument(
    '--max_iteration',
    type=int,
    help="max number of iterations",
    default=100000
)
args = parser.parse_args()
env_set=[]
min_epsilon = 0.20
max_epsilon = 1.0    
decay_rate = 0.0005
gamma = 0.95  

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
houseRoomToType = np.ones((args.num_envs, args.num_rooms), dtype = np.int8) * -1
#table 2: Map from (nothing) to room type probability. experience we have collected after we wander inside houses
#initially we just assume that all rooms can be any kind of room. we will use Dirichlet distribution.
#we set alpha to be (1,1,1,....1). So any kinds of distribution will be uniformly possible 
RoomToTypeProb = np.ones((args.num_rooms, args.num_roomtypes)) 

#Map from room types to objects. (times it doesn't show up vs how many times it shows up )
roomtypeToObject = np.ones((args.num_roomtypes, args.num_objects, 2))

#test
RoomToTypeProb[0,0] = 10
RoomToTypeProb[1,1] = 10
RoomToTypeProb[2,2] = 10
roomtypeToObject[2,0,0] = 100
roomtypeToObject[2,1,0] = 100
roomtypeToObject[2,2,0] = 100
roomtypeToObject[1,0,1] = 100
roomtypeToObject[1,1,1] = 100
roomtypeToObject[1,2,1] = 100

def sampleRooms(ith_house):
    
    rooms = np.copy(houseRoomToType[ith_house])

    for ith_room in range(len(rooms)):
        #if we don't know what kind of room the ith_room is, we take a guess based on the past experience
        #if we know what room it is, no need to take a guess.
        if rooms[ith_room] == -1:
            #we sample a distribution based on the prior and then take a guess based on the distribution
            prob = np.random.dirichlet(alpha=RoomToTypeProb[ith_room])
            rooms[ith_room] = np.random.choice(args.num_roomtypes, p=prob)
    return rooms

def sampleObjects(rooms):
    #we create a table to record what objects each room has. 
    objects_in_rooms = np.zeros((args.num_rooms, args.num_objects))
    for ith_room in range(len(objects_in_rooms)):
        for ith_object in range(len(objects_in_rooms[ith_room])):
            #we sample a distribution based on the prior and then take a guess based on the distribution
            #roomtypeToObject[rooms[ith_room], ith_object] records how many times a item doesn't show up in a room vs show up in a room
            prob = np.random.dirichlet(alpha=roomtypeToObject[rooms[ith_room], ith_object])
            objects_in_rooms[ith_room, ith_object] = np.random.choice(2, p=prob)
        print('room ', ith_room, ' is a type ', rooms[ith_room], ' room and it has ', objects_in_rooms[ith_room])

    return objects_in_rooms

def selectAction(qtable, state, epsilon, i):
    e = random.uniform(0, 1)
    if e < epsilon:
        a = np.random.choice(3)
    else:
        a = np.argmax(qtable[state[0], state[1], state[2], :])
    return a


def sampleMDPs(ith_house, goal_type):

    qtables = []

    #once we get into a house, we create 10 mdps, 10 imagined environment based on past experience, 
    for ith_mdp in range(args.K):
        #find out what we know about the room layout for this house, -1 means we don't know for a particular room
        rooms=sampleRooms(ith_house)
        #after we knew/guessed what room type is for each room, we need to guess what kind of objects in each room
        objects_in_rooms = sampleObjects(rooms)
        #after we know what inside each room, we need to create our imagining world. 
        imaginingHouse = gym.make(args.env)
        #and then we set the environment into what we have sampled
        imaginingHouse.recreate(objects_in_rooms,rooms,goal_type)
        #create a qtable for this imagined environment
        qtable = np.zeros((imaginingHouse.grid.width, imaginingHouse.grid.height, 4, 3))
        #epsilon for the mdp
        epsilon = args.epsilon
        #checking our overall performance
        total_Reward = 0
        #the current state is the agent's position plus its direction.
        state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir] 
        #solve this mdp
        for i in range(args.max_iteration):
            #select action based on epsilon greedy strategy
            a = selectAction(qtable, state, epsilon, i)

            #take a step based on the action we select
            obs, reward, done, info = imaginingHouse.step(a)
 
            if done == True:
                #reset the agent's postion to its starting position, objects remain in the same locations
                obs = imaginingHouse.reset2()

                print("total reward: ", total_Reward, ' target is:', goal_type, 'epsilon:', epsilon, ' No of iterations:', i)

            new_state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir]
            #update the q table
            if not done:
                qtable[state[0],state[1],state[2],a] +=  (reward + gamma * np.max(qtable[new_state[0], new_state[1], new_state[2], :]) - qtable[state[0],state[1],state[2],a])
            else:
                qtable[state[0],state[1],state[2],a] += (reward - gamma * qtable[state[0],state[1],state[2],a])
            total_Reward += reward
            #if total reward is more than 500, our qtalbe of this mdp is good enough
            if total_Reward >400:
                break
            state = new_state
            #decrease the epsilon
            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*i) 

        #sanity check, after solved the mdp ,check if it works
        obs = imaginingHouse.reset2()

        state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir]

        for i in range(50):
            a = np.argmax(qtable[state[0], state[1], state[2], :])
            obs, reward, done, info = imaginingHouse.step(a)
            img = imaginingHouse.render('rgb_array', tile_size=args.tile_size)
            window.show_img(img)
            if done == True:
                obs = imaginingHouse.reset2()
                print("get to the target!!!!")
                break

            new_state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir]

            state = new_state
        qtables.append(np.expand_dims(qtable, 4))
    #shape of merged qtable is (width, height, orientaion, action, K)
    merged_qtable = np.concatenate(qtables, axis = 4)

    print('marges qtable shape: ', merged_qtable.shape)

    return merged_qtable




#each house we visit several times:
for ith_visit in range(args.num_visitsPerHouse):
    #loop through all the houses:
    for ith_house in range(args.num_envs):
        env = env_set[ith_house]
        if args.agent_view:
            env = RGBImgPartialObsWrapper(env)
            env = ImgObsWrapper(env)
        window = Window('gym_minigrid - ' + args.env +' house ' + str(ith_house) +' ' + str(ith_visit) + ' visits')
        #window.reg_key_handler(key_handler)
        #we reset the house environment, which doesn't change the room layout, some minor issues with object
        obs = env.reset2()

        goal_type = env.goal_type

        state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir] 
        
        #When an agent arrives at a house, it initializes table 3 to “I don’t know” everywhere.
        #table 3: Map from current house location to object at that location (can include “empty” but also “I don’t know yet”)
        #houseLocToObject[0][0] denotes location 0,0.   if value is 0 then it means we don't know yet
        houseLocToObject = np.zeros((env.grid.width, env.grid.height))

        merged_qtable = sampleMDPs(ith_house, goal_type)

        max_merged_qtable = np.max(merged_qtable, 4)

        print('max_merged_qtable shape:', max_merged_qtable.shape)



        for i in range(10):
            print('check under current state, value: ', max_merged_qtable[state[0], state[1], state[2], :])
            a = np.argmax(max_merged_qtable[state[0], state[1], state[2], :])
            obs, reward, done, info = step(a, ith_house)

            state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]

            #based on the observation, we check if we have collected new knowedge, if yes, we resample. 
            #use get_view_exts to get (topX, topY, botX, botY)
            print('topx, topy, botx, boty:', env.get_view_exts())




# each house we visit 10 times:
#   loop through all the houses:
#  
#       once we get into a house, we create 10 mdps, 10 imagined environment based on past experience, 
#       Then we solve those 10 MDPs, and combine them into 1 qtable, select 1 action based on the q table
#       while not found the target: 
#           we check if we know something new based on the observation, if yes, we recreate those 10 mdps, and solve it
#           we choose an action to go

