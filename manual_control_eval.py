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
    #if not args.agent_view:
    #    img = env.render('rgb_array', tile_size=args.tile_size)

    #window.show_img(img)
    pass

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        #window.set_caption(env.mission)

    redraw(obs)

def reset2():
    obs = env.reset2()
    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        #window.set_caption(env.mission)
    redraw(obs)

def step(action, index=0, eval=False):
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))
    obs['house'] = index
    print('obs:', obs)
    '''
    sanity check
    '''
    if eval:
        foundNewKnowledge = updateKnowledge_eval(env, obs, index)
    else:
        foundNewKnowledge = updateKnowledge(env, obs, index)

    redraw(obs)


    if done:
        print('done!')
        #reset2()

    return obs, reward, done, info, foundNewKnowledge

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        #window.close()
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
    default=3
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
    default=3
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



#test


#currently, items can only be at the following locations. 
#location (x, y) means at x row, y column.
#                 room0   room1     room2   room3
ball_locations = [(5, 5), (11, 5), (5, 11), (11, 11)]
box_locations =  [(2, 1), (8, 1),  (2, 7),  (8, 7)]
box2_locations = [(3, 3), (9, 3),  (3, 9),  (9, 9)]
# all locations that may contain target object
object_locations = [ball_locations, box_locations, box2_locations]


def sampleRooms(ith_house):
    #first of we check what we know, -1 means we don't know for a particular room
    rooms = np.copy(houseRoomToType[ith_house])

    for ith_room in range(len(rooms)):
        #if we don't know what kind of room the ith_room is, we take a guess based on the past experience
        #if we know what room it is, no need to take a guess.
        if rooms[ith_room] == -1:
            #we sample a distribution based on the prior and then take a guess based on the distribution
            prob = np.random.dirichlet(alpha=RoomToTypeProb[ith_room])
            rooms[ith_room] = np.random.choice(args.num_roomtypes, p=prob)
    return rooms

def sampleObjects(ith_house, rooms):
    #we create a table to record what objects each room has. 
    objects_in_rooms = np.zeros((args.num_rooms, args.num_objects))
    for ith_room in range(len(objects_in_rooms)):
        for ith_object in range(len(objects_in_rooms[ith_room])):
            location = object_locations[ith_object][ith_room]
            #value 0 means we don't have any previous info about that location, 
            print("sampling, key location ", location[0], location[1],' has ', houseLocToObject[ith_house, location[0], location[1]])
            if houseLocToObject[ith_house, location[0], location[1]]== 0:
                #so we sample a distribution based on the prior and then take a guess based on the distribution
                #roomtypeToObject[rooms[ith_room], ith_object] records how many times an item doesn't show up in a room vs show up in a room
                prob = np.random.dirichlet(alpha=roomtypeToObject[rooms[ith_room], ith_object])
                objects_in_rooms[ith_room, ith_object] = np.random.choice(2, p=prob)
            #if we know there is nothing in the location, then we don't sample, we know there is nothing there
            elif houseLocToObject[ith_house, location[0], location[1]] == 1:
                print("object is not there, location: ", location[0], ' ',location[1])
                objects_in_rooms[ith_room, ith_object] = 0
            #otherwise, we know for sure the item is there. 
            else:
                print("object is there, location: ", location[0], ' ',location[1])
                objects_in_rooms[ith_room, ith_object] = 1
        print('room ', ith_room, ' is a type ', rooms[ith_room], ' room and it has ', objects_in_rooms[ith_room])

    return objects_in_rooms

def selectAction(qtable, state, epsilon, i):
    e = random.uniform(0, 1)
    if e < epsilon:
        a = np.random.choice(3)
    else:
        a = np.argmax(qtable[state[0], state[1], state[2], :])
    return a

def value_iteration(num_iterations, env):
    V = np.zeros((env.grid.width, env.grid.height, 4))
    Q = np.zeros((env.grid.width, env.grid.height, 4, 3))
    for i in range(num_iterations):
        V_old = V.copy()
        Q_old = Q.copy()
        bellman_update(V, Q, env)
        oldstate = np.concatenate([V_old.reshape((-1)), Q_old.reshape((-1))])
        newstate = np.concatenate([V.reshape((-1)), Q.reshape((-1))])
        diff = newstate - oldstate
        diffnorm = np.linalg.norm(diff, ord=2)
        #print(i, ' iterations!!!!!!!!!!!!!!!!!!!!!!!', ' diffnorm: ', diffnorm)
        #if the update is very small, we assume it converged. 
        #also if there is no target object in the house, it will end in the first iteration as well. 
        if diffnorm < 0.01:
            break
    return V, Q

def bellman_update(V, Q, env):
    for i in range(env.grid.width):
        for j in range(env.grid.height):
            #we only update states whose location is not occupied by walls or objects. 
            if env.grid.get(i,j) == None:
                #four different orientations
                for k in range(4):
                    
                    s = (i,j,k)
                    front_position = front_pos(env, i, j, k)
                    #if it is the goal state, we don't give it a value

                    if env.grid.get(front_position[0], front_position[1])!= None and env.grid.get(front_position[0], front_position[1]).type == env.goal_type:
                        #print(env.goal_type, ' is in front of me!! at ', front_position[0], ' ', front_position[1], 'my position is ', i,' ', j,' ', k)
                        continue
                    else:
                        for ai in range(3):
                            s_prime = transition(env, s, ai)
                            #print('s: ',s, 'action: ', ai, 's_prime:', s_prime)
                            Q[i, j, k, ai] = rewardFunc(env, s, ai, s_prime) + gamma * V[s_prime]
                        #print(Q[s])
                        V[s] = Q[s].max()
def rewardFunc(env, s, ai, s_prime):
    front_position = front_pos(env, s_prime[0], s_prime[1], s_prime[2])
    if env.grid.get(front_position[0], front_position[1]) == None:
        return 0
    else:
        if env.grid.get(front_position[0], front_position[1]).type == env.goal_type:
            #print('you got reward here!! at : ', s_prime)
            return 1
        else:
            return 0

def front_pos(env, i, j, k):
    DIR_TO_VEC = [
        # Pointing right (positive X)
        np.array((1, 0)),
        # Down (positive Y)
        np.array((0, 1)),
        # Pointing left (negative X)
        np.array((-1, 0)),
        # Up (negative Y)
        np.array((0, -1)),]
    front_pos = (i, j) + DIR_TO_VEC[k]
    return front_pos

#for value iteration.
def transition(env, s, a):
    #if action is turning left
    if a == 1:
        s_leftTurn = (s[0],s[1], s[2]-1 if s[2]-1 >=0 else 3)
        return s_leftTurn
    #if action is turning right
    elif a == 2:
        s_rightTurn = (s[0], s[1], s[2]+ 1 if s[2]+1 <=3 else 0)
        return s_rightTurn
    #if action is moving forward

    elif a == 0:
        front_position = front_pos(env, s[0], s[1], s[2])
        #if the forward postion is not empty, agent can't go there
        if env.grid.get(front_position[0], front_position[1]) == None:
            s_forward = (front_position[0], front_position[1], s[2])
            return s_forward
        else:
            s_forward = s
            return s_forward
    else:
        assert False, "invalid action"

                



def sampleMDPs(ith_house, goal_type, starting_pos, starting_dir):

    qtables = []

    #here we create k mdps, k imagined environment based on past experience, 
    for ith_mdp in range(args.K):
        #find out what we know about the room layout for this house, 
        rooms=sampleRooms(ith_house)
        #after we knew/guessed what room type is for each room, we need to guess what kind of objects in each room
        objects_in_rooms = sampleObjects(ith_house, rooms)
        #after we know what inside each room, we need to create our imagining world. 
        imaginingHouse = gym.make(args.env)
        #and then we set the environment into what we have sampled
        imaginingHouse.recreate(objects_in_rooms,rooms,goal_type)

        #testing
        V, Q = value_iteration(60, imaginingHouse)
        '''
        #sanity check, after solved the mdp ,check if it works
        obs = imaginingHouse.reset2()

        state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir]

        
        for i in range(50):
            a = np.argmax(Q[state[0], state[1], state[2], :])
            print('current location: ', state, ' q values: ',Q[state[0], state[1], state[2], :],' action taking: ', a)
            obs, reward, done, info = imaginingHouse.step(a)
            img = imaginingHouse.render('rgb_array', tile_size=args.tile_size)
            window.show_img(img)
            if done == True:
                obs = imaginingHouse.reset2()
                print("get to the target!!!!")
                break

            new_state = [imaginingHouse.agent_pos[0], imaginingHouse.agent_pos[1], imaginingHouse.agent_dir]

            state = new_state
        '''
        #after we solve each MDP, we append it into a group
        qtables.append(np.expand_dims(Q, 4))
    
    #Then we merge all the q tables, the shape(width, height, num_directions, num_actions, K)
    merged_qtable = np.concatenate(qtables, axis = 4)

    print('marges qtable shape: ', merged_qtable.shape)

    #shape of merged qtable is (width, height, orientaion, action, K)
    return merged_qtable


def updateKnowledge_eval(env, obs, ith_house):
    # an indicator shows whehter we have found new knowledge
    foundNewKnowledge = False
    # we update the table 3 based on what we have seen in the house
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = env.agent_pos + f_vec * (env.agent_view_size - 1) - r_vec * (env.agent_view_size // 2)
    observed_info = np.zeros((env.grid.width, env.grid.height), dtype=np.int8)
    for vis_j in range(0, env.agent_view_size):
        for vis_i in range(0, env.agent_view_size):
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            # the view can't go beyond the space of the environment.
            if abs_i < 0 or abs_i >= env.width:
                continue
            if abs_j < 0 or abs_j >= env.height:
                continue
            # NOTE!!! indices in numpy table and the actual location coordinate indices are different
            observed_info[abs_j, abs_i] = obs['image'][vis_i, vis_j]
            # if the observed info is zero, it means we can't see it through the wall, so don't update,
            # and if we already know the information, then we also don't have to update.
            if observed_info[abs_j, abs_i] != 0 and observed_info[abs_j, abs_i] != houseLocToObject[
                ith_house, abs_j, abs_i]:
                houseLocToObject[ith_house, abs_j, abs_i] = observed_info[abs_j, abs_i]

                # if the location is where the itmes can show up, we need to update the tables based on whether
                # we have seen objects or not. it also means we have acquired new knowledge.


    # then we also need to update table 1 the room layout.
    if houseRoomToType[ith_house, obs['room']] != obs['roomtype']:
        houseRoomToType[ith_house, obs['room']] = obs['roomtype']
        foundNewKnowledge = True
        # also update our prior,
    # print(houseRoomToType[ith_house,:])

    # print(observed_info)
    # print(houseLocToObject[ith_house,:,:])
    # print("whether have new knowledge:", foundNewKnowledge)
    # return foundNewKnowledge
        
    
    



def updateKnowledge(env, obs, ith_house):
    #an indicator shows whehter we have found new knowledge
    foundNewKnowledge = False
    # we update the table 3 based on what we have seen in the house
    f_vec = env.dir_vec
    r_vec = env.right_vec
    top_left = env.agent_pos + f_vec * (env.agent_view_size-1) - r_vec * (env.agent_view_size // 2)
    observed_info = np.zeros((env.grid.width, env.grid.height), dtype = np.int8)
    for vis_j in range(0, env.agent_view_size):
        for vis_i in range(0, env.agent_view_size):
            abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)
            #the view can't go beyond the space of the environment.
            if abs_i < 0 or abs_i >= env.width:
                continue
            if abs_j < 0 or abs_j >= env.height:
                continue
            #NOTE!!! indices in numpy table and the actual location coordinate indices are different
            observed_info[abs_j,abs_i] = obs['image'][vis_i,vis_j]
            #if the observed info is zero, it means we can't see it through the wall, so don't update, 
            #and if we already know the information, then we also don't have to update. 
            if observed_info[abs_j,abs_i] != 0 and observed_info[abs_j,abs_i] != houseLocToObject[ith_house, abs_j, abs_i]:
                houseLocToObject[ith_house, abs_j, abs_i] = observed_info[abs_j,abs_i]
                
                #if the location is where the itmes can show up, we need to update the tables based on whether
                #we have seen objects or not. it also means we have acquired new knowledge. 
                if (abs_j,abs_i) in ball_locations:
                    foundNewKnowledge = True
                    if observed_info[abs_j,abs_i] == 3:
                        #we find ball here, we count 1 more ball encounter when we are in this roomtype
                        roomtypeToObject[obs['roomtype'], 0, 1] +=  1 
                    else:
                        #we didn't find ball here
                        roomtypeToObject[obs['roomtype'], 0, 0] +=  1 
                if (abs_j,abs_i) in box_locations:
                    foundNewKnowledge = True
                    if observed_info[abs_j,abs_i] == 4:
                        roomtypeToObject[obs['roomtype'], 1, 1] +=  1 
                    else:
                        roomtypeToObject[obs['roomtype'], 1, 0] +=  1 
                if (abs_j,abs_i) in box2_locations:
                    foundNewKnowledge = True
                    if observed_info[abs_j,abs_i] == 5:
                        roomtypeToObject[obs['roomtype'], 2, 1] +=  1 
                    else:
                        roomtypeToObject[obs['roomtype'], 2, 0] +=  1

    # then we also need to update table 1 the room layout.
    if houseRoomToType[ith_house, obs['room']]!= obs['roomtype']:
        houseRoomToType[ith_house, obs['room']] = obs['roomtype']
        foundNewKnowledge = True
        #also update our prior, 
        RoomToTypeProb[obs['room'],obs['roomtype']] += 1
    #print(houseRoomToType[ith_house,:])

    #print(observed_info)
    #print(houseLocToObject[ith_house,:,:])
    #print("whether have new knowledge:", foundNewKnowledge)
    #return foundNewKnowledge



#########################################main loop####################################
#each house we visit several times:
eval_num = 40
#reward_set = np.zeros((args.num_envs, 5))
for iter in range(eval_num):
    reward_set = np.zeros((args.num_envs + 6, 78))
    for j in range(args.num_envs):
        env = gym.make(args.env)
        env_set.append(env)
    random.shuffle(env_set)

    # table 1: Map from house id and room location to room type. (-1 menas “I don’t know yet”):
    # houseToRoomtype[0][0] denotes room types of room 0 in house 0. if value is -1 then it means we don't know yet

    houseRoomToType = np.ones((args.num_envs, args.num_rooms), dtype=np.int8) * -1
    # table 2: Map from (nothing) to room type probability. experience we have collected after we wander inside houses
    # initially we just assume that all rooms can be any kind of room. we will use Dirichlet distribution.
    # we set alpha to be (1,1,1,....1). So any kinds of distribution will be uniformly possible
    RoomToTypeProb = np.ones((args.num_rooms, args.num_roomtypes))

    # table 3, zero means we don't know yet
    houseLocToObject = np.zeros((args.num_envs, 13, 13), dtype=np.int8)

    # Map from room types to objects. (times it doesn't show up vs how many times it shows up )
    roomtypeToObject = np.ones((args.num_roomtypes, args.num_objects, 2))
    eval_env_set = [env_set[0], env_set[60], env_set[120], env_set[180], env_set[240], env_set[300]]
    eval_env_index = [0,60,120,180,240,300]
    for ith_visit in range(args.num_visitsPerHouse):
        #loop through all the houses:
        for ith_house in range(args.num_envs):

            env = env_set[ith_house]
            if args.agent_view:
                env = RGBImgPartialObsWrapper(env)
                env = ImgObsWrapper(env)
            #window = Window('gym_minigrid - ' + args.env +' house ' + str(ith_house) +' ' + str(ith_visit) + ' visits')
            #window.reg_key_handler(key_handler)
            #we reset the house environment, which doesn't change the room layout, some minor issues with object
            if ith_house in eval_env_index:
                for house in range(len(eval_env_set)):
                    houseRoomToType[ith_house,:] = houseRoomToType[ith_house,:] * 0
                    houseLocToObject[ith_house, :, :] = houseLocToObject[ith_house, :, :] * 0
                    env = eval_env_set[house]
                    temp_goal = env.goal_type
                    index = np.random.choice(len(env.goal_set))
                    env.goal_type = env.goal_set[index]
                    for episode in range(13):
                        e_reward = 0
                        obs = env.reset2()
                        # figure out what kind of goal we have
                        goal_type = env.goal_type

                        state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]


                        # after we get into a house, first of all we will sample K mdps based on past experience.
                        merged_qtable = sampleMDPs(ith_house, goal_type, env.agent_pos, env.agent_dir)
                        # in merged_qtable, each action has k values(cause we got k mdps), now we only need the max for each action.
                        max_merged_qtable = np.max(merged_qtable, 4)

                        for i in range(20):

                            # we select an action that has the highest value from the smapled MDPs
                            a = np.argmax(max_merged_qtable[state[0], state[1], state[2], :])
                            # then we take a step, after taking a step, we will know if we have found some new knowledge.
                            obs, reward, done, info, foundNewKnowledge = step(a, ith_house, eval=True)

                            state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]

                            if done:
                                e_reward = reward
                                break
                            # if we have found new knowledge, then we need to re sample those mdps based on the new knowledge
                            # Our starting position for the agent in those mdps should be the agent's current location.
                            if foundNewKnowledge:
                                merged_qtable = sampleMDPs(ith_house, goal_type, env.agent_pos, env.agent_dir)
                                max_merged_qtable = np.max(merged_qtable, 4)
                        reward_set[ith_house//60 + args.num_envs, house * 13 + episode] += e_reward
                    env.goal_type = temp_goal
                houseRoomToType[ith_house, :] = houseRoomToType[ith_house, :] * 0
                houseLocToObject[ith_house, :, :] = houseLocToObject[ith_house, :, :] * 0
            for episode in range(78):
                e_reward = 0
                obs = env.reset2()
                #figure out what kind of goal we have
                goal_type = env.goal_type

                state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]

                #testing code
                for j in range(env.grid.height):
                    tempString = ""
                    for i in range(env.grid.width):
                        if env.grid.get(i,j) == None:
                            tempString += 'null' + ' '
                        else:
                            tempString += str(env.grid.get(i,j).type) + ' '
                    print(tempString)


                #after we get into a house, first of all we will sample K mdps based on past experience.
                merged_qtable = sampleMDPs(ith_house, goal_type, env.agent_pos, env.agent_dir)
                #in merged_qtable, each action has k values(cause we got k mdps), now we only need the max for each action.
                max_merged_qtable = np.max(merged_qtable, 4)

                for i in range(20):

                    print('check under current state, value: ', max_merged_qtable[state[0], state[1], state[2], :])
                    #we select an action that has the highest value from the smapled MDPs
                    a = np.argmax(max_merged_qtable[state[0], state[1], state[2], :])
                    #then we take a step, after taking a step, we will know if we have found some new knowledge.
                    obs, reward, done, info, foundNewKnowledge = step(a, ith_house)

                    state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]

                    if done:
                        e_reward = reward
                        break
                    #if we have found new knowledge, then we need to re sample those mdps based on the new knowledge
                    #Our starting position for the agent in those mdps should be the agent's current location.
                    if foundNewKnowledge:
                        merged_qtable = sampleMDPs(ith_house, goal_type, env.agent_pos, env.agent_dir)
                        max_merged_qtable = np.max(merged_qtable, 4)
                reward_set[ith_house, episode] += e_reward
                #print('tem_rew:', reward_set)


            #print('roomtypeToObject:',roomtypeToObject)
            #print('RoomToTypeProb: ', RoomToTypeProb)
            #print('houseLocToObject:', houseLocToObject)
    reward_set = reward_set
    np.save("1rew1new78_{ith}.npy".format(ith=iter), reward_set)
        #print("Finished!")

#print("epi_rew:",reward_set/eval_num)
#reward_set = reward_set/eval_num
#np.save("rew4.npy", reward_set)







