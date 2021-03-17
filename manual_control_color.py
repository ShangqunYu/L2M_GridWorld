#!/usr/bin/env python3
import random
import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
import copy

def redraw(img):
    if not args.agent_view:
      img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

    pass


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset2()


    redraw(obs)


def reset2():
    obs = env.reset2()

    redraw(obs)


def step(action, index=0, env=None, eval=False):
    obs, reward, done, info = env.step(action)
    # print('step=%s, reward=%.2f' % (env.step_count, reward))
    obs['house'] = index

    # if eval:
    #     foundNewKnowledge = updateKnowledge_eval(env, obs, index)
    # else:
    #     foundNewKnowledge = updateKnowledge(env, obs, index)
    foundNewKnowledge = False
    redraw(obs)

    if done:
        pass
        # print('done!')
        # reset2()

    return obs, reward, done, info, foundNewKnowledge


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        # window.close()
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
    default=4
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

parser.add_argument(
    '--env_width',
    type=int,
    help="width of the environment",
    default=12
)
parser.add_argument(
    '--env_height',
    type=int,
    help="height of the environment",
    default=12
)

args = parser.parse_args()
env_set = []
min_epsilon = 0.20
max_epsilon = 1.0
decay_rate = 0.0005
gamma = 0.95


def sampleweights(ith_house):
    # first of we check what we know, -1 means we don't know for a particular room
    # rooms = np.copy(houseRoomToType[ith_house])
    color_reward = np.random.normal(model_mu, model_sigma).clip(0,10)
    # for ith_room in range(len(rooms)):
    #     # if we don't know what kind of room the ith_room is, we take a guess based on the past experience
    #     # if we know what room it is, no need to take a guess.
    #     if rooms[ith_room] == -1:
    #         # we sample a distribution based on the prior and then take a guess based on the distribution
    #         prob = np.random.dirichlet(alpha=RoomToTypeProb[ith_room])
    #         rooms[ith_room] = np.random.choice(args.num_roomtypes, p=prob)
    return color_reward


def selectAction(qtable, state, epsilon, i):
    e = random.uniform(0, 1)
    if e < epsilon:
        a = np.random.choice(3)
    else:
        a = np.argmax(qtable[state[0], state[1], state[2], :])
    return a


def value_iteration(num_iterations, colorrewards, env, id):
    V = np.zeros((args.env_width, args.env_height, 4))
    Q = np.zeros((args.env_width, args.env_height, 4, 3, args.K))
    for i in range(num_iterations):
        V_old = V.copy()
        Q_old = Q.copy()
        bellman_update(V, Q, env, colorrewards, id)
        oldstate = np.concatenate([V_old.reshape((-1)), Q_old.reshape((-1))])
        newstate = np.concatenate([V.reshape((-1)), Q.reshape((-1))])
        diff = newstate - oldstate
        diffnorm = np.linalg.norm(diff, ord=2)
        # if the update is very small, we assume it converged.
        # also if there is no target object in the house, it will end in the first iteration as well.
        if diffnorm < 0.01:
            break
    return V, Q

def bellman_update(V, Q, env, colorrewards, id):
    for i in range(args.env_width):
        for j in range(args.env_height):
            #we only update states whose location is not occupied by walls or objects.

                    # four different orientations
            for k in range(4):
                for m in range(args.K):

                    # we only update states whose location is not occupied by walls or objects.

                    if env.grid.get(i, j).type != 'wall':

                        s = (i, j, k)

                        # if the state is a goal state in any of the mdps, we don't give it a value
                        if is_goalState(env, s):
                            continue
                        else:
                            for ai in range(3):
                                s_prime = transition(env, s, ai)
                                # print('s: ',s, 'action: ', ai, 's_prime:', s_prime)
                                Q[i, j, k, ai, m] = rewardFunc(env, s, ai, s_prime, colorrewards, id) + gamma * V[s_prime]
                            # print(Q[s])
                            V[s] = Q[s].max()


# the goal state is when the agent is in front of the target object.
def is_goalState(env, s):
    # find out the front location of the current state

    if env.grid.get(s[0] - 1, s[1]).type == 'ball' or env.grid.get(s[0] + 1, s[1]).type == 'ball' or env.grid.get(
            s[0], s[1] - 1).type == 'ball' or env.grid.get(s[0], s[1] + 1).type == 'ball':
        return True
    return False

def rewardFunc(env, s, ai, s_prime, color_reward, id):
    reward = 0
    grid1 = env.grid.get(s_prime[0] - 1, s_prime[1])
    grid2 = env.grid.get(s_prime[0] + 1, s_prime[1])
    grid3 = env.grid.get(s_prime[0], s_prime[1] - 1)
    grid4 = env.grid.get(s_prime[0], s_prime[1] + 1)
    if reward_map[id, s_prime[0], s_prime[1]] != -1:
        reward = reward_map[id, s_prime[0], s_prime[1]]
    else:
        for cell in [grid1, grid2, grid3, grid4]:
            if cell.type == 'wall':
                reward -= 8
            elif cell.type == 'goal1':
                reward -= np.random.normal([model_mu[0]], [model_sigma[0]]).clip(0,10)[0]
            elif cell.type == 'goal2':
                reward -= np.random.normal([model_mu[1]], [model_sigma[1]]).clip(0,10)[0]
            elif cell.type == 'goal3':
                reward -= np.random.normal([model_mu[2]], [model_sigma[2]]).clip(0,10)[0]
            elif cell.type == 'goal4':
                reward -= np.random.normal([model_mu[3]], [model_sigma[3]]).clip(0,10)[0]
            elif cell.type == 'goal5':
                reward -= np.random.normal([model_mu[4]], [model_sigma[4]]).clip(0,10)[0]



    return reward

def front_pos(i, j, k):
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


# for value iteration.
def transition(env, s, a):
    # if action is turning left
    if a == 1:
        s_leftTurn = (s[0], s[1], s[2] - 1 if s[2] - 1 >= 0 else 3)


        front_position = front_pos(s_leftTurn[0], s_leftTurn[1], s_leftTurn[2])

        # if the forward postion is not empty, agent can't go there
        if env.grid.get(front_position[0], front_position[1]).type=='ball' or env.grid.get(front_position[0], front_position[1]).type=='wall':

            s_forward = s
            return s_forward
        else:
            s_forward = (front_position[0], front_position[1], s_leftTurn[2])
            return s_forward

    # if action is turning right
    elif a == 2:
        s_rightTurn = (s[0], s[1], s[2] + 1 if s[2] + 1 <= 3 else 0)
        front_position = front_pos(s_rightTurn[0], s_rightTurn[1], s_rightTurn[2])
        # if the forward postion is not empty, agent can't go there
        if env.grid.get(front_position[0], front_position[1]).type == 'ball' or env.grid.get(front_position[0],
                                                                                        front_position[1]).type == 'wall':
            s_forward = s
            return s_forward
        else:
            s_forward = (front_position[0], front_position[1], s_rightTurn[2])
            return s_forward
    # if action is moving forward

    elif a == 0:
        front_position = front_pos(s[0], s[1], s[2])
        # if the forward postion is not empty, agent can't go there
        if env.grid.get(front_position[0], front_position[1]).type == 'ball' or env.grid.get(front_position[0],
                                                                                        front_position[1]).type == 'wall':
            s_forward = s
            return s_forward
        else:
            s_forward = (front_position[0], front_position[1], s[2])
            return s_forward
    else:
        assert False, "invalid action"


def sampleMDPs(ith_house, env):

    # a list that contains all sampled environment
    colorRewardList = []
    #here we create k mdps, k imagined environment based on past experience,
    # for ith_mdp in range(args.K):
    #
    #     color_reward=sampleweights(ith_house)
    #
    #
    #     colorRewardList.append(color_reward)

    # we have 1 q tables, the shape(width, height, num_directions, num_actions, K)
    V, Q = value_iteration(60, colorRewardList, env, ith_house)

    #print('qtable shape: ', Q.shape)

    Q_max = np.max(Q, 4)

    return Q_max

def UpdateModels(Red_rew, Yellow_rew, Blue_rew, Green_rew, Purple_rew):
    rew = [Red_rew, Yellow_rew, Blue_rew, Green_rew, Purple_rew]
    for i in range(5):
        rew_n = rew[i]
        N = len(rew_n)
        mu = model_mu[i]
        mu_var = model_mu_var[i]
        sigma = model_sigma[i]

        mean = np.mean(rew_n)
        mu_temp = N*mu_var/(N*mu_var+sigma)*mean + sigma/(N*mu_var+sigma)*mu
        var_temp = mu_var*sigma/(N*mu_var+sigma)
        model_mu[i] = mu_temp
        model_mu_var[i] = var_temp


#########################################main loop####################################
# each house we visit several times:
eval_num = 50

model_mu = [1, 1, 1, 1, 1]
model_mu_var = [1, 1, 1, 1, 1]
model_sigma = [1, 1, 1, 1, 1]

for iter in range(eval_num):

    reward_set = np.zeros((args.num_envs, 2))
    reward_set_eval = np.zeros((5, args.num_envs//2))
    reward_map = np.ones((args.num_envs, args.env_width, args.env_height))*-1
    for j in range(args.num_envs):
        env = gym.make(args.env)
        env_set.append(env)
    random.shuffle(env_set)



    eval_env_set = [env_set[0], env_set[20], env_set[60], env_set[120], env_set[240]]
    eval_env_index = [0, 20, 60, 120, 240]
    Red_rew = []
    Yellow_rew = []
    Blue_rew = []
    Green_rew = []
    Purple_rew = []
    global_step = 0
    for ith_visit in range(args.num_visitsPerHouse):
        # loop through all the houses:
        for ith_house in range(args.num_envs):
            print("visiting house No.", ith_house)
            print("mu", model_mu)
            print("var", model_mu_var)
            window = Window(
               'gym_minigrid - ' + args.env + ' house ' + str(ith_house) + ' ' + str(ith_visit) + ' visits')
            env = env_set[ith_house]
            if args.agent_view:
                env = RGBImgPartialObsWrapper(env)
                env = ImgObsWrapper(env)

            know_n = 0
            for episode in range(2):

                e_reward = 0
                obs = env.reset2()


                state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]
                Q_max = sampleMDPs(ith_house, env)


                for i in range(100):

                    # print('check under current state, value: ', max_merged_qtable[state[0], state[1], state[2], :])
                    # we select an action that has the highest value from the smapled MDPs
                    a = np.argmax(Q_max[state[0], state[1], state[2], :])
                    # then we take a step, after taking a step, we will know if we have found some new knowledge.
                    obs, reward, done, info, foundNewKnowledge = step(a, ith_house, env)
                    global_step += 1
                    state = [env.agent_pos[0], env.agent_pos[1], env.agent_dir]

                    e_reward += reward
                    if reward_map[ith_house, env.agent_pos[0], env.agent_pos[1]] == -1:
                        know_n +=1
                        Green_rew.extend(obs['goal1'])
                        Blue_rew.extend(obs['goal2'])
                        Red_rew.extend(obs['goal3'])
                        Purple_rew.extend(obs['goal4'])
                        Yellow_rew.extend(obs['goal5'])
                    reward_map[ith_house, env.agent_pos[0], env.agent_pos[1]] = reward
                    if len(Red_rew)>=20:
                        UpdateModels(Red_rew, Yellow_rew, Blue_rew, Green_rew, Purple_rew)
                        Red_rew = []
                        Yellow_rew = []
                        Blue_rew = []
                        Green_rew = []
                        Purple_rew = []
                        Q_max = sampleMDPs(ith_house, env)
                    if know_n%10 == 0:
                        # Q_max = sampleMDPs(ith_house, env)

                        Q_max = sampleMDPs(ith_house, env)
                    if done:
                        break
                        # Q_max = sampleMDPs(ith_house, goal_type, env.agent_pos, env.agent_dir)
                        #max_merged_qtable = np.max(merged_qtable, 4)
                #print('ep:', episode, 'reward:', e_reward, end = ' ', flush=True)
                reward_set[ith_house, episode] += e_reward
            #print("average reward=", sum(reward_set[ith_house, :]) / 60)

    reward_set = reward_set
    reward_set_eval = reward_set_eval
    np.save("./results/3rew2new2_{ith}.npy".format(ith=iter), reward_set)
    # np.save("./results/3rew2eval2_{ith}.npy".format(ith=iter), reward_set_eval)
