#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
import random

class FourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, fixed_room_dist=False, test=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.fixed_room_dist = fixed_room_dist
        goal_typeset = ['ball','box1', 'box2']

        print(test)

        self.room_set = []
        super().__init__(grid_size=13, max_steps=1000,agent_view_size=3, goal_type=goal_typeset)
        #print(1)
        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(self.agent_view_size, self.agent_view_size),
            dtype='uint8'
        )
    # defalut is the starting position, but for imagined mdps, we may want to modify the starting position
    def reset2(self, agent_pos = (7, 1), agent_dir = 1):
        # Current position and direction of the agent
        self.agent_pos = agent_pos
        self.grid.set(agent_pos[0], agent_pos[1], None)
        self.agent_dir = agent_dir

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()

        start_cell = self.grid.get(*self.agent_pos)


        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        #print("obs:",obs)
        return obs
    # recreate our imagining house
    def recreate(self, objects_in_rooms, rooms, goal_type):
        self.grid = Grid(self.width, self.height)
        self.room_set = []
        self.goal_type = goal_type
        self.buildWallsAndDoors(self.width, self.height)
        self.setAgentStartPos()
        '''four rooms in a house
        rooms are numbered as following pattern:
            0 2
            1 3
        '''
        tops = [[1,1],[1,7],[7,1],[7,7]]
        for ith_room in range(len(objects_in_rooms)):
            top_i = tops[ith_room]
            objects_ith_room = objects_in_rooms[ith_room]
            self.putObjectsInRoom(top_i[0],top_i[1],objects_ith_room[0],objects_ith_room[1],objects_ith_room[2])
            self.room_set.append(rooms[ith_room])



        obs = self.gen_obs()
        return obs

    def buildWallsAndDoors(self, width, height):
        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    #we fixed the positon of the doors
                    pos = (xR, yT + 3)
                    #pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    #we fixed the positon of the doors
                    pos = (xL + 3, yB)
                    #pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)
    def setAgentStartPos(self):
        self.agent_pos = (7,1)
        self.grid.set(7,1, None)
        self.agent_dir = 1          

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)
        self.goal_set = []

        self.buildWallsAndDoors(width, height)

        self.setAgentStartPos()

        doorColors = set(COLOR_NAMES)

        self.color1 = self._rand_elem(sorted(doorColors))
        self.color2 = self._rand_elem(sorted(doorColors))
        ###Room 1###
        if self.fixed_room_dist:
            if np.random.random() > 0.3:
                if np.random.random() > 0.5:
                    self.put_obj(ball, 5,1)
                else:
                    self.put_obj(ball, 5, 5)
            if np.random.random() > 0.3:
                if np.random.random() > 0.5:
                    self.put_obj(Box(), 3, 1)
                else:
                    self.put_obj(Box(), 3, 3)
            if np.random.random() > 0.3:
                self.put_obj(Box2(), 1, 5)
            ###Room 2###
            if np.random.random() > 0.2:
                if np.random.random() > 0.6:
                    self.put_obj(ball, 10, 2)
                else:
                    self.put_obj(ball, 10, 4)
            if np.random.random() > 0.2:
                if np.random.random() > 0.7:
                    self.put_obj(Box(), 7, 5)
                else:
                    self.put_obj(Box(), 9, 5)
            ###Room 3###
            if np.random.random() > 0.1:
                if np.random.random() > 0.8:
                    self.put_obj(ball, 4, 10)
                else:
                    self.put_obj(ball, 2, 10)
            if np.random.random() > 0.1:
                if np.random.random() > 0.7:
                    self.put_obj(Box2(), 5, 7)
                else:
                    self.put_obj(Box2(), 5, 9)
            ###Room 4###
            if np.random.random() > 0.1:
                if np.random.random() > 0.5:
                    self.put_obj(Box(), 8, 8)
                else:
                    self.put_obj(Box(), 9, 7)
            if np.random.random() > 0.1:
                self.put_obj(Box2(), 9, 9)
        else:
            tops = [[1,1],[1,7],[7,1],[7,7]]
            # distribution of roomtype for each room. 
            p = [[0.8, 0., 0.2], [0., 1.0, 0.], [0.2, 0., 0.8], [0.7, 0., 0.3]]

            for i in range(4):
                top_i = tops[i]
                p_i = p[i]
                room_i = np.random.choice(3, p=p[i])
                if room_i == 0:
                    self.Room2(top_i[0], top_i[1])
                elif room_i == 1:
                    self.Room3(top_i[0], top_i[1])
                else:
                    self.Room4(top_i[0], top_i[1])
                self.room_set.append(room_i)
        self.mission = 'Reach the goal'
    #roomtype1 didn't used
    def Room1(self, top_x=1, top_y=1):
        if np.random.random() > 0.3:
            if np.random.random() > 0.5:
                self.put_obj(Ball(), top_x+4, top_y)
            else:
                self.put_obj(Ball(), top_x+4, top_y+4)
        if np.random.random() > 0.3:
            if np.random.random() > 0.5:
                self.put_obj(Box(self.color1), top_x+2, top_y)
            else:
                self.put_obj(Box(self.color1), top_x+2, top_y+2)
        if np.random.random() > 0.3:
            self.put_obj(Box(self.color2), top_x, top_y+4)
    #room2 80% has ball, 20% has box, 5% has box2
    def Room2(self, top_x=7, top_y=1):
        self.putObjectsInRoom(top_x, top_y, 0.2, 0.8, 0.)

    #room3 20% has ball, 5% has box, 80% has box2
    def Room3(self, top_x=1, top_y=7):
        self.putObjectsInRoom(top_x, top_y, 0., 0., 1.)
    #room4 5% has ball, 80 has box, 20% has box2
    def Room4(self, top_x=7, top_y=7):
        self.putObjectsInRoom(top_x, top_y, 0.8, 0.2, 0.)

    def putObjectsInRoom(self, top_x, top_y, probs_ball, probs_box, probs_box2):
        if np.random.random() < probs_ball:
            self.put_obj(Ball(), top_x + 4, top_y + 4)
            if 'ball' not in self.goal_set:
                self.goal_set.append('ball')
        if np.random.random() < probs_box:
            self.put_obj(Box(), top_x + 4, top_y)
            if 'box' not in self.goal_set:
                self.goal_set.append('box')
        if np.random.random() < probs_box2:
            self.put_obj(Box2(), top_x+2, top_y+2)
            if 'box2' not in self.goal_set:
                self.goal_set.append('box2')

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        #obs['image'] = np.array(obs['image']).flatten()
        obs['pos'] = np.array(obs['pos']).flatten()
        if obs['pos'][0] < 6 and obs['pos'][1] < 6:
            obs['roomtype'] = self.room_set[0]
            obs['room'] = 0
        elif obs['pos'][0] < 6 and obs['pos'][1] > 6:
            obs['roomtype'] = self.room_set[1]
            obs['room'] = 1
        elif obs['pos'][0] > 6 and obs['pos'][1] < 6:
            obs['roomtype'] = self.room_set[2]
            obs['room'] = 2
        else:
            obs['roomtype'] = self.room_set[3]
            obs['room'] = 3
        #print('o:',obs)
        return obs, reward, done, info

register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)

