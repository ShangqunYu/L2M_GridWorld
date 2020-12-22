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

    def __init__(self, agent_pos=None, goal_pos=None, fixed_room_dist=False):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.fixed_room_dist = fixed_room_dist

        super().__init__(grid_size=13, max_steps=100,agent_view_size=3)
        #print(1)
        self.observation_space = spaces.Box(
            low=0,
            high=5,
            shape=(self.agent_view_size, self.agent_view_size),
            dtype='uint8'
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

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
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        #if self._agent_default_pos is not None:
        self.agent_pos = (7,1)
        self.grid.set(7,1, None)
        self.agent_dir = 1  # assuming random start direction
        #else:
        #    self.place_agent()

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
            p = [[1,1],[1,7],[7,1],[7,7]]
            index = np.random.choice(4,4, p = [0.4, 0.3, 0.2, 0.1],replace=False)

            self.Room1(p[index[0]][0], p[index[0]][1])
            self.Room2(p[index[1]][0], p[index[1]][1])
            self.Room3(p[index[2]][0], p[index[2]][1])
            self.Room4(p[index[3]][0], p[index[3]][1])
        self.mission = 'Reach the goal'
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
    def Room2(self, top_x=7, top_y=1):
        if np.random.random() > 0.2:
            if np.random.random() > 0.6:
                self.put_obj(Ball(), top_x+3, top_y+1)
            else:
                self.put_obj(Ball(), top_x+3, top_y+3)
        if np.random.random() > 0.2:
            if np.random.random() > 0.7:
                self.put_obj(Box(self.color1), top_x, top_y+4)
            else:
                self.put_obj(Box(self.color1), top_x+2, top_y+4)

    def Room3(self, top_x=1, top_y=7):
        if np.random.random() > 0.1:
            if np.random.random() > 0.8:
                self.put_obj(Ball(), top_x+3, top_y+3)
            else:
                self.put_obj(Ball(), top_x+1, top_y+3)
        if np.random.random() > 0.1:
            if np.random.random() > 0.7:
                self.put_obj(Box(self.color2), top_x+4, top_y)
            else:
                self.put_obj(Box(self.color2), top_x+4, top_y+2)

    def Room4(self, top_x=7, top_y=7):
        if np.random.random() > 0.1:
            if np.random.random() > 0.5:
                self.put_obj(Box(self.color1), top_x+1, top_y+1)
            else:
                self.put_obj(Box(self.color1), top_x+2, top_y)
        if np.random.random() > 0.1:
            self.put_obj(Box(self.color2), top_x+2, top_y+2)
    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        obs = np.array(obs).flatten()
        #print('o:',obs)
        return obs, reward, done, info

register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)
