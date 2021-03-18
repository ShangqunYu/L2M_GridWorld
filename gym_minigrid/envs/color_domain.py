from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import numpy as np
class EmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=10,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )
        self.color_reward = np.random.normal([0, 2, 5, 7, 8, 11, 13, 14], [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]).clip(0,15)
    def reset2(self, agent_pos=(1, 1), agent_dir=0):
        # Current position and direction of the agent
        self.agent_pos = agent_pos

        self.agent_dir = agent_dir

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()

        start_cell = self.grid.get(*self.agent_pos)

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()
        return obs
    def _gen_grid(self, width, height):
        # Create an empty grid
        self.rew_map = np.ones((width-2, height-2))
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for i in range(width-2):
            for j in range(height-2):
                if i==(width-3) and j ==(height-3):
                    self.put_obj(Ball(), i + 1, j + 1)
                    continue

                index = np.random.choice(8)

                if index == 0:
                    self.put_obj(Goal1(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([0], [1.]).clip(0,15)[0]
                elif index == 1:
                    self.put_obj(Goal2(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([2], [1.]).clip(0, 15)[0]
                elif index == 2:
                    self.put_obj(Goal3(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([5], [1.]).clip(0, 15)[0]
                elif index == 3:
                    self.put_obj(Goal4(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([7], [1.]).clip(0, 15)[0]
                elif index == 4:
                    self.put_obj(Goal5(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([8], [1.]).clip(0, 15)[0]
                elif index == 5:
                    self.put_obj(Goal6(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([11], [1.]).clip(0, 15)[0]
                elif index == 6:
                    self.put_obj(Goal7(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([13], [1.]).clip(0, 15)[0]
                else:
                    self.put_obj(Goal8(), i + 1, j + 1)
                    self.rew_map[i, j] = np.random.normal([14], [1.]).clip(0, 15)[0]


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()


    def step(self, action):
        self.step_count += 1
        action_set = [self.actions.forward, self.actions.left, self.actions.right]
        action = action_set[action]
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4
            fwd_pos1 = self.front_pos
            fwd_cell1 = self.grid.get(*fwd_pos1)
            if fwd_cell1 == None or fwd_cell1.can_overlap():
                self.agent_pos = fwd_pos1
        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4
            fwd_pos1 = self.front_pos
            fwd_cell1 = self.grid.get(*fwd_pos1)
            if fwd_cell1 == None or fwd_cell1.can_overlap():
                self.agent_pos = fwd_pos1
        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            '''if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True'''

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        reward, done, obs_r = self.get_reward()


        obs = self.gen_obs()
        obs['pos'] = np.array(obs['pos']).flatten()
        obs.update(obs_r)
        return obs, reward, done, {}

    def get_reward(self, color_reward=None):
        if color_reward is None:
            color_reward = self.color_reward
        reward = 0
        done = False
        obs = {'goal1':[],'goal2':[],'goal3':[],'goal4':[],'goal5':[], 'goal6':[], 'goal7':[], 'goal8':[]}
        cell = self.grid.get(self.agent_pos[0]-1, self.agent_pos[1])
        if cell.type == 'wall':
            reward += -10
        elif cell.type == 'ball':
            done = True
        else:
            reward += -self.rew_map[self.agent_pos[0]-2, self.agent_pos[1]-1]
            obs[cell.type].append(self.rew_map[self.agent_pos[0]-2, self.agent_pos[1]-1])
        cell = self.grid.get(self.agent_pos[0]+1, self.agent_pos[1])
        if cell.type == 'wall':
            reward += -10
        elif cell.type == 'ball':
            done = True
        else:
            reward += -self.rew_map[self.agent_pos[0], self.agent_pos[1]-1]
            obs[cell.type].append(self.rew_map[self.agent_pos[0], self.agent_pos[1]-1])
        cell = self.grid.get(self.agent_pos[0], self.agent_pos[1]-1)
        if cell.type == 'wall':
            reward += -10
        elif cell.type == 'ball':
            done = True
        else:
            reward += -self.rew_map[self.agent_pos[0]-1, self.agent_pos[1]-2]
            obs[cell.type].append(self.rew_map[self.agent_pos[0]-1, self.agent_pos[1]-2])
        cell = self.grid.get(self.agent_pos[0], self.agent_pos[1]+1)
        if cell.type == 'wall':
            reward += -10
        elif cell.type == 'ball':
            done = True
        else:
            reward += -self.rew_map[self.agent_pos[0]-1, self.agent_pos[1]]
            obs[cell.type].append(self.rew_map[self.agent_pos[0]-1, self.agent_pos[1]])

            # if cell.type == 'goal1':
            #     reward += -color_reward[0]
            # elif cell.type == 'goal2':
            #     reward += -color_reward[1]
            # elif cell.type == 'goal3':
            #     reward += -color_reward[2]
            # elif cell.type == 'goal4':
            #     reward += -color_reward[3]
            # elif cell.type == 'wall':
            #     reward += -10
            # elif cell.type == 'ball':
            #     done = True

        return reward, done, obs
class EmptyEnv5x5(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=5, **kwargs)

class EmptyRandomEnv5x5(EmptyEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None)

class EmptyEnv6x6(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)

class EmptyRandomEnv6x6(EmptyEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None)

class EmptyEnv16x16(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=16, **kwargs)



register(
    id='MiniGrid-Empty-10x10-v0',
    entry_point='gym_minigrid.envs:EmptyEnv'
)


