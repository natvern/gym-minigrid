from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from operator import add
import sys
sys.path.append('../..')
from dio.prologkb.diokb import Dio
import dio.prologkb.config as config
import numpy
import csv
import threading


## Extension of DynamicObstacles Environment.
class ProbGridEnv(MiniGridEnv):
    def __init__(
            self,
            size=8,
            agent_start_pos=(1, 1),
            agent_start_dir=0,
            n_obstacles=4,
            goal = True
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.dio = Dio()
        self.setGoal = goal
        self.goal = (0,0)
        self.steps = 0
        self.weight = config.config.weight_dio 
        self.FORWARD = 2 
        self.LEFT = 0 
        self.RIGHT = 1

        self.n_obstacles = int(n_obstacles)

        # Reduce obstacles if there are too many
        #if n_obstacles <= size/2 + 1:
        #    self.n_obstacles = int(n_obstacles)
        #else:
        #    self.n_obstacles = int(size/2)
        super().__init__(
            grid_size=size,
            max_steps=config.config.max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
        )
        # Allow only 4 actions permitted: left, right, top, bottom
        self.action_space = spaces.Discrete(4)
        self.reward_range = (-1, 1)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        if self.setGoal:
            self.grid.set(width-2, height-2, Goal())
            self.goal = (width-2, height-2)
        else:
            goalX = numpy.random.randint(0, high=width-2)
            goalY = numpy.random.randint(0, high=height-2)
            self.grid.set(goalX, goalY, Goal())
            self.goal = (goalX, goalY)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.n_obstacles):
            self.obstacles.append(Ball())
            self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.mission = "get to the green goal square"

    def step(self, action):
        self.steps += 1

        # Invalid action
        if action >= self.action_space.n:
            action = 0

        # Check if there is an obstacle in front of the agent
        front_cell = self.grid.get(*self.front_pos)
        not_clear = front_cell and front_cell.type != 'goal'

        # Update obstacle positions
        for i_obst in range(len(self.obstacles)):
            old_pos = self.obstacles[i_obst].cur_pos
            top = tuple(map(add, old_pos, (-1, -1)))

            try:
                if (numpy.random.rand() <= 0.8): 
                    self.place_obj(self.obstacles[i_obst], top=top, size=(3,3), max_tries=100)
                    self.grid.set(*old_pos, None)
            except:
                pass

        # Update the agent's position/direction

        posobs = []
        direction = None
        for obs in self.obstacles:
            posobs += [obs.cur_pos]

        if (action == 0): # Move Right
            direction = 0
            if (self.agent_dir == 1):
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
            elif (self.agent_dir == 2):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            elif (self.agent_dir == 3):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            obs, reward, done, info = MiniGridEnv.step(self, self.FORWARD)
        
        if (action == 1): # Move Left
            direction = 2
            if (self.agent_dir == 0):
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
            elif (self.agent_dir == 1):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            elif (self.agent_dir == 3):
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
            obs, reward, done, info = MiniGridEnv.step(self, self.FORWARD)
        
        if (action == 2): # Move Up
            direction = 3
            if (self.agent_dir == 0):
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
            elif (self.agent_dir == 1):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            elif (self.agent_dir == 2):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            obs, reward, done, info = MiniGridEnv.step(self, self.FORWARD)

        if (action == 3): # Move Down 
            direction = 1
            if (self.agent_dir == 0):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            elif (self.agent_dir == 2):
                obs, reward, done, info = MiniGridEnv.step(self, self.LEFT)
            elif (self.agent_dir == 3):
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
                obs, reward, done, info = MiniGridEnv.step(self, self.RIGHT)
            obs, reward, done, info = MiniGridEnv.step(self, self.FORWARD)

        self.dio.updateWorld(self.agent_pos, direction, posobs, self.goal, self.steps)

        # If the agent exhausted the number of steps allowed 
        if (self.steps >= config.config.max_steps):
            reward = config.config.reward_exhaust
            done = True
            info["termination"] = 0
            self.steps = 0

        # If the agent tried to walk over an obstacle or wall
        if action == self.actions.forward and not_clear:
            reward = config.config.reward_fail
            done = True
            info["termination"] = -1
            self.steps = 0

        if done and self.steps != 0:
            reward = config.config.reward_succ 
            info["termination"] = 1
            self.steps = 0

        elif not done:
            reward = config.config.reward_life
        
        dio_feedback = self.dio.getFeedback() 
    
        ## UPDATE OF THE REWARD GIVEN CALL TO DIO
        reward = reward + dio_feedback

        return obs, reward, done, info

class ProbGridEnv5x5(ProbGridEnv):
    def __init__(self):
        super().__init__(size=5, n_obstacles=2)

class ProbGridRandomEnv5x5(ProbGridEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None, n_obstacles=2, goal=False)

class ProbGridRandomGoalEnv5x5(ProbGridEnv):
    def __init__(self):
        super().__init__(size=5, n_obstacles=2, goal=False)

class ProbGridRandomInitEnv5x5(ProbGridEnv):
    def __init__(self):
        super().__init__(size=5, agent_start_pos=None, n_obstacles=2)

class ProbGridEnv6x6(ProbGridEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=3)

class ProbGridRandomEnv6x6(ProbGridEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_obstacles=2, goal=False)

class ProbGridRandomGoalEnv6x6(ProbGridEnv):
    def __init__(self):
        super().__init__(size=6, n_obstacles=2, goal=False)

class ProbGridRandomInitEnv6x6(ProbGridEnv):
    def __init__(self):
        super().__init__(size=6, agent_start_pos=None, n_obstacles=2)

class ProbGridEnv8x8(ProbGridEnv):
    def __init__(self):
        super().__init__(size=8, n_obstacles=config.config.obstacles)

class ProbGridRandomEnv8x8(ProbGridEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None, n_obstacles=config.config.obstacles, goal=False)

class ProbGridRandomGoalEnv8x8(ProbGridEnv):
    def __init__(self):
        super().__init__(size=8, n_obstacles=config.config.obstacles, goal=False)

class ProbGridRandomInitEnv8x8(ProbGridEnv):
    def __init__(self):
        super().__init__(size=8, agent_start_pos=None, n_obstacles=config.config.obstacles)

class ProbGridEnv16x16(ProbGridEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=config.config.obstacles)

class ProbGridRandomEnv16x16(ProbGridEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None, n_obstacles=config.config.obstacles, goal=False)

class ProbGridRandomGoalEnv16x16(ProbGridEnv):
    def __init__(self):
        super().__init__(size=16, n_obstacles=config.config.obstacles, goal=False)

class ProbGridRandomInitEnv16x16(ProbGridEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None, n_obstacles=config.config.obstacles)


class ProbGridEnv40x40(ProbGridEnv):
    def __init__(self):
        super().__init__(size=40, n_obstacles=config.config.obstacles)

class ProbGridEnv100x100(ProbGridEnv):
    def __init__(self):
        super().__init__(size=100, n_obstacles=config.config.obstacles)

register(
    id='MiniGrid-ProbGrid-5x5-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv5x5'
)

register(
    id='MiniGrid-ProbGrid-Random-5x5-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomEnv5x5'
)

register(
    id='MiniGrid-ProbGrid-Random-Goal-5x5-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomGoalEnv5x5'
)

register(
    id='MiniGrid-ProbGrid-Random-Init-5x5-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomInitEnv5x5'
)

register(
    id='MiniGrid-ProbGrid-6x6-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv6x6'
)

register(
    id='MiniGrid-ProbGrid-Random-6x6-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomEnv6x6'
)

register(
    id='MiniGrid-ProbGrid-Random-Goal-6x6-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomGoalEnv6x6'
)

register(
    id='MiniGrid-ProbGrid-Random-Init-6x6-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomInitEnv6x6'
)

register(
    id='MiniGrid-ProbGrid-8x8-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv8x8'
)

register(
    id='MiniGrid-ProbGrid-Random-8x8-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomEnv8x8'
)

register(
    id='MiniGrid-ProbGrid-Random-Goal-8x8-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomGoalEnv8x8'
)

register(
    id='MiniGrid-ProbGrid-Random-Init-8x8-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomInitEnv8x8'
)

register(
    id='MiniGrid-ProbGrid-16x16-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv16x16'
)


register(
    id='MiniGrid-ProbGrid-Random-16x16-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomEnv16x16'
)

register(
    id='MiniGrid-ProbGrid-Random-Goal-16x16-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomGoalEnv16x16'
)

register(
    id='MiniGrid-ProbGrid-Random-Init-16x16-v0',
    entry_point='gym_minigrid.envs:ProbGridRandomInitEnv16x16'
)


register(
    id='MiniGrid-ProbGrid-40x40-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv40x40'
)

register(
    id='MiniGrid-ProbGrid-100x100-v0',
    entry_point='gym_minigrid.envs:ProbGridEnv100x100'
)
