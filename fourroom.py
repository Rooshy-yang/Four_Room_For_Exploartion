#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register

"""
action space [-2,2]^2
observation space 10*10:
"""

AN = 1.0                # action norm
SN = 5.0                # space norm
RA = 0.5                # if current state is within the circle of target, then done

DL = 2.3                # door low
DH = 2.7                # door high
DM = 0.5 * (DL + DH)    # door mid
SDM = 1.4142135623730951 * DM   # near door

IS = 5               # initial seed of random generator
ERROR = 1e-3            # numerical error
COE = 0.01          # reward coefficients
FRICTION = 0.5

DOORS = np.array([[0.0, DM], [0.0, -DM], [DM, 0.0], [-DM, 0.0]], dtype=np.float32)

norm = np.linalg.norm


class FourRoom(gym.Env):

    def __init__(self):

        # define continuous action space and continuous observation space
        self.action_space = spaces.Box(low=-AN, high=AN, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-SN, high=SN, shape=(2,), dtype=np.float32)

        # random number generator
        self.rng = np.random.RandomState(IS)

        # Re target
        self.target = None
        self.target_area = None

        # current state
        self.state = None
        self.area = None

        #
        self.original_target = np.array([SN / 2, SN / 2], dtype=np.float)
        self.original_state = np.array([-SN / 2, -SN / 2], dtype=np.float)
        self.re_target()

    def re_target(self):
        """Reset the target"""

        # target
        self.target = self.original_target.copy()
        # target area
        self.target_area = ''
        for i in range(2):
            if self.target[i] > 0:
                self.target_area += '+'
            else:
                self.target_area += '-'

    def seed(self, seed=None):
        """Seed the environment"""

        if seed is not None:
            self.rng = np.random.RandomState(seed)

    def reset(self):
        """Reset the environment (do not reset goal)"""

        while True:
            self.state = self.original_state.copy()
            distance = norm(self.state - self.target)
            self._update_area()
            if distance > RA and self.area != self.target_area:
                break

        # return self.state, self.target
        return self.state

    def step(self, action):
        """Step the environment"""

        # init reward
        distance = self._get_distance()
        reward = COE * distance

        # transform original state to [0,SN]*[0,SN]
        state = self.state.copy()
        door = np.zeros(2, dtype=np.float32)
        for i in range(2):
            if self.area[i] == '-':
                state[i] += SN
                door[i] = SN

        # calculate the state
        state = self._move(state, action, door)

        # change to original space
        for i in range(2):
            if self.area[i] == '+':
                self.state[i] = state[i]
            else:
                self.state[i] = state[i] - SN

        # update area and door
        self._update_area()

        # Calculate reward
        distance = self._get_distance()
        if distance < RA:
            reward = 1.0
            # for exploration
            # done = True
            done = False
        else:
            reward -= COE * distance
            done = False

        return self.state, reward, done, None

    def render(self, mode='human'):
        return

    def _random_state(self):
        """Return random state in the observation space"""

        # avoid the state is on the wall
        while True:
            state = self.rng.uniform(-SN, SN, 2)
            if np.abs(state[0]) > ERROR and np.abs(state[1]) > ERROR:
                break

        return state

    def _update_area(self):
        """Update the doors"""

        # update area
        self.area = ''
        for i in range(2):
            if self.state[i] > 0:
                self.area += '+'
            else:
                self.area += '-'

    def _move(self, state, action, door):
        """Move inside [0,SN]*[0,SN]"""

        # calculate action ratio t
        mi = np.zeros(2, dtype=np.float32)
        for i in range(2):
            if action[i] > 0.0:
                mi[i] = (SN - state[i]) / action[i]
            elif action[i] < 0.0:
                mi[i] = -state[i] / action[i]
            else:
                mi[i] = 2.0

        # find min ratio
        o, t = 2, 1.0
        for i in range(2):
            if mi[i] < t:
                o, t = i, mi[i]

        if o == 2:
            return state + action
        else:
            rstate = state + t * action
            if action[o] > 0.0:
                # through the door
                if door[o] == SN and DL <= rstate[1 - o] <= DH:
                    return state + action
                else:
                    action[o] = -action[o]
                    action *= (1 - t) * FRICTION
                    return self._move(rstate, action, door)
            else:
                # through the door
                if door[o] == 0.0 and DL <= rstate[1 - o] <= DH:
                    return state + action
                else:
                    action[o] = -action[o]
                    action *= (1 - t) * FRICTION
                    return self._move(rstate, action, door)

    def _get_distance(self):
        """Get the available distance between current state and target"""

        st = (self.area, self.target_area)
        if self.area == self.target_area:
            return norm(self.state - self.target)
        elif st == ('++', '-+') or st == ('-+', '++'):
            return norm(self.state - DOORS[0]) + norm(self.target - DOORS[0])
        elif st == ('+-', '--') or st == ('--', '+-'):
            return norm(self.state - DOORS[1]) + norm(self.target - DOORS[1])
        elif st == ('++', '+-') or st == ('+-', '++'):
            return norm(self.state - DOORS[2]) + norm(self.target - DOORS[2])
        elif st == ('--', '-+') or st == ('-+', '--'):
            return norm(self.state - DOORS[3]) + norm(self.target - DOORS[3])
        elif st == ('++', '--'):
            return min(norm(self.state - DOORS[0]) + norm(self.target - DOORS[3]),
                       norm(self.state - DOORS[2]) + norm(self.target - DOORS[1])) + SDM
        elif st == ('--', '++'):
            return min(norm(self.state - DOORS[3]) + norm(self.target - DOORS[0]),
                       norm(self.state - DOORS[1]) + norm(self.target - DOORS[2])) + SDM
        elif st == ('-+', '+-'):
            return min(norm(self.state - DOORS[0]) + norm(self.target - DOORS[2]),
                       norm(self.state - DOORS[3]) + norm(self.target - DOORS[1])) + SDM
        elif st == ('+-', '-+'):
            return min(norm(self.state - DOORS[2]) + norm(self.target - DOORS[0]),
                       norm(self.state - DOORS[1]) + norm(self.target - DOORS[3])) + SDM


# register(
#     id='Fourrooms-v1',
#     entry_point='continus_four_room:FourRoom',
#     max_episode_steps=200,
#     reward_threshold=1.0,
# )
