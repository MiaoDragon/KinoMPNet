"""
Python version of CartPole without obstacles for visualization.
"""

import numpy as np
from sparse_rrt.systems.system_interface import BaseSystem

class CartPoleObs(BaseSystem):
    '''
    Python implementation of the CartPole environment with obstacles.
    '''
    # parameters involved in the environment
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5  # cart
    # define the name for each state index and action index
    STATE_X, STATE_V, STATE_THETA, STATE_W = 0, 1, 2, 3
    CONTROL_A = 0
    # define boundary
    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    # obstacle information
    OBS_W = 4

    def __init__(self, obstacle_list):
        '''
        :obstacle_list: numpy array describing the list of rectangle obstacles
        each obstacle is described by 2 parameters.
        (x,y): the position of the middle point of the obstacles
        assume the cart has 0 y axis.
        '''
        BaseSystem.__init__(self)
        #super(CartPoleObs, self).__init__()
        self.obs = obstacle_list
    def propagate(self, start_state, control, num_steps, integration_step):
        '''
        Integrate system dynamics
        :param start_state: numpy array with the start state for the integration
        :param control: numpy array with constant controls to be applied during integration
        :param num_steps: number of steps to integrate
        :param integration_step: dt of integration
        :return: new state of the system
        '''
        state = np.array(start_state)
        for i in range(num_steps):
            # simulate forward transition by first order integration
            deriv = self.update_derivative(state, control)
            # integrate to obtain next state
            state = state + integration_step*deriv
            state = self.enforce_bounds(state)
            if not self.valid_state(state):
                print('in collision')
                return None
        return state

    def enforce_bounds(self, state):
        '''
        check if state satisfies the bound
        apply threshold to velocity and angle
        return a new state toward which the bound has been enforced
        '''
        new_state = np.array(state)
        if state[self.STATE_V] < self.MIN_V:
            new_state[self.STATE_V] = self.MIN_V
        elif state[self.STATE_V] > self.MAX_V:
            new_state[self.STATE_V] = self.MAX_V

        if state[self.STATE_THETA] < -np.pi:
            new_state[self.STATE_THETA] += 2*np.pi
        elif state[self.STATE_THETA] > np.pi:
            new_state[self.STATE_THETA] -= 2*np.pi

        if state[self.STATE_W] < self.MIN_W:
            new_state[self.STATE_W] = self.MIN_W
        elif state[self.STATE_W] > self.MAX_W:
            new_state[self.STATE_W] = self.MAX_W
        return new_state

    def valid_state(self, state):
        '''
        Implements the collision checking function for the state.
        '''
        # check if within boundary
        #if state[self.STATE_X] < self.MIN_X or state[self.STATE_X] > self.MAX_X:
        #    return False
        # given the position of the middle point of the pole, use MPNet environment
        # rigidbody collision checker
        # since the cart has position (state[0], 0)
        # the end-point of the pole has position (state[0]+l*sin(theta), l*cos(theta))
        midpoint = np.array([state[self.STATE_X] + self.L * np.sin(state[self.STATE_THETA]), self.H + self.L * np.cos(state[self.STATE_THETA])])
        midpoint = midpoint / 2.
        midpoint = np.append(midpoint, state[self.STATE_THETA])  # need the orientation as well
        return True

    def update_derivative(self, state, control):
        '''
        implement the function x_dot = f(x,u)
        return the derivative w.r.t. x
        '''
        _v = state[self.STATE_V]
        _w = state[self.STATE_W]
        _theta = state[self.STATE_THETA]
        _a = control[self.CONTROL_A]
        mass_term = (self.M + self.m)*(self.I + self.m * self.L * self.L) - \
                self.m * self.m * self.L * self.L * np.cos(_theta) * np.cos(_theta)

        deriv = np.zeros(4)  # init derivative
        deriv[self.STATE_X] = _v
        deriv[self.STATE_THETA] = _w
        mass_term = (1.0 / mass_term)
        deriv[self.STATE_V] = ((self.I + self.m * self.L * self.L)* \
            (_a + self.m * self.L * _w * _w * np.sin(_theta)) + \
            self.m * self.m * self.L * self.L * np.cos(_theta) * np.sin(_theta) * self.g) * mass_term
        deriv[self.STATE_W] = ((-self.m * self.L * np.cos(_theta)) * \
            (_a + self.m * self.L * _w * _w * np.sin(_theta))+(self.M + self.m) * \
            (-self.m * self.g * self.L * np.sin(_theta))) * mass_term
        return deriv

    def visualize_point(self, state):
        '''
        Project state space point to 2d visualization plane
        :param state: numpy array of the state point
        :return: x, y of visualization coordinates for this state point
        '''
        x = (state[0] - self.MIN_X) / (self.MAX_X - self.MIN_X)
        y = (state[1] - self.MIN_Y) / (self.MAX_Y - self.MIN_Y)
        return x, y

    def get_state_bounds(self):
        '''
        Return bounds for the state space
        :return: list of (min, max) bounds for each coordinate in the state space
        '''
        return [(self.MIN_X, self.MAX_X),
                (self.MIN_V, self.MAX_V),
                (-np.pi, np.pi),
                (self.MIN_W, self.MAX_W)]

    def get_control_bounds(self):
        '''
        Return bounds for the control space
        :return: list of (min, max) bounds for each coordinate in the control space
        '''
        return [(-300., 300.)]

    def is_circular_topology(self):
        '''
        Indicate whether state system has planar or circular topology
        :return: boolean flag for each coordinate (False for planar topology)
        '''
        return [False, False, True, False]
