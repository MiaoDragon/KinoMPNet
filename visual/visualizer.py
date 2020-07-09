"""
Given a list of states, render the environment
"""
from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate

#from IPython.display import HTML

class Visualizer():
    def __init__(self, system, params):
        self.system = system
        self.params = params
        self.dt = 0.05
    def _init(self):
        pass
    def _animate(self, i):
        pass
    def animate(self, states, actions, obstacles):
        '''
        given a list of states, actions and obstacles, animate the robot
        '''
        pass
