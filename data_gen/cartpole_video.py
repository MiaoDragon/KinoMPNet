from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')

from sparse_rrt.planners import SST
f#rom env.cartpole_obs import CartPoleObs
f#rom env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd

import pickle
import os


# visualize the path
"""
Given a list of states, render the environment
"""
from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.patches as patches
from visual.visualizer import Visualizer




def IsInCollision(x, obc, obc_width=4.):
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2


    if x[0] < MIN_X or x[0] > MAX_X:
        return True

    H = 0.5
    pole_x1 = x[0]
    pole_y1 = H
    pole_x2 = x[0] + L * np.sin(x[2])
    pole_y2 = H + L * np.cos(x[2])


    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False




class CartPoleVisualizer(Visualizer):
    def __init__(self, system, params):
        super(CartPoleVisualizer, self).__init__(system, params)
        self.dt = 0.05

    def _init(self):
        # clear the current ax
        ax = plt.gca()
        ax.clear()
        # add patches
        state = self.states[0]
        self.pole = patches.Rectangle((state[0]-self.params['pole_w']/2,self.params['cart_h']),\
                                       self.params['pole_w'],self.params['pole_l'],\
                                      linewidth=.5,edgecolor='red',facecolor='red')
        self.cart = patches.Rectangle((state[0]-self.params['cart_w']/2,0),\
                                       self.params['cart_w'],self.params['cart_h'],\
                                      linewidth=.5,edgecolor='blue',facecolor='blue')
        self.recs = []
        self.recs.append(self.pole)
        self.recs.append(self.cart)
        for i in range(len(self.obs)):
            x, y = self.obs[i]
            obs = patches.Rectangle((x-self.params['obs_w']/2,y-params['obs_h']/2),\
                                       self.params['obs_w'],self.params['obs_h'],\
                                      linewidth=.5,edgecolor='black',facecolor='black')
            self.recs.append(obs)
            ax.add_patch(obs)
        # transform pole according to state
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.pole.set_transform(t)
        ax.add_patch(self.pole)
        ax.add_patch(self.cart)
        return self.recs
    def _animate(self, i):
        ax = plt.gca()
        ax.set_xlim(-40, 40)
        ax.set_ylim(-20, 20)
        state = self.states[i]
        self.recs[0].set_xy((state[0]-self.params['pole_w']/2,self.params['cart_h']))
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.recs[0].set_transform(t)
        self.recs[1].set_xy((state[0]-self.params['cart_w']/2,params['cart_h']))
        # print location of cart
        return self.recs



    def animate(self, states, actions, costs, obstacles):
        '''
        given a list of states, actions and obstacles, animate the robot
        '''

        new_obs_i = []
        obs_width = 4.0
        for k in range(len(obstacles)):
            obs_pt = []
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            obs_pt.append(obstacles[k][0]-obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]+obs_width/2)
            obs_pt.append(obstacles[k][0]+obs_width/2)
            obs_pt.append(obstacles[k][1]-obs_width/2)
            new_obs_i.append(obs_pt)
        obs_i = new_obs_i


        # transform the waypoint states and actions into trajectory
        traj = []
        s = states[0]
        for i in range(len(states)-1):
            print('state: %d, remaining: %d' % (i, len(states)-i))
            action = actions[i]
            # number of steps for propagtion
            num_steps = int(np.round(costs[i]/self.params['integration_step']))

            for j in range(num_steps):
                traj.append(np.array(s))
                #print("porpagating...")
                #print(s)
                #print('st:')
                #print(sT)
                s = self.system(s, action, self.params['integration_step'])
                assert not IsInCollision(s, obs_i)


        traj = np.array(traj)
        print("animating...")
        # animate
        self.states = traj
        self.obs = obstacles
        print(len(self.states))
        ani = animation.FuncAnimation(plt.gcf(), self._animate, range(0, len(self.states)),
                                      interval=self.dt, blit=True, init_func=self._init,
                                      repeat=True)
        return ani





obs_list = []
width = 4.
near = width * 1.2
H = 0.5
L = 2.5

# convert from obs to point cloud
# load generated point cloud
writer=animation.FFMpegFileWriter(fps=50)
for obs_idx in range(5):
    for p_idx in range(10):
        # Create custom system
        #obs_list = [[-10., -3.],
        #            [0., 3.],
        #            [10, -3.]]
        if os.path.exists('../cartpole_env%d_path%d.mp4' % (obs_idx, p_idx)):
            continue
        file = open('../data/cartpole_obs/obs_%d.pkl' % (obs_idx), 'rb')
        obs_list = pickle.load(file)
        file = open('../data/cartpole_obs/obc_%d.pkl' % (obs_idx), 'rb')
        obc_list = pickle.load(file)
        print('generated.')
        print(obs_list.shape)
        # load path
        path = open('../data/cartpole_obs/%d/path_%d.pkl' % (obs_idx, p_idx), 'rb')
        path = pickle.load(path)
        controls = open('../data/cartpole_obs/%d/control_%d.pkl' % (obs_idx, p_idx), 'rb')
        controls = pickle.load(controls)
        costs = open('../data/cartpole_obs/%d/cost_%d.pkl' % (obs_idx, p_idx), 'rb')
        costs = pickle.load(costs)
        params = {}
        params['pole_l'] = 2.5
        params['pole_w'] = 0.1
        params['cart_w'] = 1.
        params['cart_h'] = 0.5
        params['obs_w'] = 4
        params['obs_h'] = 4
        params['integration_step'] = 0.002
        #system = CartPole(obs_list)
        system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)

        vis = CartPoleVisualizer(dynamics, params)
        states = path
        actions = controls
        anim = vis.animate(np.array(states), np.array(actions), np.array(costs), obs_list)
        #HTML(anim.to_html5_video())
        anim.save('../cartpole_env%d_path%d.mp4' % (obs_idx, p_idx), writer=writer)
