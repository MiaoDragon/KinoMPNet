from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')

from sparse_rrt.planners import SST
#from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd
from plan_utility.line_line_cc import line_line_cc

import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.patches as patches
#from IPython.display import HTML

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
        self.dt = 2
        self.fig = plt.gcf()
        self.fig.set_figheight(4)
        self.fig.set_figwidth(8)
        self.ax1 = plt.subplot(121)
        self.ax2 = plt.subplot(122)

    def _init(self):
        ##### handle the animation
        # clear the current ax
        print("in init")
        ax = self.ax1
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

        #### handle search space
        ax = self.ax2
        ax.clear()
        ax.set_xlim(-30, 30)
        ax.set_ylim(-np.pi, np.pi)

        dx = 1
        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*30./dx)
        jmin = 0
        jmax = int(2*np.pi/dtheta)

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                x = np.array([dx*i-30, 0., dtheta*j-np.pi, 0.])
                if IsInCollision(x, self.cc_obs):
                    infeasible_points.append(x)
                else:
                    feasible_points.append(x)
        feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        
        print('feasible points')
        print(feasible_points)
        print('infeasible points')
        print(infeasible_points)
        scat_feas =ax.scatter(feasible_points[:,0], feasible_points[:,2], c='yellow')
        scat_infeas = ax.scatter(infeasible_points[:,0], infeasible_points[:,2], c='pink')

        self.recs.append(scat_feas)
        self.recs.append(scat_infeas)

        scat_state = ax.scatter(state[0], state[2], c='green')
        self.recs.append(scat_state)
        print("after init")

        return self.recs
    def _animate(self, i):
        print('animating, frame %d/%d' % (i, self.total))
        
        ax = self.ax1
        ax.set_xlim(-40, 40)
        ax.set_ylim(-20, 20)
        state = self.states[i]
        self.recs[0].set_xy((state[0]-self.params['pole_w']/2,self.params['cart_h']))
        t = mpl.transforms.Affine2D().rotate_deg_around(state[0], self.params['cart_h'], \
                                                        -state[2]/np.pi * 180) + ax.transData
        self.recs[0].set_transform(t)
        self.recs[1].set_xy((state[0]-self.params['cart_w']/2,0))


        # handle search space
        ax = self.ax2
        ax.set_xlim(-30, 30)
        ax.set_ylim(-np.pi, np.pi)
        self.recs[-1].set_offsets([state[0], state[2]])
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
        self.cc_obs = obs_i

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
        self.total = len(self.states)
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
writer=animation.FFMpegFileWriter(fps=500)
for obs_idx in range(0,5):
    for p_idx in range(2):
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
