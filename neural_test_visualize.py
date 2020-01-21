'''
This is the main file to run gem_end2end network.
It simulates the real scenario of observing a data, puts it inside the memory (or not),
and trains the network using the data
after training at each step, it will output the R matrix described in the paper
https://arxiv.org/abs/1706.08840
and after sevral training steps, it needs to store the parameter in case emergency
happens
To make it work in a real-world scenario, it needs to listen to the observer at anytime,
and call the network to train if a new data is available
(this thus needs to use multi-process)
here for simplicity, we just use single-process to simulate this scenario
'''
from __future__ import print_function
import sys
sys.path.append('deps/sparse_rrt')

import model.AE.identity as cae_identity
from model.AE import CAE_acrobot_voxel_2d
from model import mlp, mlp_acrobot
#from model.mlp import MLP
from model.mpnet import KMPNet
import numpy as np
import argparse
import os
import torch
from gem_eval import eval_tasks
from torch.autograd import Variable
import copy
import os
import gc
import random
from tools.utility import *
from plan_utility import pendulum, acrobot_obs
#from sparse_rrt.systems import standard_cpp_systems
#from sparse_rrt import _sst_module
from tools import data_loader
import jax
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import torch.nn as nn
from torch.autograd import Variable
import math
import time
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
from plan_utility.informed_path import *

#import matplotlib.pyplot as plt
#fig = plt.figure()

import sys
sys.path.append('..')

import numpy as np
#from tvlqr.python_tvlqr import tvlqr
#from tvlqr.python_lyapunov import sample_tv_verify
from plan_utility.data_structure import *
def propagate(x, us, dts, dynamics, enforce_bounds, step_sz=None):
    # use the dynamics to interpolate the state x
    # can implement different interpolation method for this
    new_xs = [x]
    new_us = []
    new_dts = []
    for i in range(len(us)):
        dt = dts[i]
        u = us[i]
        num_steps = int(dt / step_sz)
        last_step = dt - num_steps*step_sz
        for k in range(num_steps):
            x = x + step_sz*dynamics(x, u)
            x = enforce_bounds(x)
            new_xs.append(x)
            new_us.append(u)
            new_dts.append(step_sz)
        x = x + last_step*dynamics(x, u)
        x = enforce_bounds(x)
        new_xs.append(x)
        new_us.append(u)
        new_dts.append(last_step)
    new_xs = np.array(new_xs)
    new_us = np.array(new_us)
    new_dts = np.array(new_dts)
    return new_xs, new_us, new_dts
def traj_opt(x0, x1, solver):
    # use trajectory optimization method to compute trajectory between x0 and x1
    # load the dynamics function corresponding to the envname
    xs, us, ts = solver.solve(x0, x1)
    return xs, us, ts

def pathSteerTo(x0, x1, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction, step_sz=0.002, compute_funnel=True):
    # direciton 0 means forward from x0 to x1
    # direciton 1 means backward from x0 to x1
    # jac_A: given x, u -> linearization A
    # jac_B: given x, u -> linearization B
    # traj_opt: a function given two endpoints x0, x1, compute the optimal trajectory
    if direction == 0:
        xs, us, dts = traj_opt(x0.x, x1.x)
        """
        print('----------------forward----------------')
        print('trajectory opt:')
        print('start:')
        print(x0.x)
        print('end:')
        print(x1.x)
        print('xs[0]:')
        print(xs[0])
        print('xs[-1]:')
        print(xs[-1])
        print('us:')
        print(us)
        print('dts:')
        print(dts)
        """
        # ensure us and dts have length 1 less than xs
        if len(us) == len(xs):
            us = us[:-1]
        xs, us, dts = propagate(x0.x, us, dts, dynamics=dynamics, enforce_bounds=enforce_bounds, step_sz=step_sz)
        """
            print('propagation result:')
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        edge_dt = np.sum(dts)
        start = x0
        goal = Node(xs[-1])
        x1 = goal
    else:
        xs, us, dts = traj_opt(x1.x, x0.x)
        """
        print('----------------backward----------------')
        print('trajectory opt:')
        print('start:')
        print(x1.x)
        print('end:')
        print(x0.x)
        print('xs[0]:')
        print(xs[0])
        print('xs[-1]:')
        print(xs[-1])
        print('us:')
        print(us)
        print('dts:')
        print(dts)
        """
        if len(us) == len(xs):
            us = us[:-1]
        us = np.flip(us, axis=0)
        dts = np.flip(dts, axis=0)
        # reversely propagate the system
        xs, us, dts = propagate(x0.x, us, dts, dynamics=lambda x, u: -dynamics(x, u), enforce_bounds=enforce_bounds, step_sz=step_sz)
        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)
        dts = np.flip(dts, axis=0)
        """
            print('propagation result:')
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        edge_dt = np.sum(dts)
        start = Node(xs[0])  # after flipping, the first in xs is the start
        goal = x0
        x1 = start


    # after trajopt, make actions of dimension 2
    us = us.reshape(len(us), -1)

    # notice that controller time starts from 0, hence locally need to shift the time by minusing t0_edges
    # start from 0
    time_knot = np.cumsum(dts)
    time_knot = np.insert(time_knot, 0, 0.)

    # can also change the resolution by the following function (for instance, every 10)
    #indices = np.arange(0, len(time_knot), 10)
    #time_knot = time_knot[indices]
    #print(time_knot)

    edge = Edge(xs, us, dts, time_knot, edge_dt)
    edge.next = goal
    start.edge = edge
    start.next = goal
    goal.prev = start
    if goal.S0 is None or not compute_funnel:
        return x1, edge
    # the values to return: new node, new edge
    res_x = x1
    res_edge = edge    
    # if the upper is defined, then we can backpropagate the tvlqr and funnel computation    
    while start is not None:
        # assuming we haven't computed tvlqr for start->goal
        edge = start.edge
        xs = edge.xs
        us = edge.us
        dts = edge.dts
        Qf = goal.S0
        if Qf is not None:
            Qf = np.array(Qf)
        controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B, Qf=Qf)
        start.S0 = S(0).reshape((len(start.x),len(start.x)))
        edge.xtraj = xtraj
        edge.utraj = utraj
        edge.S = S
        edge.controller = controller

        # otherwise, recursively backpropagate the funnel computation
        #print("xs (which is used to construct xtraj):")
        #print(xs)
        #print('xtraj.x:')
        #print(xtraj.x)        
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        #print('time_knot:')
        #print(time_knot)
        #print('goal.x:')
        #print(goal.x)
        #print('xtraj(last_time):')
        #print(xtraj(time_knot[-1]))
        S = start.edge.S
        print('time_knot: %d' % (len(time_knot)))
        #todo: to add rho0s and rho1s list to edge
        # reversely construct the funnel
        rho0s = []
        rho1s = []
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            #if i == len(time_knot)-1:
            #    x1 = goal.x
            u1 = utraj(t1)
            A0 = jac_A(x0, u0)
            B0 = jac_B(x0, u0)
            A1 = jac_A(x1, u1)
            B1 = jac_B(x1, u1)
            A0 = np.asarray(A0)
            B0 = np.asarray(B0)
            A1 = np.asarray(A1)
            B1 = np.asarray(B1)
            S0 = S(t0).reshape(len(x0),len(x0))
            S1 = S(t1).reshape(len(x0),len(x0))
            Q = np.identity(len(x0))
            R = np.identity(len(u0))
            ##TODO: check the output of sample_tv_verify
            #print('verifying...')
            #print('x0:')
            #print(x0)
            #print('x1:')
            #print(x1)
            #print('upper_x')
            #print(upper_x)
            #print("S1:")
            #print(S1)
            #print("upper_S:")
            #print(upper_S)
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=100)
            rho0s.append(rho0)
            rho1s.append(rho1)
            upper_rho = rho0
            upper_x = x0
            upper_S = S0
            #tvlqr_rhos.append([rho0, rho1, t0, t1])
            if i == len(time_knot)-1:
                # the endpoint
                start.edge.rho1 = rho1
                goal.rho1 = rho1
                goal.S1 = S1
        start.edge.rho0 = rho0
        rho0s.reverse()
        rho1s.reverse()
        start.edge.rho0s = rho0s
        start.edge.rho1s = rho1s
        start.rho0 = rho0
        start = start.prev
        goal = goal.prev
    return res_x, res_edge



def lazyFunnel(xg, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz=0.02):
    # compute funnel backward until xg
    # recursively backpropagate the funnel computation
    start = xG.prev
    goal = xG
    while start is not None:
        # already at xg
        if xg.prev is not None and np.linalg.norm(xg.prev.x - start.x) <= 1e-6:
            # xg already computed
            break

        # assuming we haven't computed tvlqr for start->goal
        edge = start.edge
        xs = edge.xs
        us = edge.us
        dts = edge.dts
        Qf = goal.S0
        if Qf is not None:
            Qf = np.array(Qf)
        controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B, Qf=Qf)
        start.S0 = S(0).reshape((len(start.x),len(start.x)))
        edge.xtraj = xtraj
        edge.utraj = utraj
        edge.S = S
        edge.controller = controller

        # otherwise, recursively backpropagate the funnel computation
        #print("xs (which is used to construct xtraj):")
        #print(xs)
        #print('xtraj.x:')
        #print(xtraj.x)        
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        #print('time_knot:')
        #print(time_knot)
        #print('goal.x:')
        #print(goal.x)
        #print('xtraj(last_time):')
        #print(xtraj(time_knot[-1]))
        S = start.edge.S
        print('time_knot: %d' % (len(time_knot)))
        #todo: to add rho0s and rho1s list to edge
        rho0s = []
        rho1s = []
        # reversely construct the funnel
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            #if i == len(time_knot)-1:
            #    x1 = goal.x
            u1 = utraj(t1)
            A0 = jac_A(x0, u0)
            B0 = jac_B(x0, u0)
            A1 = jac_A(x1, u1)
            B1 = jac_B(x1, u1)
            A0 = np.asarray(A0)
            B0 = np.asarray(B0)
            A1 = np.asarray(A1)
            B1 = np.asarray(B1)
            S0 = S(t0).reshape(len(x0),len(x0))
            S1 = S(t1).reshape(len(x0),len(x0))
            Q = np.identity(len(x0))
            R = np.identity(len(u0))
            ##TODO: check the output of sample_tv_verify
            #print('verifying...')
            #print('x0:')
            #print(x0)
            #print('x1:')
            #print(x1)
            #print('upper_x')
            #print(upper_x)
            #print("S1:")
            #print(S1)
            #print("upper_S:")
            #print(upper_S)
            print('upper rho: %f' % (upper_rho))
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=100)
            print('rho0: %f' % (rho0))
            print('rho1: %f' % (rho1))
            upper_rho = rho0
            upper_x = x0
            upper_S = S0
            rho0s.append(rho0)
            rho1s.append(rho1)
            #tvlqr_rhos.append([rho0, rho1, t0, t1])
            if i == len(time_knot)-1:
                # the endpoint
                start.edge.rho1 = rho1
                goal.rho1 = rho1
                goal.S1 = S1
        start.edge.rho0 = rho0
        rho0s.reverse()
        rho1s.reverse()
        start.edge.rho0s = rho0s
        start.edge.rho1s = rho1s
        start.rho0 = rho0
        start = start.prev
        goal = goal.prev


def funnelSteerTo(x0, x1, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direciton, step_sz=0.02):
    start = x0
    goal = x1
    if direciton == 0:
        start = x0
        goal = x1
    else:
        start = x1
        goal = x0
    # recursively backpropagate the funnel computation
    while start is not None:
        # assuming we haven't computed tvlqr for start->goal
        edge = start.edge
        xs = edge.xs
        us = edge.us
        dts = edge.dts
        Qf = goal.S0
        if Qf is not None:
            Qf = np.array(Qf)
        controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B, Qf=Qf)
        start.S0 = S(0).reshape((len(start.x),len(start.x)))
        edge.xtraj = xtraj
        edge.utraj = utraj
        edge.S = S
        edge.controller = controller

        # otherwise, recursively backpropagate the funnel computation
        #print("xs (which is used to construct xtraj):")
        #print(xs)
        #print('xtraj.x:')
        #print(xtraj.x)        
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        #print('time_knot:')
        #print(time_knot)
        #print('goal.x:')
        #print(goal.x)
        #print('xtraj(last_time):')
        #print(xtraj(time_knot[-1]))
        S = start.edge.S
        print('time_knot: %d' % (len(time_knot)))
        #todo: to add rho0s and rho1s list to edge
        # reversely construct the funnel
        rho0s = []
        rho1s = []
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            #if i == len(time_knot)-1:
            #    x1 = goal.x
            u1 = utraj(t1)
            A0 = jac_A(x0, u0)
            B0 = jac_B(x0, u0)
            A1 = jac_A(x1, u1)
            B1 = jac_B(x1, u1)
            A0 = np.asarray(A0)
            B0 = np.asarray(B0)
            A1 = np.asarray(A1)
            B1 = np.asarray(B1)
            S0 = S(t0).reshape(len(x0),len(x0))
            S1 = S(t1).reshape(len(x0),len(x0))
            Q = np.identity(len(x0))
            R = np.identity(len(u0))
            ##TODO: check the output of sample_tv_verify
            #print('verifying...')
            #print('x0:')
            #print(x0)
            #print('x1:')
            #print(x1)
            #print('upper_x')
            #print(upper_x)
            #print("S1:")
            #print(S1)
            #print("upper_S:")
            #print(upper_S)
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=100)
            rho0s.append(rho0)
            rho1s.append(rho1)
            upper_rho = rho0
            upper_x = x0
            upper_S = S0
            #tvlqr_rhos.append([rho0, rho1, t0, t1])
            if i == len(time_knot)-1:
                # the endpoint
                start.edge.rho1 = rho1
                goal.rho1 = rho1
                goal.S1 = S1
        start.edge.rho0 = rho0
        rho0s.reverse()
        rho1s.reverse()
        start.edge.rho0s = rho0s
        start.edge.rho1s = rho1s
        start.rho0 = rho0
        start = start.prev
        goal = goal.prev


        
def node_nearby(x0, x1, S, rho, system):
    # state x0 to state x1
    delta_x = x0 - x1
    circular = system.is_circular_topology()
    for i in range(len(delta_x)):
        if circular[i]:
            # if it is angle
            if delta_x[i] > np.pi:
                delta_x[i] = delta_x[i] - 2*np.pi
            if delta_x[i] < -np.pi:
                delta_x[i] = delta_x[i] + 2*np.pi

    xTSx = delta_x.T@S@delta_x
    if xTSx <= 4.:
        print('nearby:')
        print('S:')
        print(S)
        print('xTSx: %f' % (xTSx))
        print('rho^2: %f' % (rho*rho))
    if xTSx <= rho*rho:
         return True
    return False

def line_nearby(x0, x1, system):
    # state x0 to line starting from node x1
    e = x1.edge
    xs = e.xs
    us = e.us
    ts = e.time_knot
    for k in range(len(ts)-1):
        S = e.S(ts[k]).reshape((len(x0), len(x0)))
        rho = e.rho0s[k]
        if node_nearby(x0, xs[k], S, rho, system):
            return True, k
    return False, 0

def nearby(x0, x1, system):
    # using the S and rho stored by the node to determine distance
    # if x0 lies in x1, and within the boundary of x1 (S, rho0)
    # notice that for circulating state, needs to map the angle
    # if edge is defined on xG, then use edge
    if x0.edge is not None:
        e = x0.edge
        xs = e.xs
        us = e.us
        ts = e.time_knot
        for k in range(len(ts)-1):
            if x1.edge is not None:
                line_near, k1 = line_nearby(xs[k], x1, system)
                if line_near:
                    # near the line, with node index k1
                    return True, k, k1
            else:
                if node_nearby(xs[k], x1.x, x1.S0, x1.rho0, system):
                    return True, k, 0
    else:
        if x1.edge is not None:
            line_near, k1 = line_nearby(x0.x, x1, system)
            if line_near:
                return True, 0, k1
        else:
            if node_nearby(x0.x, x1.x, x1.S0, x1.rho0, system):
                return True, 0, 0
    return False, 0, 0

def node_h_nearby(x0, x1, S, rho, system):
    # given two nodes, S and rho, check if x0 is near x1
    delta_x = x0 - x1
    # this is pendulum specific. For other envs, need to do similar things
    circular = system.is_circular_topology()
    for i in range(len(delta_x)):
        if circular[i]:
            # if it is angle
            if delta_x[i] > np.pi:
                delta_x[i] = delta_x[i] - 2*np.pi
            if delta_x[i] < -np.pi:
                delta_x[i] = delta_x[i] + 2*np.pi
    xTSx = delta_x.T@S@delta_x
    if xTSx <= 4.:
        print('delta_x:')
        print(delta_x)
        print('S:')
        print(S)
        print('xTSx: %f' % (xTSx))
        # notice that we define rho to be ||S^{1/2}x||
        print('rho^2: %f' % (rho*rho))
    if xTSx <= rho*rho:
        return True
    else:
        return False
def line_h_nearby(x0, xG, S, rho, system):
    # check state x0 against line starting from node xG
    e2 = xG.edge
    xs2 = e2.xs
    us2 = e2.us
    ts2 = e2.time_knot
    for k2 in range(len(ts2)-1):
        if e2.S is not None:
            S = e2.S(ts2[k2]).reshape((len(x0), len(x0)))
            rho = e2.rho0s[k2]
        elif xG.S0 is not None:
            S = xG.S0
            rho = xG.rho0
        if node_h_nearby(x0, xs2[k2], S, rho, system):
            return True
    return False
def h_nearby(node, xG, S, rho, system):
    # check if the edge starting from node and the edge starting from xG has intersection
    if node.edge is not None:
        e1 = node.edge
        xs1 = e1.xs
        us1 = e1.us
        ts1 = e1.time_knot
        for k1 in range(len(ts1)-1):
            if xG.edge is not None:
                if line_h_nearby(xs1[k1], xG, S, rho, system):
                    return True
            else:
                if xG.S0 is not None:
                    S = xG.S0
                    rho = xG.rho0
                if node_h_nearby(xs1[k1], xG.x, S, rho, system):
                    return True
    else:
        if xG.edge is not None:
            if line_h_nearby(node.x, xG, S, rho, system):
                return True
        else:
            if xG.S0 is not None:
                S = xG.S0
                rho = xG.rho0
            if node_h_nearby(node.x, xG.x, S, rho, system):
                return True            
    return False
                
                
                
                
def plan(env, x0, xG, informer, system, dynamics, enforce_bounds, traj_opt, jac_A, jac_B, data, step_sz=0.02, MAX_LENGTH=1000):
    # informer: given (xt, x_desired) ->  x_t+1
    # jac_A: given (x, u) -> linearization A
    # jac B: given (x, u) -> linearization B
    # traj_opt: given (x0, x1) -> (xs, us, dts)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    #hl_real, = ax.plot([], [], 'r')
    hl_for, = ax.plot([], [], 'g')
    hl_back, = ax.plot([], [], 'r')
    hl_for_mpnet, = ax.plot([], [], 'lightgreen')
    hl_back_mpnet, = ax.plot([], [], 'salmon')
    
    def update_line(h, ax, new_data):
        h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
        #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        #h.set_ydata(np.append(h.get_ydata(), new_data[1]))


    def draw_update_line(ax):
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.show()
    #update_line(hl, ax, x0.x)
    #draw_update_line(ax)
    for i in range(len(data)):
        update_line(hl, ax, data[i])
    draw_update_line(ax)
    update_line(hl_for, ax, x0.x)
    draw_update_line(ax)
    update_line(hl_back, ax, xG.x)
    draw_update_line(ax)
    plt.waitforbuttonpress()
    
        
    itr=0
    target_reached=0
    tree=0
    time_norm = 0.
    start = x0
    goal = xG
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        print('iter: %d' % (itr))
        if tree==0:
            print('forward')
            # since we ensure each step we can steer to the next waypoint
            # the edge connecting the two nodes will store the trajectory
            # information, TVLQR and the funnel size factors
            # the edge information is stored at the endpoint
            # here direciton=0 means we are computing forward steer, and 1 means
            # we are computing backward
            next_nodes = []
            for i in range(100):
                xnext = informer(env, x0, xG, direction=0)
                next_nodes.append(xnext.x)
            next_nodes = np.array(next_nodes)
            ax.scatter(next_nodes[:,0], next_nodes[:,1], color='lime', alpha=0.5)
            update_line(hl_for_mpnet, ax, xnext.x)
            draw_update_line(ax)
            x, e = pathSteerTo(x0, xnext, dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=0, compute_funnel=False)

            for i in range(len(e.xs)):
                update_line(hl_for, ax, e.xs[i])    
            ax.scatter(e.xs[:,0], e.xs[:,1], c='g')
            draw_update_line(ax)
            plt.waitforbuttonpress()
            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            tree=1

            node = xG
            
            while node is not None:
                #target_reached = False
                # check if it is near the goal using goal heuristics, if so, then compute the real funnel distance
                h_reached = h_nearby(x0, node, goal.S0, goal.rho0, system)
                if not h_reached:
                    node = node.next
                    continue       
                # passed the heuristics test, compute funnel and check nearby
                # if funnel already computed then don't need to
                if node.S0 is None:
                    lazyFunnel(node, goal, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz)    
                    goal = node
                target_reached, node_i0, node_i1 = nearby(x0, node, system)
                if target_reached:
                    # if node_i0 and node_i1 are subsets of the line, then update the edge info
                    if x0.edge is not None:
                        # need to extract subset of x0.edge
                        edge = x0.edge
                        edge.xs = edge.xs[:node_i0]
                        edge.us = edge.us[:node_i0]
                        edge.dts = edge.dts[:node_i0]
                        edge.time_knot = edge.time_knot[:node_i0]
                        edge.dt = edge.time_knot[-1]
                        # add a new node
                        x0_next = Node(edge.xs[-1])
                        edge.next = x0_next
                    if x1.edge is not None:
                        # need to extract subset of x1.edge
                        # change the t0 of edge starting from node to be time_knot[node_i] (use subset of edge)
                        edge = node.edge
                        edge.t0 = node.edge.time_knot[node_i1]
                        edge.i0 = node_i1
                        # change the node to be xs[node_i], S0 to be S(time_knot[node_i]), rho0 to be rho0s[node_i]
                        new_node = Node(node.edge.xs[node_i1])
                        new_node.S0 = node.edge.S(edge.t0)
                        new_node.rho0 = node.edge.rho0s[node_i1]
                        new_node.edge = edge
                        new_node.next = node.next
                        node = new_node

                    xG = node                    
                    break
                node = node.next

        else:
            print('backward')
            next_nodes = []
            for i in range(100):
                xnext = informer(env, x0, xG, direction=0)
                next_nodes.append(xnext.x)
            next_nodes = np.array(next_nodes)
            ax.scatter(next_nodes[:,0], next_nodes[:,1], color='orange', alpha=0.5)
            update_line(hl_back_mpnet, ax, xnext.x)
            draw_update_line(ax)
            
            x, e = pathSteerTo(xG, informer(env, xG, x0, direction=1), dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=1, compute_funnel=False)
            for i in range(len(e.xs)-1, -1, -1):
                update_line(hl_back, ax, e.xs[i])
            ax.scatter(e.xs[:,0], e.xs[:,1], c='r')
            draw_update_line(ax)
            plt.waitforbuttonpress()
            x.next = xG
            xG.prev = x
            e.next = xG
            x.edge = e
            xG = x
            tree=0
            node = x0
            while node is not None:
                #target_reached = False
                # check if it is near the goal using goal heuristics, if so, then compute the real funnel distance
                h_reached = h_nearby(node, xG, goal.S0, goal.rho0, system)
                if not h_reached:
                    node = node.prev
                    continue

                # passed the heuristics test, compute funnel and check nearby
                if xG.S0 is None:
                    lazyFunnel(xG, goal, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz)
                    goal = xG  # update the last node that we have computed funnel

                target_reached, node_i0, node_i1 = nearby(node, xG, system)
                if target_reached:
                    # if node_i0 and node_i1 are subsets of the line, then update the edge info
                    if node.edge is not None:
                        # need to extract subset of x0.edge
                        edge = node.edge
                        edge.xs = edge.xs[:node_i0]
                        edge.us = edge.us[:node_i0]
                        edge.dts = edge.dts[:node_i0]
                        edge.time_knot = edge.time_knot[:node_i0]
                        edge.dt = edge.time_knot[-1]
                        # add a new node
                        node_next = Node(edge.xs[-1])
                        edge.next = node_next
                    if x1.edge is not None:
                        # need to extract subset of x1.edge
                        # change the t0 of edge starting from node to be time_knot[node_i] (use subset of edge)
                        edge = xG.edge
                        edge.t0 = xG.edge.time_knot[node_i1]
                        edge.i0 = node_i1
                        # change the node to be xs[node_i], S0 to be S(time_knot[node_i]), rho0 to be rho0s[node_i]
                        new_node = Node(xG.edge.xs[node_i1])
                        new_node.S0 = xG.edge.S(edge.t0)
                        new_node.rho0 = xG.edge.rho0s[node_i1]
                        new_node.edge = edge
                        new_node.next = xG.next
                        xG = new_node
                    x0 = node
                    break
                node = node.prev


        #xG_, e_ = pathSteerTo(x0, xG, dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=0, compute_funnel=False)
        # check if x0 can connect to one node in the backward tree directly, if so, no need to construct a controller from x0 to the node
        # version one: only check endpoint
        #target_reached = nearby(x0, xG)  # check the funnel if can connect
        # version two: new node in start tree: check all goal tree, and otherwise conversely

    if target_reached:
        print('target reached.')
        # it is near enough, so we connect in the node data structure from x0 to xG, although the endpoint of x0.edge
        # in state is still xG_
        x0 = x0.prev  # since the x0 can directly connect to xG, we only need to set the next state of the previous x to xG
        x0.next = xG  # update endpoint (or should I?)
        x0.edge.next = xG
        xG.prev = x0
        # connect the lsat node
        # construct the funnel later
        # connect from x0 to xG, the endpoint of x0 is xG_, but it is near xG
        print('before funnelsteerto')
        funnelSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direciton=0, step_sz=step_sz)
        print('after funnelsteerto')

        #xG_.next = xG
        #e_.next = xG
        #xG_.edge = e_
    else:
        x0.next = None
        x0.edge = None

    # construct a list of the path
    path_list = []
    node = start
    while node is not None:
        path_list.append(node.x)
        node = node.next
    if not target_reached:
        # xG is the first in the goal tree
        while xG is not None:
            path_list.append(xG.x)
            xG = xG.next
    return target_reached, path_list



def eval_tasks(mpNet0, mpNet1, env_type, test_data, save_dir, data_type, normalize_func = lambda x:x, unnormalize_func=lambda x: x, dynamics=None, jac_A=None, jac_B=None, enforce_bounds=None):
    DEFAULT_STEP=0.02
    # data_type: seen or unseen
    obc, obs, paths, path_lengths = test_data
    if obs is not None:
        obc = obc.astype(np.float32)
        obc = torch.from_numpy(obc)
    if torch.cuda.is_available():
        obc = obc.cuda()
    def informer(env, x0, xG, direction):
        x0 = torch.from_numpy(x0.x).type(torch.FloatTensor)
        xG = torch.from_numpy(xG.x).type(torch.FloatTensor)
        if torch.cuda.is_available():
            x0 = x0.cuda()
            xG = xG.cuda()
        if direction == 0:
            x = torch.cat([x0,xG], dim=0)
            mpNet = mpNet0
        else:
            x = torch.cat([xG,x0], dim=0)
            mpNet = mpNet1
        if torch.cuda.is_available():
            x = x.cuda()
        res = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data.numpy()[0]
        res = Node(res)
        return res

    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        #print(obs, flush=True)

        # feasible paths for each env
        if env_type == 'pendulum':
            system = standard_cpp_systems.PSOPTPendulum()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 20, step_sz)
        elif env_type == 'cartpole_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
            system = _sst_module.CartPole()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, step_sz)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0
        elif env_type == 'acrobot_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            system = _sst_module.PSOPTAcrobot()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.02
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 10, 1, 5, step_sz)
            goal_S0 = np.identity(4)
            goal_S0[2,2] = 0.
            goal_S0[3,3] = 0.
            goal_rho0 = 0.1

        for j in range(len(paths[0])):
            time0 = time.time()
            time_norm = 0.
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            p1_ind=0
            p2_ind=0
            p_ind=0
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>0:
                fp = 0
                valid_path.append(1)
                #paths[i][j][0][1] = 0.
                #paths[i][j][path_lengths[i][j]-1][1] = 0.
                path = [paths[i][j][0], paths[i][j][path_lengths[i][j]-1]]
                # plot the entire path
                #plt.plot(paths[i][j][:,0], paths[i][j][:,1])

                start = Node(path[0])
                goal = Node(path[-1])
                print('start:')
                print(path[0])
                print('goal:')
                print(path[-1])
                goal.S0 = goal_S0
                goal.rho0 = goal_rho0    # change this later

                control = []
                time_step = []
                step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 11
                if obs is None:
                    obs_i = None
                    obc_i = None
                else:
                    obs_i = obs[i]
                    obc_i = obc[i]
                for t in range(MAX_NEURAL_REPLAN):
                    # adaptive step size on replanning attempts
                    res, path_list = plan(obc[i], start, goal, informer, system, dynamics, \
                               enforce_bounds, traj_opt, jac_A, jac_B, paths[i][j], step_sz=step_sz, MAX_LENGTH=1000)
                    #print('after neural replan:')
                    #print(path)
                    #path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    #print('after lvc:')
                    #print(path)
                    if res:
                        fp = 1
                        print('feasible ok!')
                        break
                    #if feasibility_check(bvp_solver, path, obc_i, IsInCollision, step_sz=0.01):
                    #    fp = 1
                    #    print('feasible, ok!')
                    #    break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time1 -= time_norm
                time_path.append(time1)
                print('test time: %f' % (time1))
                # write the path
                #print('planned path:')
                #print(path)
                #path = np.array(path)
                #np.savetxt('results/path_%d.txt' % (j), path)
                #np.savetxt('results/control_%d.txt' % (j), np.array(control))
                #np.savetxt('results/timestep_%d.txt' % (j), np.array(time_step))

            fes_path.append(fp)

        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (float(np.sum(fes_env)) / np.sum(valid_env)))
        time_path = save_dir + 'mpnet_%s_time.pkl' % (data_type)
        pickle.dump(time_env, open(time_path, "wb" ))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)





def main(args):
    # set seed
    print(args.model_path)
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 'pendulum':
        IsInCollision =pendulum.IsInCollision
        normalize = pendulum.normalize
        unnormalize = pendulum.unnormalize
        obs_file = None
        obc_file = None
        dynamics = pendulum.dynamics
        jax_dynamics = pendulum.jax_dynamics
        enforce_bounds = pendulum.enforce_bounds
        cae = cae_identity
        mlp = MLP
        obs_f = False
        #system = standard_cpp_systems.PSOPTPendulum()
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
    elif args.env_type == 'cartpole_obs':
        IsInCollision =cartpole.IsInCollision
        normalize = cartpole.normalize
        unnormalize = cartpole.unnormalize
        obs_file = None
        obc_file = None
        dynamics = cartpole.dynamics
        jax_dynamics = cartpole.jax_dynamics
        enforce_bounds = cartpole.enforce_bounds
        cae = CAE_acrobot_voxel_2d
        mlp = mlp_acrobot.MLP
        obs_f = True
        #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'cartpole')
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
    elif args.env_type == 'acrobot_obs':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        dynamics = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d
        mlp = mlp_acrobot.MLP
        obs_f = True
        #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)

    jac_A = jax.jacfwd(jax_dynamics, argnums=0)
    jac_B = jax.jacfwd(jax_dynamics, argnums=1)
    mpNet0 = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp)
    mpNet1 = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp)
    # load previously trained model if start epoch > 0
    model_path='kmpnet_epoch_%d_direction_0.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet0, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet0.cuda()
        mpNet0.mlp.cuda()
        mpNet0.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet0.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet0.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet0.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
    if args.start_epoch > 0:
        load_opt_state(mpNet0, os.path.join(args.model_path, model_path))
    # load previously trained model if start epoch > 0
    model_path='kmpnet_epoch_%d_direction_1.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet1, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet1.cuda()
        mpNet1.mlp.cuda()
        mpNet1.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet1.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet1.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet1.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
    if args.start_epoch > 0:
        load_opt_state(mpNet1, os.path.join(args.model_path, model_path))


    # load data
    print('loading...')
    if args.seen_N > 0:
        seen_test_data = data_loader.load_test_dataset(args.seen_N, args.seen_NP,
                                  args.data_folder, obs_f, args.seen_s, args.seen_sp)
    if args.unseen_N > 0:
        unseen_test_data = data_loader.load_test_dataset(args.unseen_N, args.unseen_NP,
                                  args.data_folder, obs_f, args.unseen_s, args.unseen_sp)
    # test
    # testing


    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.
    T = 1
    for _ in range(T):
        # unnormalize function
        normalize_func=lambda x: normalize(x, args.world_size)
        unnormalize_func=lambda x: unnormalize(x, args.world_size)
        # seen
        if args.seen_N > 0:
            time_file = os.path.join(args.model_path,'time_seen_epoch_%d_mlp.p' % (args.start_epoch))
            fes_path_, valid_path_ = eval_tasks(mpNet0, mpNet1, args.env_type, seen_test_data, args.model_path, 'seen', normalize_func, unnormalize_func, dynamics, jac_A, jac_B, enforce_bounds)
            valid_path = valid_path_.flatten()
            fes_path = fes_path_.flatten()   # notice different environments are involved
            seen_test_suc_rate += fes_path.sum() / valid_path.sum()
        # unseen
        if args.unseen_N > 0:
            time_file = os.path.join(args.model_path,'time_unseen_epoch_%d_mlp.p' % (args.start_epoch))
            fes_path_, valid_path_ = eval_tasks(mpNet0, mpNet1, args.env_type, unseen_test_data, args.model_path, 'unseen', normalize_func, unnormalize_func, dynamics, jac_A, jac_B, enforce_bounds)
            valid_path = valid_path_.flatten()
            fes_path = fes_path_.flatten()   # notice different environments are involved
            unseen_test_suc_rate += fes_path.sum() / valid_path.sum()
    if args.seen_N > 0:
        seen_test_suc_rate = seen_test_suc_rate / T
        f = open(os.path.join(args.model_path,'seen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
        f.write(str(seen_test_suc_rate))
        f.close()
    if args.unseen_N > 0:
        unseen_test_suc_rate = unseen_test_suc_rate / T    # Save the models
        f = open(os.path.join(args.model_path,'unseen_accuracy_epoch_%d.txt' % (args.start_epoch)), 'w')
        f.write(str(unseen_test_suc_rate))
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_lr005_SGD/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=1)
    parser.add_argument('--seen_NP', type=int, default=10)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=0)
    parser.add_argument('--unseen_N', type=int, default=0)
    parser.add_argument('--unseen_NP', type=int, default=0)
    parser.add_argument('--unseen_s', type=int, default=0)
    parser.add_argument('--unseen_sp', type=int, default=0)
    parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=4, help='dimension of total input')
    parser.add_argument('--AE_input_size', nargs='+', type=int, default=32, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=136, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=4, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_folder', type=str, default='./data/acrobot_obs/')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=99)
    parser.add_argument('--env_type', type=str, default='acrobot_obs', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')

    args = parser.parse_args()
    print(args)
    main(args)
