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

#from model.mlp import MLP
import numpy as np
import argparse
import os
import copy
import os
import gc
import random
#from sparse_rrt.systems import standard_cpp_systems
#from sparse_rrt import _sst_module
from tools import data_loader
import jax
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


import math
import time
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module

import matplotlib.pyplot as plt
#fig = plt.figure()

import sys
sys.path.append('..')

import numpy as np
#from tvlqr.python_tvlqr import tvlqr
#from tvlqr.python_lyapunov import sample_tv_verify
from plan_utility.data_structure import *
from plan_utility.line_line_cc import line_line_cc
import torch
import model.AE.identity as cae_identity
from model.mlp import MLP
from model import mlp_acrobot, mlp_cartpole, mlp_car
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3, CAE_cartpole_voxel_2d, CAE_car_voxel_2d
from model.mpnet import KMPNet
from tools import data_loader
from tools.utility import *
from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs, car_obs
import argparse
import numpy as np
import random
import os
from sparse_rrt import _sst_module



def enforce_bounds(state):
    '''
    check if state satisfies the bound
    apply threshold to velocity and angle
    return a new state toward which the bound has been enforced
    '''
    WIDTH = 2.0
    LENGTH = 1.0

    STATE_X = 0
    STATE_Y = 1

    STATE_THETA = 2 
    MIN_X = -25
    MAX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    
    
    new_state = np.array(state)
    """
    if state[STATE_V] < MIN_V/30.:
        new_state[STATE_V] = MIN_V/30.
    elif state[STATE_V] > MAX_V/30.:
        new_state[STATE_V] = MAX_V/30.
    """
    if state[2] < -np.pi:
        new_state[2] += 2*np.pi
    elif state[2] > np.pi:
        new_state[2] -= 2*np.pi
    return new_state



def overlap(b1corner,b1axis,b1orign,b1dx,b1dy,b2corner,b2axis,b2orign,b2dx,b2dy):
    # this only checks overlap of b1 w.r.t. b2
    # a complete check should do in both directions
    b2ds = [b2dx, b2dy]
    for a in range(0,2):
        t=b1corner[0][0]*b2axis[a][0]+b1corner[0][1]*b2axis[a][1] # project corner to the axis by inner product

        tMin = t
        tMax = t
        for c in range(1,4):
            t = b1corner[c][0]*b2axis[a][0]+b1corner[c][1]*b2axis[a][1] # project corner to the axis by inner product
            # find range by [tMin, tMax]
            if t < tMin:
                tMin = t
            elif t > tMax:
                tMax = t
        # since b2 the other corners (corner 1, 2, 3) are larger than b2orign (project of corner 0 to axis)
        # specifically, the range is [b2orign[i], b2orign[i]+size(i)] (of the projected point by dot product)
        # we only need to compare tMax with b2orign[i], and tMin with size(i)+b2orign[i]
        if ((tMin > (b2ds[a] + b2orign[a])) or (tMax < b2orign[a])):
            return False

    return True

def IsInCollision(x, obc, obc_width=4.):
    car_width = 2.0
    car_len = 1.0
    width = 8.0
    WIDTH = car_width
    LENGTH = car_len
    STATE_X = 0
    STATE_Y = 1
    STATE_THETA = 2
    MIN_X = -25
    MAX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    if x[0] < MIN_X or x[0] > MAX_X or x[1] < MIN_Y or x[1] > MAX_Y:
        return True
        
    robot_corner=np.zeros((4,2),dtype=np.float32)
    robot_axis=np.zeros((2,2),dtype=np.float32)
    robot_orign=np.zeros(2,dtype=np.float32)
    length=np.zeros(2,dtype=np.float32)
    X1=np.zeros(2,dtype=np.float32)
    Y1=np.zeros(2,dtype=np.float32)

    X1[0]=np.cos(x[STATE_THETA])*(WIDTH/2.0)
    X1[1]=-np.sin(x[STATE_THETA])*(WIDTH/2.0)
    Y1[0]=np.sin(x[STATE_THETA])*(LENGTH/2.0)
    Y1[1]=np.cos(x[STATE_THETA])*(LENGTH/2.0)

    for j in range(0,2):
        # order: (left-bottom, right-bottom, right-upper, left-upper)
        # assume angle (state_theta) is clockwise
        robot_corner[0][j]=x[j]-X1[j]-Y1[j]
        robot_corner[1][j]=x[j]+X1[j]-Y1[j]
        robot_corner[2][j]=x[j]+X1[j]+Y1[j]
        robot_corner[3][j]=x[j]-X1[j]+Y1[j]

        # axis: horizontal and vertical
        robot_axis[0][j] = robot_corner[1][j] - robot_corner[0][j]
        robot_axis[1][j] = robot_corner[3][j] - robot_corner[0][j]

    #print('robot corners:')
    #print(robot_corner)
    length[0]=np.sqrt(robot_axis[0][0]*robot_axis[0][0]+robot_axis[0][1]*robot_axis[0][1])
    length[1]=np.sqrt(robot_axis[1][0]*robot_axis[1][0]+robot_axis[1][1]*robot_axis[1][1])
    # normalize the axis
    for i in range(0,2):
        for j in range(0,2):
            robot_axis[i][j]=robot_axis[i][j]/float(length[i])

    # obtain the projection of the left-bottom corner to the axis, to obtain the minimal projection length
    robot_orign[0]=robot_corner[0][0]*robot_axis[0][0]+ robot_corner[0][1]*robot_axis[0][1]
    robot_orign[1]=robot_corner[0][0]*robot_axis[1][0]+ robot_corner[0][1]*robot_axis[1][1]
    #print('robot orign:')
    #print(robot_orign)
    for i in range(len(obc)):
        cf=True

        obs_corner=np.zeros((4,2),dtype=np.float32)
        obs_axis=np.zeros((2,2),dtype=np.float32)
        obs_orign=np.zeros(2,dtype=np.float32)
        length2=np.zeros(2,dtype=np.float32)

        for j in range(0,2):
            # order: (left-bottom, right-bottom, right-upper, left-upper)
            obs_corner[0][j] = obc[i][j]
            obs_corner[1][j] = obc[i][2+j]
            obs_corner[2][j] = obc[i][2*2+j]
            obs_corner[3][j] = obc[i][3*2+j]
            
            # horizontal axis and vertical
            obs_axis[0][j] = obs_corner[1][j] - obs_corner[0][j]
            obs_axis[1][j] = obs_corner[3][j] - obs_corner[0][j]
            #obs_axis[0][j] = obs_corner[3][j] - obs_corner[0][j]
            #obs_axis[1][j] = obs_corner[1][j] - obs_corner[0][j]

        length2[0]=np.sqrt(obs_axis[0][0]*obs_axis[0][0]+obs_axis[0][1]*obs_axis[0][1])
        length2[1]=np.sqrt(obs_axis[1][0]*obs_axis[1][0]+obs_axis[1][1]*obs_axis[1][1])

        # normalize the axis
        for i1 in range(0,2):
            for j1 in range(0,2):
                obs_axis[i1][j1]=obs_axis[i1][j1]/float(length2[i1])

        # obtain the inner product of the left-bottom corner with the axis to obtain the minimal of projection value
        obs_orign[0]=obs_corner[0][0]*obs_axis[0][0]+ obs_corner[0][1]*obs_axis[0][1]  # dot product at 0-th corner
        obs_orign[1]=obs_corner[0][0]*obs_axis[1][0]+ obs_corner[0][1]*obs_axis[1][1]
        # do checking in both direction (b1 -> b2, b2 -> b1). If at least one shows not-overlapping, then it is not overlapping
        cf=overlap(robot_corner,robot_axis,robot_orign,car_width,car_len,obs_corner,obs_axis,obs_orign,width,width)
        cf=cf and overlap(obs_corner,obs_axis,obs_orign,width,width,robot_corner,robot_axis,robot_orign,car_width,car_len)
        if cf==True:
            return True
    return False




def main(args):
    # set seed
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models

    # setup evaluation function and load function
    if args.env_type == 'pendulum':
        obs_file = None
        obc_file = None
        obs_f = False
        #system = standard_cpp_systems.PSOPTPendulum()
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
    elif args.env_type == 'cartpole_obs':
        obs_file = None
        obc_file = None
        obs_f = True
        obs_width = 4.0
        step_sz = 0.002
        psopt_system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 100
    elif args.env_type == 'cartpole_obs_2':
        obs_file = None
        obc_file = None
        obs_f = True
        obs_width = 4.0
        step_sz = 0.002
        psopt_system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP2
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 100
    elif args.env_type == 'cartpole_obs_3':
        obs_file = None
        obc_file = None
        obs_f = True
        obs_width = 4.0
        step_sz = 0.002
        psopt_system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP4
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 200
    elif args.env_type == 'cartpole_obs_4':
        obs_file = None
        obc_file = None
        obs_f = True
        obs_width = 4.0
        step_sz = 0.002
        psopt_system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP3
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 200
  
    elif args.env_type == 'car_obs':
        obs_file = None
        obc_file = None
        obs_f = True
        obs_width = 8.0
        step_sz = 0.002
        psopt_system = _sst_module.Car()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

        normalize = car_obs.normalize
        unnormalize = car_obs.unnormalize
        system = _sst_module.Car()
        mlp = mlp_car.MLP
        cae = CAE_car_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = car_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 500
  
    mpnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp, None)
    # load net
    # load previously trained model if start epoch > 0
    model_dir = args.model_dir
    if args.loss == 'mse':
        if args.multigoal == 0:
            model_dir = model_dir+args.env_type+"_lr%f_%s_step_%d/" % (args.learning_rate, args.opt, args.num_steps)
        else:
            model_dir = model_dir+args.env_type+"_lr%f_%s_step_%d_multigoal/" % (args.learning_rate, args.opt, args.num_steps)
    else:
        if args.multigoal == 0:
            model_dir = model_dir+args.env_type+"_lr%f_%s_loss_%s_step_%d/" % (args.learning_rate, args.opt, args.loss, args.num_steps)
        else:
            model_dir = model_dir+args.env_type+"_lr%f_%s_loss_%s_step_%d_multigoal/" % (args.learning_rate, args.opt, args.loss, args.num_steps)
            
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path='kmpnet_epoch_%d_direction_%d_step_%d.pkl' %(args.start_epoch, args.direction, args.num_steps)
    torch_seed, np_seed, py_seed = 0, 0, 0
    if args.start_epoch > 0:
        #load_net_state(mpnet, os.path.join(args.model_path, model_path))
        load_net_state(mpnet, os.path.join(model_dir, model_path))
        #torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(model_dir, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    if torch.cuda.is_available():
        mpnet.cuda()
        mpnet.mlp.cuda()
        mpnet.encoder.cuda()
        if args.opt == 'Adagrad':
            mpnet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpnet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpnet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            mpnet.set_opt(torch.optim.ASGD, lr=args.learning_rate)
    if args.start_epoch > 0:
        #load_opt_state(mpnet, os.path.join(args.model_path, model_path))
        load_opt_state(mpnet, os.path.join(model_dir, model_path))
    
    #mpnet.eval()
        
        
        
        
        
        
        
        
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
    
    # find path
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    
    #hl_real, = ax.plot([], [], 'r')
    def update_line(h, ax, new_data):
        h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
        #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        #h.set_ydata(np.append(h.get_ydata(), new_data[1]))


    def draw_update_line(ax):
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
 

    # randomly pick up a point in the data, and find similar data in the dataset
    # plot the next point
    obc, obs, paths, sgs, path_lengths, controls, costs = seen_test_data
    for envi in range(2):
        for pathi in range(10):
            obs_i = obs[envi]
            new_obs_i = []
            obs_i = obs[envi]
            plan_res_path = []
            plan_time_path = []
            plan_cost_path = []
            data_cost_path = []
            for k in range(len(obs_i)):
                obs_pt = []
                obs_pt.append(obs_i[k][0]-obs_width/2)
                obs_pt.append(obs_i[k][1]-obs_width/2)
                obs_pt.append(obs_i[k][0]-obs_width/2)
                obs_pt.append(obs_i[k][1]+obs_width/2)
                obs_pt.append(obs_i[k][0]+obs_width/2)
                obs_pt.append(obs_i[k][1]+obs_width/2)
                obs_pt.append(obs_i[k][0]+obs_width/2)
                obs_pt.append(obs_i[k][1]-obs_width/2)
                new_obs_i.append(obs_pt)
            obs_i = new_obs_i

            # visualization
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax_vel = fig.add_subplot(122)
            #ax.set_autoscale_on(True)
            ax.set_xlim(-25, 25)
            ax.set_ylim(-35, 35)
            ax_vel.set_xlim(-40, 40)
            ax_vel.set_ylim(-2, 2)
            
            hl, = ax.plot([], [], 'b')
            #hl_real, = ax.plot([], [], 'r')
            hl_for, = ax.plot([], [], 'g')
            hl_back, = ax.plot([], [], 'r')
            hl_for_mpnet, = ax.plot([], [], 'lightgreen')
            hl_back_mpnet, = ax.plot([], [], 'salmon')

            #print(obs)
            def update_line(h, ax, new_data):
                new_data = wrap_angle(new_data, propagate_system)
                h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
                #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
                #h.set_ydata(np.append(h.get_ydata(), new_data[1]))

            def remove_last_k(h, ax, k):
                h.set_data(h.get_xdata()[:-k], h.get_ydata()[:-k])

            def draw_update_line(ax):
                #ax.relim()
                #ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                #plt.show()

            def wrap_angle(x, system):
                circular = system.is_circular_topology()
                res = np.array(x)
                for i in range(len(x)):
                    if circular[i]:
                        # use our previously saved version
                        res[i] = x[i] - np.floor(x[i] / (2*np.pi))*(2*np.pi)
                        if res[i] > np.pi:
                            res[i] = res[i] - 2*np.pi
                return res
            dx = 1
            dtheta = 0.1
            feasible_points = []
            infeasible_points = []
            imin = 0
            imax = int(2*25./dx)
            jmin = 0
            jmax = int(2*35./dx)

            for i in range(imin, imax):
                for j in range(jmin, jmax):
                    x = np.array([dx*i-25, dx*j-35, 0.])
                    if IsInCollision(x, obs_i):
                        infeasible_points.append(x)
                    else:
                        feasible_points.append(x)
            feasible_points = np.array(feasible_points)
            infeasible_points = np.array(infeasible_points)
            print('feasible points')
            print(feasible_points)
            print('infeasible points')
            print(infeasible_points)
            ax.scatter(feasible_points[:,0], feasible_points[:,1], c='yellow')
            ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c='pink')
            #for i in range(len(data)):
            #    update_line(hl, ax, data[i])
            draw_update_line(ax)
            #state_t = start_state

            xs = paths[envi][pathi]
            us = controls[envi][pathi]
            ts = costs[envi][pathi]
            # propagate data
            p_start = xs[0]
            detail_paths = [p_start]
            detail_controls = []
            detail_costs = []
            state = [p_start]
            control = []
            cost = []
            for k in range(len(us)):
                #state_i.append(len(detail_paths)-1)
                max_steps = int(ts[k]/step_sz)
                accum_cost = 0.
                #print('p_start:')
                #print(p_start)
                #print('data:')
                #print(paths[i][j][k])
                # modify it because of small difference between data and actual propagation
                p_start = xs[k]
                state[-1] = xs[k]
                for step in range(1,max_steps+1):
                    p_start = dynamics(p_start, us[k], step_sz)
                    p_start = enforce_bounds(p_start)
                    detail_paths.append(p_start)
                    accum_cost += step_sz
                    if (step % 1 == 0) or (step == max_steps):
                        state.append(p_start)
                        #print('control')
                        #print(controls[i][j])
                        cost.append(accum_cost)
                        accum_cost = 0.
            #print('p_start:')
            #print(p_start)
            #print('data:')
            #print(paths[i][j][-1])
            state[-1] = xs[-1]
            #print(len(state))


            xs_to_plot = np.array(state)
            for i in range(len(xs_to_plot)):
                xs_to_plot[i] = wrap_angle(xs_to_plot[i], psopt_system)
            ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='green')
            # draw start and goal
            #ax.scatter(start_state[0], goal_state[0], marker='X')
            draw_update_line(ax)
            #ax_vel.scatter(xs_to_plot[:,1], xs_to_plot[:,3], c='green', s=0.1)
            #draw_update_line(ax_vel)
            
            plt.waitforbuttonpress()

            
            
            # visualize mPNet path
            mpnet_paths = []
            mpnet_dropout_paths = []  # list of list
            state = xs[0]
            #for k in range(int(len(xs_to_plot)/args.num_steps)):
            print('length:')
            print(len(xs_to_plot))
            print('end point:')
            print(sgs[envi][pathi][-1])
            for k in range(int(len(xs_to_plot)/args.num_steps)):
                # using eval (without dropout, to obtain the mean points)
                #mpnet.eval()
                mpnet.train()
                mpnet_paths.append(state)
                #bi = np.concatenate([state, xs[-1]])
                bi = np.concatenate([state, sgs[envi][pathi][-1]])
                bi = np.array([bi])
                print('input:')
                print(bi)
                bi = torch.from_numpy(bi).type(torch.FloatTensor)
                #print(bi)
                bi = normalize(bi, args.world_size)
                print('normalized input:')
                print(bi)
                bi=to_var(bi)
                if obc is None:
                    bobs = None
                else:
                    bobs = np.array([obc[envi]]).astype(np.float32)
                    print(bobs.shape)
                    bobs = torch.FloatTensor(bobs)
                    bobs = to_var(bobs)
                bt = mpnet(bi, bobs).cpu()
                bt = unnormalize(bt, args.world_size)
                bt = bt.detach().numpy()
                #print(bt.shape)
                
                # using train (with dropout)
                mpnet.train()
                bi = np.concatenate([state, sgs[envi][pathi][-1]])
                bi = np.array([bi])
                bi = torch.from_numpy(bi).type(torch.FloatTensor)
                bi = normalize(bi, args.world_size)
                bi = bi.repeat(16, 1)
                bi = to_var(bi)
                if obc is None:
                    bobs = None
                else:
                    bobs = np.array([obc[envi]]).astype(np.float32)
                    print(bobs.shape)
                    bobs = torch.FloatTensor(bobs)
                    bobs = bobs.repeat(16,1,1,1)
                    bobs = to_var(bobs)
                bt = mpnet(bi, bobs).cpu()
                bt = unnormalize(bt, args.world_size)
                bt = bt.detach().numpy()
                mpnet_dropout_paths.append(bt)
                
                
                state = bt[0]

            
            
            # plot with dropout
            for k in range(len(mpnet_dropout_paths)):
                xs_to_plot = np.array(mpnet_dropout_paths[k])
                print(len(xs_to_plot))
                for i in range(len(xs_to_plot)):
                    xs_to_plot[i] = wrap_angle(xs_to_plot[i], psopt_system)
                ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='lightgreen', alpha=0.5)
                            
            #print(mpnet_paths)
            xs_to_plot_mean = np.array(mpnet_paths)
            #print(len(xs_to_plot_mean))
            for i in range(len(xs_to_plot_mean)):
                xs_to_plot_mean[i] = wrap_angle(xs_to_plot_mean[i], psopt_system)
            ax.scatter(xs_to_plot_mean[:,0], xs_to_plot_mean[:,1], c='lightblue')

            
            for k in range(len(mpnet_dropout_paths)):
                xs_to_plot = np.array(mpnet_dropout_paths[k])
                #print(len(xs_to_plot))
                for i in range(len(xs_to_plot)):
                    xs_to_plot[i] = wrap_angle(xs_to_plot[i], psopt_system)
                for i in range(len(xs_to_plot)):
                    ax.plot([xs_to_plot_mean[k,0], xs_to_plot[i,0]], [xs_to_plot_mean[k,1], xs_to_plot[i,1]], c='skyblue', alpha=0.5)

            
            
            # draw start and goal
            #ax.scatter(start_state[0], goal_state[0], marker='X')
            draw_update_line(ax)
            #ax_vel.scatter(xs_to_plot[:,1], xs_to_plot[:,3], c='lightblue')
            #draw_update_line(ax_vel)
            plt.waitforbuttonpress()

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
    parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
    parser.add_argument('--num_steps', type=int, default=500)
    parser.add_argument('--direction', type=int, default=0)

    parser.add_argument('--seen_N', type=int, default=1)
    parser.add_argument('--seen_NP', type=int, default=20)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=800)
    parser.add_argument('--unseen_N', type=int, default=0)
    parser.add_argument('--unseen_NP', type=int, default=0)
    parser.add_argument('--unseen_s', type=int, default=0)
    parser.add_argument('--unseen_sp', type=int, default=0)
    parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=6, help='dimension of total input')
    parser.add_argument('--AE_input_size', nargs='+', type=int, default=32, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=38, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=3, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_folder', type=str, default='./data/car_obs/')
    parser.add_argument('--obs_file', type=str, default='./data/car_obs/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/car_obs/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=950)
    parser.add_argument('--env_type', type=str, default='car_obs', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[25.0, 35.0, 3.141592653589793], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--loss', type=str, default='mse_decoupled')
    parser.add_argument('--multigoal', type=int, default=0, help='using itermediate nodes as goal or not')

    args = parser.parse_args()
    print(args)
    main(args)
