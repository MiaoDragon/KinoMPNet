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

import matplotlib.pyplot as plt
#fig = plt.figure()

import sys
sys.path.append('..')

import numpy as np
#from tvlqr.python_tvlqr import tvlqr
#from tvlqr.python_lyapunov import sample_tv_verify
from plan_utility.data_structure import *

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
    obc, obs, paths, path_lengths = seen_test_data
    for i in range(10):
        # randomly pick up a path
        p_i = np.random.randint(low=0, high=len(paths[0]))
        # randomly pick up a point
        node_i = np.random.randint(low=0, high=path_lengths[0][p_i]-1)
        node = paths[0][p_i][node_i]
        plot_nodes = np.zeros((1,len(node)))
        for j in range(len(paths[0])):
            dif = paths[0][j] - node
            norms = np.linalg.norm(dif, axis=1)
            similar_indices = np.where(norms<=1e-1)
            similar_indices = similar_indices[0]
            print(similar_indices)
            if len(similar_indices) == 0:
                continue
            similar_indices = similar_indices[0]
            plot_indices = similar_indices + 1
            #print((norms<=1e-1).shape)
            #print(norms<=1e-1)
            ax.plot([paths[0][j][plot_indices][0], paths[0][j][-1][0]], [paths[0][j][plot_indices][1], paths[0][j][-1][1]], c='blue')
            #plot_nodes = np.append(plot_nodes, paths[0][j][plot_indices], axis=0)
        print(node)
        ax.plot([node[0], paths[0][p_i][-1][0]], [node[1], paths[0][p_i][-1][1]], c='yellow')
        plt.scatter(node[0], node[1], c='yellow')
        #plt.scatter(plot_nodes[:,0], plot_nodes[:,1], c='blue')
        plt.waitforbuttonpress()
        plt.cla()
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_lr005_SGD/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=1)
    parser.add_argument('--seen_NP', type=int, default=700)
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
