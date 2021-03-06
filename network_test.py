from __future__ import print_function
from ctypes import *
import sys
sys.path.append('deps/sparse_rrt')

#import faulthandler
#faulthandler.enable()
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")
#from env.cartpole import CartPole
import sparse_rrt
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import matplotlib.pyplot as plt
from sparse_rrt.systems.pendulum import Pendulum


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


import model.AE.identity as cae_identity
from model.mlp import MLP
from model.mpnet import KMPNet
import numpy as np
import argparse
import os
import cv2
import torch
from gem_eval import eval_tasks
from torch.autograd import Variable
import copy
import os
import gc
import random
from tools.utility import *
from plan_utility import pendulum
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
from tools import data_loader
import time
from sparse_rrt.planners import SST, RRT
from sparse_rrt.visualization import show_image
import plan_general

def update_line(h, ax, fig, new_data):
    h.set_xdata(np.append(h.get_xdata(), new_data[0]))
    h.set_ydata(np.append(h.get_ydata(), new_data[1]))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def clear_line(h, ax, fig):
    h.set_xdata(np.array([]))
    h.set_ydata(np.array([]))
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


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
        cae = cae_identity
        mlp = MLP
        system = standard_cpp_systems.PSOPTPendulum()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
        max_iter = 100
        min_time_steps = 10
        max_time_steps = 200
        integration_step = 0.002
        goal_radius=0.1
        random_seed=0
        sst_delta_near=0.05
        sst_delta_drain=0.02
        vel_idx = [1]


    mpNet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp)
    # load previously trained model if start epoch > 0
    model_path='kmpnet_epoch_%d.pkl' %(args.start_epoch)
    if args.start_epoch > 0:
        load_net_state(mpNet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)
    if torch.cuda.is_available():
        mpNet.cuda()
        mpNet.mlp.cuda()
        mpNet.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
    if args.start_epoch > 0:
        load_opt_state(mpNet, os.path.join(args.model_path, model_path))


    # load train and test data
    print('loading...')
    if args.seen_N > 0:
        seen_test_data = data_loader.load_test_dataset(N=args.seen_N, NP=args.seen_NP, s=args.seen_s, sp=args.seen_sp,
                                                       p_folder=args.path_folder, obs_f=obs_file, obc_f=obc_file)
    if args.unseen_N > 0:
        unseen_test_data = data_loader.load_test_dataset(N=args.unseen_N, NP=args.unseen_NP, s=args.unseen_s, sp=args.unseen_sp,
                                                       p_folder=args.path_folder, obs_f=obs_file, obc_f=obc_file)
    # test
    # testing
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.
    T = 1


    obc, obs, paths, path_lengths = seen_test_data
    if obs is not None:
        obs = obs.astype(np.float32)
        obs = torch.from_numpy(obs)
    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    normalize_func=lambda x: normalize(x, args.world_size)
    unnormalize_func=lambda x: unnormalize(x, args.world_size)

    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        suc_n = 0
        for j in range(len(paths[0])):
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_autoscale_on(True)
            hl, = ax.plot([], [], 'black')
            hl_real, = ax.plot([], [], 'yellow')



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
                path = [paths[i][j][0], paths[i][j][path_lengths[i][j]-1]]
                start = paths[i][j][0]
                end = paths[i][j][path_lengths[i][j]-1]
                #start[1] = 0.
                #end[1] = 0.
                # plot the entire path
                #plt.plot(paths[i][j][:,0], paths[i][j][:,1])
                control = []
                time_step = []
                MAX_NEURAL_REPLAN = 11
                if obs is None:
                    obs_i = None
                    obc_i = None
                else:
                    obs_i = obs[i]
                    obc_i = obc[i]
            for k in range(path_lengths[i][j]):
                update_line(hl, ax, fig, paths[i][j][k])
            print('created RRT')
            # Run planning and print out solution is some statistics every few iterations.
            time0 = time.time()
            start = paths[i][j][0]
            #end = paths[i][j][path_lengths[i][j]-1]
            new_sample = start
            print(new_sample)
            ax.scatter(new_sample[0], new_sample[1], c='r')
            ax.scatter(end[0], end[1], c='g')
            for iteration in range(max_iter):
                clear_line(hl_real, ax, fig)
                #hl_real, = ax.plot([], [], 'yellow')
                ip1 = np.concatenate([new_sample, end])
                np.expand_dims(ip1, 0)
                #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
                time0 = time.time()
                ip1=normalize_func(ip1)
                ip1 = torch.FloatTensor(ip1)
                time_norm += time.time() - time0
                ip1=to_var(ip1)
                if obs is not None:
                    obs = torch.FloatTensor(obs).unsqueeze(0)
                    obs=to_var(obs)
                sample=mpNet(ip1,obs).squeeze(0)
                # unnormalize to world size
                sample=sample.data.cpu().numpy()
                time0 = time.time()
                sample = unnormalize_func(sample)
                ax.scatter(sample[0], sample[1], c='b')
                plt.pause(0.01)

                steer, steer_state, steer_control, steer_time_step = plan_general.steerTo(bvp_solver, start, sample, None, None, step_sz=0.02)
                for k in range(len(steer_state)):
                    update_line(hl_real, ax, fig, steer_state[k])
            plt.waitforbuttonpress()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='./results/',help='path for saving trained models')
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
    parser.add_argument('--AE_input_size', nargs='+', type=int, default=0, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=4, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--path_folder', type=str, default='./data/pendulum/')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=9)
    parser.add_argument('--env_type', type=str, default='pendulum', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 7.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    args = parser.parse_args()
    print(args)
    main(args)
