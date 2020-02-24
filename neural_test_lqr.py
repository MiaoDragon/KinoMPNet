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
sys.path.append('.')
from sparse_rrt import _sst_module
import model.AE.identity as cae_identity
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3
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
        system = _sst_module.PSOPTAcrobot()
        #dynamics = acrobot_obs.dynamics
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d
        mlp = mlp_acrobot.MLP
        obs_f = True
        
        #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
    elif args.env_type == 'acrobot_obs_2':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_2
        mlp = mlp_acrobot.MLP2
        obs_f = True
        #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
    elif args.env_type == 'acrobot_obs_5':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_3
        mlp = mlp_acrobot.MLP
        obs_f = True
        #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
    elif args.env_type == 'acrobot_obs_8':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_3
        mlp = mlp_acrobot.MLP6
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
            fes_path_, valid_path_ = eval_tasks(mpNet0, mpNet1, args.env_type, seen_test_data, args.model_path, 'seen', normalize_func, unnormalize_func, dynamics, jac_A, jac_B, enforce_bounds, IsInCollision)
            valid_path = valid_path_.flatten()
            fes_path = fes_path_.flatten()   # notice different environments are involved
            seen_test_suc_rate += fes_path.sum() / valid_path.sum()
        # unseen
        if args.unseen_N > 0:
            time_file = os.path.join(args.model_path,'time_unseen_epoch_%d_mlp.p' % (args.start_epoch))
            fes_path_, valid_path_ = eval_tasks(mpNet0, mpNet1, args.env_type, unseen_test_data, args.model_path, 'unseen', normalize_func, unnormalize_func, dynamics, jac_A, jac_B, enforce_bounds, IsInCollision)
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
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_8_lr0.010000_SGD_step_10/',help='path for saving trained models')
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
    parser.add_argument('--start_epoch', type=int, default=650)
    parser.add_argument('--env_type', type=str, default='acrobot_obs_8', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--num_step', type=int, default=10)
    args = parser.parse_args()
    print(args)
    main(args)
