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

#from gem_eval_original_mpnet import eval_tasks
from iterative_plan_and_retreat.gem_eval import eval_tasks
from torch.autograd import Variable
import copy
import os
import gc
import random
from tools.utility import *
from plan_utility import pendulum, acrobot_obs
#from sparse_rrt.systems import standard_cpp_systems
#from sparse_rrt import _sst_module

from iterative_plan_and_retreat.data_structure import *
from iterative_plan_and_retreat.plan_general import propagate

#from plan_utility.data_structure import *
#from plan_utility.plan_general_original_mpnet import propagate
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
        #IsInCollision = lambda x, obs: False
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        xdot = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d
        mlp = mlp_acrobot.MLP
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0

    elif args.env_type == 'acrobot_obs_2':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        xdot = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_2
        mlp = mlp_acrobot.MLP2
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0

    elif args.env_type == 'acrobot_obs_3':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        xdot = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        mlp = mlp_acrobot.MLP3
        cae = CAE_acrobot_voxel_2d_2
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0


    elif args.env_type == 'acrobot_obs_5':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        xdot = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_3
        mlp = mlp_acrobot.MLP
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif args.env_type == 'acrobot_obs_6':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        xdot = acrobot_obs.dynamics
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_3
        mlp = mlp_acrobot.MLP4
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0

    elif args.env_type == 'acrobot_obs_6':
        IsInCollision =acrobot_obs.IsInCollision
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        xdot = acrobot_obs.dynamics
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        mlp = mlp_acrobot.MLP5
        cae = CAE_acrobot_voxel_2d_3
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0


    elif args.env_type == 'acrobot_obs_8':
        IsInCollision =acrobot_obs.IsInCollision
        #IsInCollision = lambda x, obs: False
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        xdot = acrobot_obs.dynamics
        jax_dynamics = acrobot_obs.jax_dynamics
        enforce_bounds = acrobot_obs.enforce_bounds
        cae = CAE_acrobot_voxel_2d_3
        mlp = mlp_acrobot.MLP6
        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        #num_steps = 21
        num_steps = 21#args.num_steps*2
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 400, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        #traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init:
        #def cem_trajopt(x0, x1, step_sz, num_steps, x_init, u_init, t_init):
        #    u, t = acrobot_obs.trajopt(x0, x1, 500, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        #    xs, us, dts, valid = propagate(x0, u, t, dynamics=dynamics, enforce_bounds=enforce_bounds, IsInCollision=lambda x: False, system=system, step_sz=step_sz)
        #    return xs, us, dts
        #traj_opt = cem_trajopt
        goal_S0 = np.diag([1.,1.,0,0])
        goal_rho0 = 1.0



    # load previously trained model if start epoch > 0
    #model_path='kmpnet_epoch_%d_direction_0_step_%d.pkl' %(args.start_epoch, args.num_steps)
    mlp_path = os.path.join(os.getcwd(),'acrobot_mlp_annotated_test_gpu.pt')
    encoder_path = os.path.join(os.getcwd(),'acrobot_encoder_annotated_test_cpu.pt')

    #####################################################
    def plan_one_path(obs, obc, start_state, goal_state, max_iteration, out_queue):
        if args.env_type == 'pendulum':
            system = standard_cpp_systems.PSOPTPendulum()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
            step_sz = 0.002
            num_steps = 20
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, num_steps, 1, 20, step_sz)

        elif args.env_type == 'cartpole_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
            system = _sst_module.CartPole()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            num_steps = 20
            traj_opt = lambda x0, x1, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 500, num_steps, step_sz*1, step_sz*50, x_init, u_init, t_init)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0
        elif args.env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4', 'acrobot_obs_8']:
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            obs_width = 6.0
            psopt_system = _sst_module.PSOPTAcrobot()
            propagate_system = standard_cpp_systems.RectangleObs(obs, 6., 'acrobot')
            step_sz = 0.02
            num_steps = 21
            goal_radius=2.0
            random_seed=0
            delta_near=1.0
            delta_drain=0.5

        planner = _sst_module.DeepSMPWrapper(mlp_path, encoder_path, propagate_system, psopt_system)
        # generate a path by using SST to plan for some maximal iterations
        time0 = time.time()
        res_x, res_u, res_t = planner.plan("sst", obs, start_state, goal_state, \
                                max_iteration, goal_radius, propagate_system.distance_computer(), \
                                delta_near, delta_drain)
        plan_time = time.time() - time0
        if len(res_x) == 0:
            print('failed.')
            out_queue.put(0)
        else:
            print('path succeeded.')
            out_queue.put(1)
    ####################################################################################



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

    queue = Queue(1)
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.

    obc, obs, paths, sgs, path_lengths, controls, costs = seen_test_data
    obc = obc.astype(np.float32)
    obc = torch.from_numpy(obc)
    if torch.cuda.is_available():
        obc = obc.cuda()
    for i in range(len(paths)):
        new_obs_i = []
        obs_i = obs[i]
        for k in range(len(obs_i)):
            obs_pt = []
            obs_pt.append(obs[k][0]-obs_width/2)
            obs_pt.append(obs[k][1]-obs_width/2)
            obs_pt.append(obs[k][0]-obs_width/2)
            obs_pt.append(obs[k][1]+obs_width/2)
            obs_pt.append(obs[k][0]+obs_width/2)
            obs_pt.append(obs[k][1]+obs_width/2)
            obs_pt.append(obs[k][0]+obs_width/2)
            obs_pt.append(obs[k][1]-obs_width/2)
            new_obs_i.append(obs_pt)
        obs_i = new_obs_i
        for j in range(len(paths[i])):
            start_state = sgs[i][j][0]
            goal_state = sgs[i][j][1]
            p = Process(target=plan_one_path, args=(obs, obc, start_state, goal_state, 500, queue))
            p.start()
            p.join()
            res = queue.get()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_lr0.010000_SGD/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=10)
    parser.add_argument('--seen_NP', type=int, default=2)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=800)
    parser.add_argument('--unseen_N', type=int, default=0)
    parser.add_argument('--unseen_NP', type=int, default=0)
    parser.add_argument('--unseen_s', type=int, default=0)
    parser.add_argument('--unseen_sp', type=int, default=0)
    parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=8, help='dimension of total input')
    parser.add_argument('--AE_input_size', type=int, default=32, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=136, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=4, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_folder', type=str, default='./data/acrobot_obs/')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=5000)
    parser.add_argument('--env_type', type=str, default='acrobot_obs', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--num_steps', type=int, default=20)

    args = parser.parse_args()
    print(args)
    main(args)
