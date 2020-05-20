"""
This implements the Kinodynamic Planning using MPNet, by using MPNet
to generate random samples, that will guide the SST algorithm.
"""
import sys
sys.path.append('deps/sparse_rrt')
sys.path.append('.')
import torch
import model.AE.identity as cae_identity
from model.mlp import MLP
from model import mlp_acrobot
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3
from model.mpnet import KMPNet
from tools import data_loader
from tools.utility import *
from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs
import argparse
import numpy as np
import random
import os
from sparse_rrt import _sst_module

import matplotlib.pyplot as plt
import matplotlib.cm as cm
def main(args):
    #global hl
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    # environment setting
    cae = cae_identity
    mlp = MLP
    cpp_propagator = _sst_module.SystemPropagator()
    if args.env_type == 'pendulum':
        normalize = pendulum.normalize
        unnormalize = pendulum.unnormalize
        system = standard_cpp_systems.PSOPTPendulum()
        dynamics = None
        enforce_bounds = None
        step_sz = 0.002
        num_steps = 20

    elif args.env_type == 'cartpole':
        normalize = cart_pole.normalize
        unnormalize = cart_pole.unnormalize
        dynamics = cartpole.dynamics
        system = _sst_module.CartPole()
        enforce_bounds = cartpole.enforce_bounds
        step_sz = 0.002
        num_steps = 20
    elif args.env_type == 'cartpole_obs':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.CartPole()
        dynamics = cartpole.dynamics
        enforce_bounds = cartpole.enforce_bounds
        step_sz = 0.002
        num_steps = 20
    elif args.env_type == 'acrobot_obs':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP
        cae = CAE_acrobot_voxel_2d
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
        obs_width = 6.0
        IsInCollision = acrobot_obs.IsInCollision
    elif args.env_type == 'acrobot_obs_2':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP2
        cae = CAE_acrobot_voxel_2d_2
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_3':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP3
        cae = CAE_acrobot_voxel_2d_2
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_4':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP3
        cae = CAE_acrobot_voxel_2d_3
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_5':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP
        cae = CAE_acrobot_voxel_2d_3
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_6':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP4
        cae = CAE_acrobot_voxel_2d_3
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_7':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP5
        cae = CAE_acrobot_voxel_2d_3
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20
    elif args.env_type == 'acrobot_obs_8':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        system = _sst_module.PSOPTAcrobot()
        mlp = mlp_acrobot.MLP6
        cae = CAE_acrobot_voxel_2d_3
        #dynamics = acrobot_obs.dynamics
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = acrobot_obs.enforce_bounds
        step_sz = 0.02
        num_steps = 20

    mpnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp)
    # load net
    # load previously trained model if start epoch > 0
    model_dir = args.model_dir
    model_dir = model_dir+'cost_'+args.env_type+"_lr%f_%s_step_%d/" % (args.learning_rate, args.opt, args.num_steps)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path='cost_kmpnet_epoch_%d_direction_%d_step_%d.pkl' %(args.start_epoch, args.direction, args.num_steps)
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

    """
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
    """
    if args.start_epoch > 0:
        #load_opt_state(mpnet, os.path.join(args.model_path, model_path))
        load_opt_state(mpnet, os.path.join(model_dir, model_path))

    # load train and test data
    print('loading...')
    seen_test_data = data_loader.load_test_dataset(args.seen_N, args.seen_NP,
                              args.path_folder, True, args.seen_s, args.seen_sp)
    obc, obs, paths, sgs, path_lengths, controls, costs = seen_test_data
    obc = obc.astype(np.float32)
    
    for pi in range(len(paths)):
        new_obs_i = []
        obs_i = obs[pi]
        plan_res_env = []
        plan_time_env = []
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

        for pj in range(len(paths[pi])):
            
            # on the entire state space, visualize the cost
            # visualization
            """
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.set_autoscale_on(True)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_ylim(-np.pi, np.pi)
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
            """
            dtheta = 0.1
            feasible_points = []
            infeasible_points = []

            imin = 0
            imax = int(2*np.pi/dtheta)

            x0 = paths[pi][pj][0]
            xT = paths[pi][pj][-1]
            # visualize the cost on all grids
            costmaps = []
            cost_to_come = []
            cost_to_go = []
            for i in range(imin, imax):
                costmaps_i = []
                for j in range(imin, imax):
                    x = np.array([dtheta*i-np.pi, dtheta*j-np.pi, 0., 0.])
                    cost_to_come_in = np.array([np.concatenate([x0, x])])
                    cost_to_come_in = torch.from_numpy(cost_to_come_in).type(torch.FloatTensor)
                    cost_to_come_in = normalize(cost_to_come_in, args.world_size)
                    cost_to_go_in = np.array([np.concatenate([x, xT])])
                    cost_to_go_in = torch.from_numpy(cost_to_go_in).type(torch.FloatTensor)
                    cost_to_go_in = normalize(cost_to_go_in, args.world_size)
                    
                    cost_to_come.append(cost_to_come_in)
                    cost_to_go.append(cost_to_go_in)
            cost_to_come = torch.cat(cost_to_come, 0)
            cost_to_go = torch.cat(cost_to_go, 0)
            print(cost_to_go.size())
            obc_i_torch = torch.from_numpy(np.array([obc[pi]])).type(torch.FloatTensor).repeat(len(cost_to_go), 1, 1, 1)
            print(obc_i_torch.size())
            cost_sum = mpnet(cost_to_come, obc_i_torch) + mpnet(cost_to_go, obc_i_torch)
            cost_to_come_val = mpnet(cost_to_come, obc_i_torch).detach().numpy().reshape(imax-imin,-1)
            cost_to_go_val = mpnet(cost_to_go, obc_i_torch).detach().numpy().reshape(imax-imin,-1)
            print('cost_to_come:')
            print(cost_to_come_val)
            print('cost_to_come[(imax+imin)//2,(imax+imin)//2]: ', cost_to_come_val[(imax+imin)//2,(imax+imin)//2])
            print('cost_to_go_val:')
            print(cost_to_go_val)
            cost_sum = cost_sum[:,0].detach().numpy().reshape(imax-imin,-1)
            for i in range(imin, imax):
                costmaps_i = []
                for j in range(imin, imax):
                    costmaps_i.append(cost_sum[i][j])
                    #if IsInCollision(x, obs_i):
                    #    costmaps_i.append(1000.)
                    #else:
                    #    costmaps_i.append(cost_sum[i][j])
                costmaps.append(costmaps_i)
            costmaps = np.array(costmaps)
            # plot the costmap
            print(costmaps)
            print(costmaps.min())
            print(costmaps.max())
            costmaps = costmaps - costmaps.min() + 1.0 # map to 1.0 to infty
            costmaps = np.log(costmaps)
            im = plt.imshow(costmaps, cmap='hot', interpolation='nearest')
            
            
            for i in range(imin, imax):
                for j in range(imin, imax):
                    x = np.array([dtheta*i-np.pi, dtheta*j-np.pi, 0., 0.])
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
            #ax.scatter(feasible_points[:,0], feasible_points[:,1], c='yellow')
            #ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c='pink')
            #for i in range(len(data)):
            #    update_line(hl, ax, data[i])
            #draw_update_line(ax)
            #state_t = start_state
            
            plt.colorbar(im)
            plt.show()
            plt.waitforbuttonpress()

parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
parser.add_argument('--no_val_paths', type=int,default=50,help='number of optimal paths in each environment')
parser.add_argument('--num_steps', type=int, default=20)


parser.add_argument('--seen_N', type=int, default=10)
parser.add_argument('--seen_NP', type=int, default=100)
parser.add_argument('--seen_s', type=int, default=0)
parser.add_argument('--seen_sp', type=int, default=800)


# Model parameters
parser.add_argument('--total_input_size', type=int, default=8, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int, default=32, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int , default=136, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=1, help='dimension of the input vector')

parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--path_folder', type=str, default='./data/acrobot_obs/')
parser.add_argument('--path_file', type=str, default='path.pkl')

parser.add_argument('--start_epoch', type=int, default=1200)
parser.add_argument('--env_type', type=str, default='acrobot_obs', help='environment')
parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
parser.add_argument('--direction', type=int, default=0, help='0: forward, 1: backward')
#parser.add_argument('--opt', type=str, default='Adagrad')
args = parser.parse_args()
print(args)
main(args)
