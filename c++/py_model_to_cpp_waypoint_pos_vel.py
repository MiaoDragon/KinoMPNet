"""
This transform python cost network to C++
"""
import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
import torch
import model.AE.identity as cae_identity
from model.mlp import MLP
from model import mlp_acrobot, mlp_cartpole
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3, CAE_cartpole_voxel_2d
from model.mpnet import KMPNet
from model.mpnet_pos_vel import PosVelKMPNet
from tools.utility import *
import argparse
import numpy as np
import random
import os
import torch.nn as nn
from tensorboardX import SummaryWriter


# click can achieve similar functionality as argparse
#@click.command()
#@click.option('--system', default='sst_envs')
#@click.option('--model', default='acrobot_obs')
#@click.option('--setup', default='default_norm')
#@click.option('--ep', default=5000)
def main(args):
    # load MPNet
    #global hl
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        
    if args.debug:
        from sparse_rrt import _sst_module
        from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs
        from tools import data_loader

        cpp_propagator = _sst_module.SystemPropagator()
    if args.env_type == 'pendulum':
        if args.debug:
            normalize = pendulum.normalize
            unnormalize = pendulum.unnormalize
            system = standard_cpp_systems.PSOPTPendulum()
            dynamics = None
            enforce_bounds = None
            step_sz = 0.002
            num_steps = 20

    elif args.env_type == 'cartpole':
        if args.debug:
            normalize = cart_pole.normalize
            unnormalize = cart_pole.unnormalize
            dynamics = cartpole.dynamics
            system = _sst_module.CartPole()
            enforce_bounds = cartpole.enforce_bounds
            step_sz = 0.002
            num_steps = 20
    elif args.env_type == 'cartpole_obs':
        if args.debug:
            normalize = cart_pole_obs.normalize
            unnormalize = cart_pole_obs.unnormalize
            system = _sst_module.PSOPTCartPole()

            dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
            enforce_bounds = cart_pole_obs.enforce_bounds
            step_sz = 0.002
            num_steps = 20
        mlp = mlp_cartpole.MLP
        cae = CAE_cartpole_voxel_2d
    elif args.env_type == 'acrobot_obs':
        if args.debug:
            normalize = acrobot_obs.normalize
            unnormalize = acrobot_obs.unnormalize
            system = _sst_module.PSOPTAcrobot()
            #dynamics = acrobot_obs.dynamics
            dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
            enforce_bounds = acrobot_obs.enforce_bounds
            step_sz = 0.02
            num_steps = 20
        mlp = mlp_acrobot.MLP
        cae = CAE_acrobot_voxel_2d

    if args.loss == 'mse':
        loss_f = nn.MSELoss()
        #loss_f = mse_loss

    elif args.loss == 'l1_smooth':
        loss_f = nn.SmoothL1Loss()
        #loss_f = l1_smooth_loss

    elif args.loss == 'mse_decoupled':
        def mse_decoupled(y1, y2):
            # for angle terms, wrap it to -pi~pi
            l_0 = torch.abs(y1[:,0] - y2[:,0])
            l_1 = torch.abs(y1[:,1] - y2[:,1])
            l_2 = torch.abs(y1[:,2] - y2[:,2]) # angular dimension
            l_3 = torch.abs(y1[:,3] - y2[:,3])
            cond = l_2 > np.pi
            l_2 = torch.where(cond, 2*np.pi-l_2, l_2)
            l_0 = torch.mean(l_0)
            l_1 = torch.mean(l_1)
            l_2 = torch.mean(l_2)
            l_3 = torch.mean(l_3)
            return torch.stack([l_0, l_1, l_2, l_3])
        loss_f = mse_decoupled


    mpnet_pnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size // 2,
                   cae, mlp, loss_f)
    mpnet_vnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size // 2,
                   cae, mlp, loss_f)

    mpnet_pos_vel = PosVelKMPNet(mpnet_p, mpnet_v)
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
    
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_pnet_path='kmpnet_pnet_epoch_%d_direction_%d_step_%d.pkl' %(args.start_epoch, args.direction, args.num_steps)
    model_vnet_path='kmpnet_vnet_epoch_%d_direction_%d_step_%d.pkl' %(args.start_epoch, args.direction, args.num_steps)
    torch_seed, np_seed, py_seed = 0, 0, 0
    if args.start_epoch > 0:
        #load_net_state(mpnet, os.path.join(args.model_path, model_path))
        load_net_state(mpnet_p, os.path.join(model_dir, model_pnet_path))
        load_net_state(mpnet_v, os.path.join(model_dir, model_vnet_path))

        #torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(model_dir, model_pnet_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    if torch.cuda.is_available():
        mpnet_pnet.cuda()
        mpnet_pnet.mlp.cuda()
        mpnet_pnet.encoder.cuda()

        mpnet_vnet.cuda()
        mpnet_vnet.mlp.cuda()
        mpnet_vnet.encoder.cuda()


    # load train and test data
    print('loading...')
    if args.debug:
        obs, cost_dataset, cost_targets, env_indices, \
        _, _, _, _ = data_loader.load_train_dataset_cost(N=args.no_env, NP=args.no_motion_paths,
                                                    data_folder=args.path_folder, obs_f=True,
                                                    direction=args.direction,
                                                    dynamics=dynamics, enforce_bounds=enforce_bounds,
                                                    system=system, step_sz=step_sz, num_steps=args.num_steps)
        # randomize the dataset before training
        data=list(zip(cost_dataset,cost_targets,env_indices))
        random.shuffle(data)
        dataset,targets,env_indices=list(zip(*data))
        dataset = list(dataset)
        targets = list(targets)
        env_indices = list(env_indices)
        dataset = np.array(dataset)
        targets = np.array(targets)
        env_indices = np.array(env_indices)
        # record
        bi = dataset.astype(np.float32)
        print('bi shape:')
        print(bi.shape)
        bt = targets
        bi = torch.FloatTensor(bi)
        bt = torch.FloatTensor(bt)
        bi = normalize(bi, args.world_size)
        bi=to_var(bi)
        bt=to_var(bt)
        if obs is None:
            bobs = None
        else:
            bobs = obs[env_indices].astype(np.float32)
            bobs = torch.FloatTensor(bobs)
            bobs = to_var(bobs)
    else:
        bobs = np.random.rand(1,1,args.AE_input_size,args.AE_input_size)
        bobs = torch.from_numpy(bobs).type(torch.FloatTensor)
        bobs = to_var(bobs)
        bi = np.random.rand(1, args.total_input_size)
        bt = np.random.rand(1, args.output_size)
        bi = torch.from_numpy(bi).type(torch.FloatTensor)
        bt = torch.from_numpy(bt).type(torch.FloatTensor)
        bi = to_var(bi)
        bt = to_var(bt)
    # set to training model to enable dropout
    mpnet.train()
    #mpnet.eval()

    MLP = mpnet.mlp
    encoder = mpnet.encoder
    traced_encoder = torch.jit.trace(encoder, (bobs))
    encoder_output = encoder(bobs)
    mlp_input = torch.cat((encoder_output, bi), 1)
    traced_MLP = torch.jit.trace(MLP, (mlp_input))
    traced_encoder.save('%s_encoder_lr%f_epoch_%d_step_%d.pt' % (args.env_type, args.learning_rate, args.start_epoch, args.num_steps))
    traced_MLP.save('%s_MLP_lr%f_epoch_%d_step_%d.pt' % (args.env_type, args.learning_rate, args.start_epoch, args.num_steps))

    #traced_encoder.save("%s_encoder_epoch_%d.pt" % (args.env_type, args.start_epoch))
    #traced_MLP.save("%s_MLP_epoch_%d.pt" % (args.env_type, args.start_epoch))

    # test the traced model
    serilized_encoder = torch.jit.script(encoder)
    serilized_MLP = torch.jit.script(MLP)
    serilized_encoder_output = serilized_encoder(bobs)
    serilized_MLP_input = torch.cat((serilized_encoder_output, bi), 1)
    serilized_MLP_output = serilized_MLP(serilized_MLP_input)
    print('encoder output: ', serilized_encoder_output)
    print('MLP output: ', serilized_MLP_output)
    print('data: ', bt)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='./results/',help='path for saving trained models')
    parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
    parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
    parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
    parser.add_argument('--num_steps', type=int, default=20)

    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
    parser.add_argument('--AE_input_size', type=int, default=2800, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')

    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
    parser.add_argument('--device', type=int, default=0, help='cuda device')

    parser.add_argument('--path_folder', type=str, default='../data/simple/')
    parser.add_argument('--path_file', type=str, default='train')

    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--env_type', type=str, default='cartpole', help='environment')
    parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--direction', type=int, default=0, help='0: forward, 1: backward')
    parser.add_argument('--debug', type=int, default=0, help='0: debug mode 1: normal mode')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--multigoal', type=int, default=0, help='using itermediate nodes as goal or not')


    args = parser.parse_args()
    print(args)
    main(args)
