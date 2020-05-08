"""
This transform python cost network to C++
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

    # load train and test data
    print('loading...')
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

    # set to training model to enable dropout
    mpnet.train()

    MLP = mpnet.mlp
    encoder = mpnet.encoder
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

    loss = mpnet.loss(mpnet(bi, bobs), bt)

    traced_encoder = torch.jit.trace(encoder, (bobs))
    encoder_output = encoder(bobs)
    mlp_input = torch.cat((encoder_output, bi), 1)
    traced_MLP = torch.jit.trace(MLP, (mlp_input))
    traced_encoder.save("costnet_%s_encoder_epoch_%d_step_%d.pt" % (args.env_type, args.start_epoch, args.num_steps))
    traced_MLP.save("costnet_%s_MLP_epoch_%d_step_%d.pt" % (args.env_type, args.start_epoch, args.num_steps))

    # test the traced model
    serilized_encoder = torch.jit.script(encoder)
    serilized_MLP = torch.jit.script(MLP)
    serilized_encoder_output = serilized_encoder(bobs)
    serilized_MLP_input = torch.cat((serilized_encoder_output, bi), 1)
    serilized_MLP_output = serilized_MLP(serilized_MLP_input)
    print('encoder output: ', serilized_encoder_output)
    print('MLP output: ', serilized_MLP_output)

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

    args = parser.parse_args()
    print(args)
    main(args)
