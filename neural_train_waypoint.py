"""
This implements the Kinodynamic Planning using MPNet, by using MPNet
to generate random samples, that will guide the SST algorithm.
"""
import sys
sys.path.append('deps/sparse_rrt')
sys.path.append('.')
import torch
import torch.nn as nn
import model.AE.identity as cae_identity
from model.mlp import MLP
from model import mlp_acrobot, mlp_cartpole
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3, CAE_cartpole_voxel_2d
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
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 20
    elif args.env_type == 'cartpole_obs_2':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP2
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 20
    elif args.env_type == 'cartpole_obs_3':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP4
        cae = CAE_cartpole_voxel_2d
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        enforce_bounds = cart_pole_obs.enforce_bounds
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



    # set loss for mpnet
    if args.loss == 'mse':
        #mpnet.loss_f = nn.MSELoss()
        def mse_loss(y1, y2):
            l = (y1 - y2) ** 2
            l = torch.mean(l, dim=0)  # sum alone the batch dimension, now the dimension is the same as input dimension
        loss_f = mse_loss

    elif args.loss == 'l1_smooth':
        #mpnet.loss_f = nn.SmoothL1Loss()
        def l1_smooth_loss(y1, y2):
            l1 = torch.abs(y1 - y2)
            cond = l1 < 1
            l = torch.where(cond, 0.5 * l1 ** 2, l1)
            l = torch.mean(l, dim=0)  # sum alone the batch dimension, now the dimension is the same as input dimension
        loss_f = l1_smooth_loss

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






    mpnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp, loss_f)
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


    # load train and test data
    print('loading...')
    obs, waypoint_dataset, waypoint_targets, env_indices, \
    _, _, _, _ = data_loader.load_train_dataset(N=args.no_env, NP=args.no_motion_paths,
                                                data_folder=args.path_folder, obs_f=True,
                                                direction=args.direction,
                                                dynamics=dynamics, enforce_bounds=enforce_bounds,
                                                system=system, step_sz=step_sz,
                                                num_steps=args.num_steps, multigoal=args.multigoal)
    # randomize the dataset before training
    data=list(zip(waypoint_dataset,waypoint_targets,env_indices))
    random.shuffle(data)
    dataset,targets,env_indices=list(zip(*data))
    dataset = list(dataset)
    targets = list(targets)
    env_indices = list(env_indices)
    dataset = np.array(dataset)
    targets = np.array(targets)
    env_indices = np.array(env_indices)

    # use 5% as validation dataset
    val_len = int(len(dataset) * 0.05)
    val_dataset = dataset[-val_len:]
    val_targets = targets[-val_len:]
    val_env_indices = env_indices[-val_len:]

    dataset = dataset[:-val_len]
    targets = targets[:-val_len]
    env_indices = env_indices[:-val_len]

    # Train the Models
    print('training...')
    if args.loss == 'mse':
        if args.multigoal == 0:
            writer_fname = 'cont_%s_%f_%s_direction_%d_step_%d' % (args.env_type, args.learning_rate, args.opt, args.direction, args.num_steps, )
        else:
            writer_fname = 'cont_%s_%f_%s_direction_%d_step_%d_multigoal' % (args.env_type, args.learning_rate, args.opt, args.direction, args.num_steps, )
    else:
        if args.multigoal == 0:
            writer_fname = 'cont_%s_%f_%s_direction_%d_step_%d_loss_%s' % (args.env_type, args.learning_rate, args.opt, args.direction, args.num_steps, args.loss, )
        else:
            writer_fname = 'cont_%s_%f_%s_direction_%d_step_%d_loss_%s_multigoal' % (args.env_type, args.learning_rate, args.opt, args.direction, args.num_steps, args.loss, )


    writer = SummaryWriter('./runs/'+writer_fname)
    record_i = 0
    val_record_i = 0
    loss_avg_i = 0
    val_loss_avg_i = 0
    loss_avg = 0.
    val_loss_avg = 0.
    loss_steps = 100  # record every 100 loss
    for epoch in range(args.start_epoch+1,args.num_epochs+1):
        print('epoch' + str(epoch))
        val_i = 0
        for i in range(0,len(dataset),args.batch_size):
            print('epoch: %d, training... path: %d' % (epoch, i+1))
            dataset_i = dataset[i:i+args.batch_size]
            targets_i = targets[i:i+args.batch_size]
            env_indices_i = env_indices[i:i+args.batch_size]
            # record
            bi = dataset_i.astype(np.float32)
            print('bi shape:')
            print(bi.shape)
            bt = targets_i
            bi = torch.FloatTensor(bi)
            bt = torch.FloatTensor(bt)

            # edit: disable this for investigation of the good weights for training, and for wrapping
            #bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)


            mpnet.zero_grad()
            bi=to_var(bi)
            bt=to_var(bt)
            if obs is None:
                bobs = None
            else:
                bobs = obs[env_indices_i].astype(np.float32)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            print('before training losses:')
            print(mpnet.loss(mpnet(bi, bobs), bt))
            mpnet.step(bi, bobs, bt)
            print('after training losses:')
            print(mpnet.loss(mpnet(bi, bobs), bt))
            loss = mpnet.loss(mpnet(bi, bobs), bt)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            loss_avg += loss.cpu().data
            loss_avg_i += 1
            if loss_avg_i >= loss_steps:
                loss_avg = loss_avg / loss_avg_i
                writer.add_scalar('train_loss_0', loss_avg[0], record_i)
                writer.add_scalar('train_loss_1', loss_avg[1], record_i)
                writer.add_scalar('train_loss_2', loss_avg[2], record_i)
                writer.add_scalar('train_loss_3', loss_avg[3], record_i)

                record_i += 1
                loss_avg = 0.
                loss_avg_i = 0

            # validation
            # calculate the corresponding batch in val_dataset
            dataset_i = val_dataset[val_i:val_i+args.batch_size]
            targets_i = val_targets[val_i:val_i+args.batch_size]
            env_indices_i = val_env_indices[val_i:val_i+args.batch_size]
            val_i = val_i + args.batch_size
            if val_i > val_len:
                val_i = 0
            # record
            bi = dataset_i.astype(np.float32)
            print('bi shape:')
            print(bi.shape)
            bt = targets_i
            bi = torch.FloatTensor(bi)
            bt = torch.FloatTensor(bt)
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            bi=to_var(bi)
            bt=to_var(bt)
            if obs is None:
                bobs = None
            else:
                bobs = obs[env_indices_i].astype(np.float32)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            loss = mpnet.loss(mpnet(bi, bobs), bt)
            print('validation loss: ', loss.cpu().data)

            val_loss_avg += loss.cpu().data
            val_loss_avg_i += 1
            if val_loss_avg_i >= loss_steps:
                val_loss_avg = val_loss_avg / val_loss_avg_i
                writer.add_scalar('val_loss_0', val_loss_avg[0], val_record_i)
                writer.add_scalar('val_loss_1', val_loss_avg[1], val_record_i)
                writer.add_scalar('val_loss_2', val_loss_avg[2], val_record_i)
                writer.add_scalar('val_loss_3', val_loss_avg[3], val_record_i)
                val_record_i += 1
                val_loss_avg = 0.
                val_loss_avg_i = 0
        # Save the models
        if epoch > 0 and epoch % 50 == 0:
            model_path='kmpnet_epoch_%d_direction_%d_step_%d.pkl' %(epoch, args.direction, args.num_steps)
            #save_state(mpnet, torch_seed, np_seed, py_seed, os.path.join(args.model_path,model_path))
            save_state(mpnet, torch_seed, np_seed, py_seed, os.path.join(model_dir,model_path))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./results/',help='path for saving trained models')
parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
parser.add_argument('--no_val_paths', type=int,default=50,help='number of optimal paths in each environment')
parser.add_argument('--num_steps', type=int, default=20)

# Model parameters
parser.add_argument('--total_input_size', type=int, default=2800+4, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int, default=2800, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int , default=28+4, help='dimension of the input vector')
parser.add_argument('--output_size', type=int , default=2, help='dimension of the input vector')

parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--seen', type=int, default=0, help='seen or unseen? 0 for seen, 1 for unseen')
parser.add_argument('--device', type=int, default=0, help='cuda device')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--path_folder', type=str, default='../data/simple/')
parser.add_argument('--path_file', type=str, default='train')

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--env_type', type=str, default='cartpole', help='environment')
parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--multigoal', type=int, default=0, help='using itermediate nodes as goal or not')


parser.add_argument('--direction', type=int, default=0, help='0: forward, 1: backward')
#parser.add_argument('--opt', type=str, default='Adagrad')
args = parser.parse_args()
print(args)
main(args)
