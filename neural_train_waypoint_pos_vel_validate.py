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
        pos_indices = [0, 2]
        vel_indices = [1, 3]
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
        pos_indices = [0, 2]
        vel_indices = [1, 3]

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
        pos_indices = [0, 2]
        vel_indices = [1, 3]
        
    elif args.env_type == 'cartpole_obs_4_small':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP3
        cae = CAE_cartpole_voxel_2d
        
        # dynamics: None    -- without integration to dense trajectory
        #dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        dynamics = None
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 20
        pos_indices = np.array([0, 2])
        vel_indices = np.array([1, 3])
    elif args.env_type == 'cartpole_obs_4_big':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        system = _sst_module.PSOPTCartPole()
        mlp = mlp_cartpole.MLP3
        cae = CAE_cartpole_voxel_2d
        
        # dynamics: None    -- without integration to dense trajectory
        #dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
        dynamics = None
        enforce_bounds = cart_pole_obs.enforce_bounds
        step_sz = 0.002
        num_steps = 20
        pos_indices = np.array([0, 2])
        vel_indices = np.array([1, 3])

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
        pos_indices = [0, 1]
        vel_indices = [2, 3]

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
            return l
        loss_f_p = mse_loss
        loss_f_v = mse_loss

    elif args.loss == 'l1_smooth':
        #mpnet.loss_f = nn.SmoothL1Loss()
        def l1_smooth_loss(y1, y2):
            l1 = torch.abs(y1 - y2)
            cond = l1 < 1
            l = torch.where(cond, 0.5 * l1 ** 2, l1)
            l = torch.mean(l, dim=0)  # sum alone the batch dimension, now the dimension is the same as input dimension
            return l
        loss_f_p = l1_smooth_loss
        loss_f_v = l1_smooth_loss

    elif args.loss == 'mse_decoupled':
        def mse_decoupled(y1, y2):
            # for angle terms, wrap it to -pi~pi
            l_0 = torch.abs(y1[:,0] - y2[:,0]) ** 2
            l_1 = torch.abs(y1[:,1] - y2[:,1]) ** 2
            l_2 = torch.abs(y1[:,2] - y2[:,2]) # angular dimension
            l_3 = torch.abs(y1[:,3] - y2[:,3]) ** 2
            cond = (l_2 > 1.0) * (l_2 <= 2.0)
            l_2 = torch.where(cond, 2*1.0-l_2, l_2)
            l_2 = l_2 ** 2
            l_0 = torch.mean(l_0)
            l_1 = torch.mean(l_1)
            l_2 = torch.mean(l_2)
            l_3 = torch.mean(l_3)
            return torch.stack([l_0, l_1, l_2, l_3])
        loss_f_p = mse_decoupled
        loss_f_v = mse_decoupled

    elif args.loss == 'l1_smooth_decoupled':
        
        # this only is for cartpole, need to adapt to other systems
        #TODO
        def l1_smooth_decoupled(y1, y2):
            # for angle terms, wrap it to -pi~pi
            l_0 = torch.abs(y1[:,0] - y2[:,0])
            l_1 = torch.abs(y1[:,1] - y2[:,1]) # angular dimension
            cond = (l_1 > 1.0) * (l_1 <= 2.0)
            l_1 = torch.where(cond, 2*1.0-l_1, l_1)
            
            # then change to l1_smooth_loss
            cond = l_0 < 1
            l_0 = torch.where(cond, 0.5 * l_0 ** 2, l_0)
            cond = l_1 < 1
            l_1 = torch.where(cond, 0.5 * l_1 ** 2, l_1)
            
            l_0 = torch.mean(l_0)
            l_1 = torch.mean(l_1)
            return torch.stack([l_0, l_1])
        def l1_smooth_loss(y1, y2):
            l1 = torch.abs(y1 - y2)
            cond = l1 < 1
            l = torch.where(cond, 0.5 * l1 ** 2, l1)
            l = torch.mean(l, dim=0)  # sum alone the batch dimension, now the dimension is the same as input dimension
            return l
        loss_f_p = l1_smooth_decoupled
        loss_f_v = l1_smooth_loss



    mpnet_pnet = KMPNet(args.total_input_size//2, args.AE_input_size, args.mlp_input_size, args.output_size//2,
                   cae, mlp, loss_f_p)
    mpnet_vnet = KMPNet(args.total_input_size//2, args.AE_input_size, args.mlp_input_size, args.output_size//2,
                   cae, mlp, loss_f_v)
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
        load_net_state(mpnet_pnet, os.path.join(model_dir, model_pnet_path))
        load_net_state(mpnet_vnet, os.path.join(model_dir, model_vnet_path))

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

        if args.opt == 'Adagrad':
            mpnet_pnet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpnet_pnet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpnet_pnet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            mpnet_pnet.set_opt(torch.optim.ASGD, lr=args.learning_rate)

            
        if args.opt == 'Adagrad':
            mpnet_vnet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpnet_vnet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpnet_vnet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            mpnet_vnet.set_opt(torch.optim.ASGD, lr=args.learning_rate)

            
            
        if args.start_epoch > 0:
            #load_opt_state(mpnet, os.path.join(args.model_path, model_path))
            load_opt_state(mpnet_pnet, os.path.join(model_dir, model_path))
            load_opt_state(mpnet_vnet, os.path.join(model_dir, model_path))


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
    dataset = np.array(dataset)
    targets = np.array(targets)
    print(np.concatenate([pos_indices, pos_indices+args.total_input_size//2]))
    p_dataset = dataset[:, np.concatenate([pos_indices, pos_indices+args.total_input_size//2])]
    v_dataset = dataset[:, np.concatenate([vel_indices, vel_indices+args.total_input_size//2])]
    p_targets = targets[:,pos_indices]
    v_targets = targets[:,vel_indices]   # this is only for cartpole
                                # TODO: add string for choosing env

    #p_targets = list(p_targets)
    #v_targets = list(v_targets)
    #targets = list(targets)
    #env_indices = list(env_indices)
    dataset = np.array(dataset)
    #targets = np.array(targets)
    env_indices = np.array(env_indices)

    # use 5% as validation dataset
    val_len = int(len(dataset) * 0.05)
    val_p_dataset = p_dataset[-val_len:]
    val_v_dataset = v_dataset[-val_len:]
    val_p_targets = p_targets[-val_len:]
    val_v_targets = v_targets[-val_len:]
    val_env_indices = env_indices[-val_len:]
    print('val_p_dataset size:')
    print(val_p_dataset.shape)
    print('val_v_dataset size:')
    print(val_v_dataset.shape)
    print('val_p_targets size:')
    print(val_p_targets.shape)
    print('val_v_targets size:')
    print(val_v_targets.shape)
    print('val_env_indices size:')
    print(val_env_indices.shape)


    p_dataset = p_dataset[:-val_len]
    v_dataset = v_dataset[:-val_len]
    p_targets = p_targets[:-val_len]
    v_targets = v_targets[:-val_len]
    env_indices = env_indices[:-val_len]
    print('p_dataset size:')
    print(p_dataset.shape)
    print('v_dataset size:')
    print(v_dataset.shape)
    print('p_targets size:')
    print(p_targets.shape)
    print('v_targets size:')
    print(v_targets.shape)
    print('env_indices size:')
    print(env_indices.shape)

    # Train the Models
    print('training...')

    record_i = 0
    val_record_i = 0
    p_loss_avg_i = 0
    p_val_loss_avg_i = 0
    p_loss_avg = 0.
    p_val_loss_avg = 0.
    v_loss_avg_i = 0
    v_val_loss_avg_i = 0
    v_loss_avg = 0.
    v_val_loss_avg = 0.

    loss_steps = 100  # record every 100 loss
    
    
    world_size = np.array(args.world_size)
    pos_world_size = list(world_size[pos_indices])
    vel_world_size = list(world_size[vel_indices])
    

    
    for epoch in range(args.start_epoch+1,args.num_epochs+1):
        print('epoch' + str(epoch))
        val_i = 0
        for i in range(0,len(p_dataset),args.batch_size):
            print('epoch: %d, training... path: %d' % (epoch, i+1))
            p_dataset_i = p_dataset[i:i+args.batch_size]
            v_dataset_i = v_dataset[i:i+args.batch_size]
            p_targets_i = p_targets[i:i+args.batch_size]
            v_targets_i = v_targets[i:i+args.batch_size]
            env_indices_i = env_indices[i:i+args.batch_size]
            
            
            print("p_dataset_i:")
            print(p_dataset_i)
            print("v_dataset_i:")
            print(v_dataset_i)
            print("p_targets_i:")
            print(p_targets_i)
            print("v_targets_i:")
            print(v_targets_i)
            print("env_indices_i:")
            print(env_indices_i)
            
            # record
            p_bi = p_dataset_i.astype(np.float32)
            v_bi = v_dataset_i.astype(np.float32)
            print('p_bi shape:')
            print(p_bi.shape)
            print('v_bi shape:')
            print(v_bi.shape)
            p_bt = p_targets_i
            v_bt = v_targets_i
            p_bi = torch.FloatTensor(p_bi)
            v_bi = torch.FloatTensor(v_bi)
            p_bt = torch.FloatTensor(p_bt)
            v_bt = torch.FloatTensor(v_bt)

            # edit: disable this for investigation of the good weights for training, and for wrapping
            p_bi, v_bi, p_bt, v_bt = normalize(p_bi, pos_world_size), normalize(v_bi, vel_world_size), normalize(p_bt, pos_world_size), normalize(v_bt, vel_world_size)
            
            print('after normalization:')
            print('p_bi:')
            print(p_bi)
            print('v_bi:')
            print(v_bi)
            print('p_bt:')
            print(p_bt)
            print('v_bt:')
            print(v_bt)

            mpnet_pnet.zero_grad()
            mpnet_vnet.zero_grad()

            p_bi=to_var(p_bi)
            v_bi=to_var(v_bi)
            p_bt=to_var(p_bt)
            v_bt=to_var(v_bt)

            if obs is None:
                bobs = None
            else:
                bobs = obs[env_indices_i].astype(np.float32)
                print('bobs:')
                print(bobs)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            print('-------pnet-------')
            print('before training losses:')
            print(mpnet_pnet.loss(mpnet_pnet(p_bi, bobs), p_bt))
            mpnet_pnet.step(p_bi, bobs, p_bt)
            print('after training losses:')
            print(mpnet_pnet.loss(mpnet_pnet(p_bi, bobs), p_bt))
            p_loss = mpnet_pnet.loss(mpnet_pnet(p_bi, bobs), p_bt)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            p_loss_avg += p_loss.cpu().data
            p_loss_avg_i += 1
            
            print('-------vnet-------')
            print('before training losses:')
            print(mpnet_vnet.loss(mpnet_vnet(v_bi, bobs), v_bt))
            mpnet_vnet.step(v_bi, bobs, v_bt)
            print('after training losses:')
            print(mpnet_vnet.loss(mpnet_vnet(v_bi, bobs), v_bt))
            v_loss = mpnet_vnet.loss(mpnet_vnet(v_bi, bobs), v_bt)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            v_loss_avg += v_loss.cpu().data
            v_loss_avg_i += 1
            
               
            # validation
            # calculate the corresponding batch in val_dataset
            p_dataset_i = val_p_dataset[val_i:val_i+args.batch_size]
            v_dataset_i = val_v_dataset[val_i:val_i+args.batch_size]

            p_targets_i = val_p_targets[val_i:val_i+args.batch_size]
            v_targets_i = val_v_targets[val_i:val_i+args.batch_size]

            env_indices_i = val_env_indices[val_i:val_i+args.batch_size]
            
            print('val_i:')
            print(val_i)
            print('validation p_dataset_i:')
            print(p_dataset_i)
            print('validation v_dataset_i:')
            print(v_dataset_i)
            print('validation p_targets_i:')
            print(p_targets_i)
            print('validation v_targets_i:')
            print(v_targets_i)

            val_i = val_i + args.batch_size


            if val_i > val_len:
                val_i = 0
            # record
            p_bi = p_dataset_i.astype(np.float32)
            v_bi = v_dataset_i.astype(np.float32)

            print('p_bi shape:')
            print(p_bi.shape)
            print('v_bi shape:')
            print(v_bi.shape)

            p_bt = p_targets_i
            v_bt = v_targets_i
            p_bi = torch.FloatTensor(p_bi)
            v_bi = torch.FloatTensor(v_bi)

            p_bt = torch.FloatTensor(p_bt)
            v_bt = torch.FloatTensor(v_bt)
            p_bi, v_bi, p_bt, v_bt = normalize(p_bi, pos_world_size), normalize(v_bi, vel_world_size), normalize(p_bt, pos_world_size), normalize(v_bt, vel_world_size)
            p_bi=to_var(p_bi)
            v_bi=to_var(v_bi)
            p_bt=to_var(p_bt)
            v_bt=to_var(v_bt)

            if obs is None:
                bobs = None
            else:
                bobs = obs[env_indices_i].astype(np.float32)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            print('-------pnet loss--------')
            p_loss = mpnet_pnet.loss(mpnet_pnet(p_bi, bobs), p_bt)
            print('validation loss: ' % (p_loss.cpu().data))

            p_val_loss_avg += p_loss.cpu().data
            p_val_loss_avg_i += 1

            print('-------vnet loss--------')
            v_loss = mpnet_vnet.loss(mpnet_vnet(v_bi, bobs), v_bt)
            print('validation loss: ' % (v_loss.cpu().data))

            v_val_loss_avg += v_loss.cpu().data
            v_val_loss_avg_i += 1


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
