"""
This implements the Kinodynamic Planning using MPNet, by using MPNet
to generate random samples, that will guide the SST algorithm.
"""
import torch
import model.AE.identity as cae_identity
from model.mlp import MLP
from model import mlp_acrobot
from model.AE import CAE_acrobot_voxel_2d
from model.mpnet import KMPNet
from tools import data_loader
from tools.utility import *
from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs
import argparse
import numpy as np
import random
import os

from tensorboardX import SummaryWriter

def main(args):
    #global hl
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    # environment setting
    cae = cae_identity
    mlp = MLP
    if args.env_type == 'pendulum':
        normalize = pendulum.normalize
        unnormalize = pendulum.unnormalize
        obs_file = None
        obc_file = None
    elif args.env_type == 'cartpole':
        normalize = cart_pole.normalize
        unnormalize = cart_pole.unnormalize
        obs_file = None
        obc_file = None
    elif args.env_type == 'cartpole_obs':
        normalize = cart_pole_obs.normalize
        unnormalize = cart_pole_obs.unnormalize
        obs_file = args.obs_file
        obc_file = args.obc_file
    elif args.env_type == 'acrobot_obs':
        normalize = acrobot_obs.normalize
        unnormalize = acrobot_obs.unnormalize
        obs_file = args.obs_file
        obc_file = args.obc_file
        mlp = mlp_acrobot.MLP
        cae = CAE_acrobot_voxel_2d


    mpnet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   cae, mlp)
    # load net
    # load previously trained model if start epoch > 0
    model_path='kmpnet_epoch_%d_direction_%d.pkl' %(args.start_epoch, args.direction)
    torch_seed, np_seed, py_seed = 0, 0, 0
    if args.start_epoch > 0:
        load_net_state(mpnet, os.path.join(args.model_path, model_path))
        torch_seed, np_seed, py_seed = load_seed(os.path.join(args.model_path, model_path))
        # set seed after loading
        torch.manual_seed(torch_seed)
        np.random.seed(np_seed)
        random.seed(py_seed)

    if torch.cuda.is_available():
        mpnet.cuda()
        mpnet.mlp.cuda()
        mpnet.encoder.cuda()
        if args.opt == 'Adagrad':
            mpNet.set_opt(torch.optim.Adagrad, lr=args.learning_rate)
        elif args.opt == 'Adam':
            mpNet.set_opt(torch.optim.Adam, lr=args.learning_rate)
        elif args.opt == 'SGD':
            mpNet.set_opt(torch.optim.SGD, lr=args.learning_rate, momentum=0.9)
        elif args.opt == 'ASGD':
            mpNet.set_opt(torch.optim.ASGD, lr=args.learning_rate)
    if args.start_epoch > 0:
        load_opt_state(mpnet, os.path.join(args.model_path, model_path))

    # load train and test data
    print('loading...')
    obs, dataset, targets, env_indices = data_loader.load_train_dataset(N=args.no_env, NP=args.no_motion_paths,
                                                                        p_folder=args.path_folder,
                                                                        obs_f=obs_file, obc_f=obc_file, direction=arg.direction)
    # randomize the dataset before training
    data=list(zip(dataset,targets,env_indices))
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
    writer_fname = 'cont_%s_%f_%s_direction_%d' % (args.env_type, args.learning_rate, args.opt, args.direction)
    writer = SummaryWriter('./runs/'+writer_fname)
    record_i = 0
    train_losses = []
    val_losses = []
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
            bi, bt = normalize(bi, args.world_size), normalize(bt, args.world_size)
            mpnet.zero_grad()
            bi=to_var(bi)
            bt=to_var(bt)
            if obs is None:
                bobs = None
            else:
                bobs = obs[env_indices_i, :args.AE_input_size].astype(np.float32)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            print('before training losses:')
            print(mpnet.loss(mpnet(bi, bobs), bt))
            mpnet.step(bi, bobs, bt)
            print('after training losses:')
            print(mpnet.loss(mpnet(bi, bobs), bt))
            loss = mpnet.loss(mpnet(bi, bobs), bt)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            record_i += 1
            writer.add_scalar('train_loss', loss.data, record_i)
            train_losses.append(loss.data.numpy())

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
                bobs = obs[env_indices_i, :args.AE_input_size].astype(np.float32)
                bobs = torch.FloatTensor(bobs)
                bobs = to_var(bobs)
            loss = mpnet.loss(mpnet(bi, bobs), bt)
            print('validation loss: %f' % (loss.data))
            writer.add_scalar('val_loss', loss.data, val_record_i)
            #update_line(hl, ax, [i//args.batch_size, loss.data.numpy()])
            val_losses.append(loss.data.numpy())

        # Save the models
        if epoch > 0:
            model_path='kmpnet_epoch_%d_direction_%d.pkl' %(epoch, args.direction)
            save_state(mpnet, torch_seed, np_seed, py_seed, os.path.join(args.model_path,model_path))
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./results/',help='path for saving trained models')
parser.add_argument('--no_env', type=int, default=100,help='directory for obstacle images')
parser.add_argument('--no_motion_paths', type=int,default=4000,help='number of optimal paths in each environment')
parser.add_argument('--no_val_paths', type=int,default=50,help='number of optimal paths in each environment')

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
parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--env_type', type=str, default='cartpole', help='environment')
parser.add_argument('--world_size', nargs='+', type=float, default=20., help='boundary of world')
parser.add_argument('--opt', type=str, default='Adagrad')
parser.add_argument('--direction', type=int, default=0, help='0: forward, 1: backward')
#parser.add_argument('--opt', type=str, default='Adagrad')
args = parser.parse_args()
print(args)
main(args)
