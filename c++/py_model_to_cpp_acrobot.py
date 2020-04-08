from __future__ import division
"""
convert from trained python module to C++ pytorch module.
Since we can't use Pickle in C++ to load trained model.
Adapted from work from Anthony Simeonov.
"""
import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('../')
import argparse
from model.mpnet import KMPNet
from model import mlp_acrobot
from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3
from tools import data_loader
from tools.utility import *
from plan_utility import cart_pole, cart_pole_obs, pendulum, acrobot_obs
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from sparse_rrt import _sst_module
import os

class Encoder_acrobot(nn.Module):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    def __init__(self, input_size=32, output_size=128):
        super(Encoder, self).__init__()
        input_size = [input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[6,6], stride=[2,2]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3,3], stride=[2,2]),
            nn.PReLU(),
            #nn.MaxPool2d(2, stride=2)
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        print('length of the output of one encoder')
        print(first_fc_in_features)
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
    @torch.jit.script_method
    def forward(self, x):
        x = self.encoder(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

class Encoder_acrobot_Annotated(torch.jit.ScriptModule):
    # ref: https://github.com/lxxue/voxnet-pytorch/blob/master/models/voxnet.py
    __constants__ = ['encoder', 'head', 'device']
    def __init__(self, input_size=32, output_size=64):
        super(Encoder_acrobot_Annotated, self).__init__()
        input_size = [input_size, input_size]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=[6,6], stride=[2,2]),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=[3,3], stride=[2,2]),
            nn.PReLU(),
            #nn.MaxPool2d(2, stride=2)
        )
        x = self.encoder(torch.autograd.Variable(torch.rand([1, 1] + input_size)))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n
        print('length of the output of one encoder')
        print(first_fc_in_features)
        self.head = nn.Sequential(
            nn.Linear(first_fc_in_features, 128),
            nn.PReLU(),
            nn.Linear(128, output_size)
        )
        self.device = torch.device('cuda')
    @torch.jit.script_method
    def forward(self, x):
        x = self.encoder(x)
        #x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

class MLP_acrobot(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP_acrobot, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 2048), nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(2048, 1024), nn.PReLU())
        self.fc3 = nn.Sequential(nn.Linear(1024, 896), nn.PReLU())
        self.fc4 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
        self.fc5 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
        self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
        self.fc7 = nn.Sequential(nn.Linear(128, 32), nn.PReLU())
        self.fc8 = nn.Linear(32, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        return x


class MLP_acrobot_Annotated(torch.jit.ScriptModule):
    __constants__ = ['fc1','fc2','fc3','fc4','fc5','fc6','fc7', 'device']
    def __init__(self, input_size, output_size):
        super(MLP_acrobot_Annotated, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 2048), nn.PReLU())
        self.fc2 = nn.Sequential(nn.Linear(2048, 1024), nn.PReLU())
        self.fc3 = nn.Sequential(nn.Linear(1024, 896), nn.PReLU())
        self.fc4 = nn.Sequential(nn.Linear(896, 512), nn.PReLU())
        self.fc5 = nn.Sequential(nn.Linear(512, 256), nn.PReLU())
        self.fc6 = nn.Sequential(nn.Linear(256, 128), nn.PReLU())
        self.fc7 = nn.Sequential(nn.Linear(128, 32), nn.PReLU())
        self.fc8 = nn.Linear(32, output_size)
        
        self.device = torch.device('cuda')
    @torch.jit.script_method
    def forward(self, x):
        prob = 0.5

        p = 1 - prob
        scale = 1.0/p
        drop1 = (scale)*torch.bernoulli(torch.full((1, 2048), p)).to(device=self.device)
        drop2 = (scale)*torch.bernoulli(torch.full((1, 1024), p)).to(device=self.device)
        drop3 = (scale)*torch.bernoulli(torch.full((1, 896), p)).to(device=self.device)
        drop4 = (scale)*torch.bernoulli(torch.full((1, 512), p)).to(device=self.device)
        drop5 = (scale)*torch.bernoulli(torch.full((1, 256), p)).to(device=self.device)
        drop6 = (scale)*torch.bernoulli(torch.full((1, 128), p)).to(device=self.device)

        out1 = self.fc1(x)
        out1 = torch.mul(out1, drop1)

        out2 = self.fc2(out1)
        out2 = torch.mul(out2, drop2)

        out3 = self.fc3(out2)
        out3 = torch.mul(out3, drop3)

        out4 = self.fc4(out3)
        out4 = torch.mul(out4, drop4)

        out5 = self.fc5(out4)
        out5 = torch.mul(out5, drop5)

        out6 = self.fc6(out5)
        out6 = torch.mul(out6, drop6)

        out7 = self.fc7(out6)

        out8 = self.fc8(out7)

        return out7

def copyMLP(MLP_to_copy, mlp_weights):
    # this function is where weights are manually copied from the originally trained
    # MPNet models (which have different naming convention for the weights that doesn't
    # work with manual dropout implementation) into the models defined in this script
    # which have the new layer naming convention

    # mlp_weights is just a state_dict() with the good model weights, not loaded into a particular model yet
    # MLP_to_copy is one of the MLP_Python models defined above (depending on 1.0 or 2.0)
    print(MLP_to_copy.state_dict().keys())
    MLP_to_copy.state_dict()['fc1.0.weight'].copy_(mlp_weights['fc.0.weight'])
    MLP_to_copy.state_dict()['fc2.0.weight'].copy_(mlp_weights['fc.3.weight'])
    MLP_to_copy.state_dict()['fc3.0.weight'].copy_(mlp_weights['fc.6.weight'])
    MLP_to_copy.state_dict()['fc4.0.weight'].copy_(mlp_weights['fc.9.weight'])
    MLP_to_copy.state_dict()['fc5.0.weight'].copy_(mlp_weights['fc.12.weight'])
    MLP_to_copy.state_dict()['fc6.0.weight'].copy_(mlp_weights['fc.15.weight'])
    MLP_to_copy.state_dict()['fc7.0.weight'].copy_(mlp_weights['fc.18.weight'])
    MLP_to_copy.state_dict()['fc8.weight'].copy_(mlp_weights['fc.20.weight'])

    MLP_to_copy.state_dict()['fc1.0.bias'].copy_(mlp_weights['fc.0.bias'])
    MLP_to_copy.state_dict()['fc2.0.bias'].copy_(mlp_weights['fc.3.bias'])
    MLP_to_copy.state_dict()['fc3.0.bias'].copy_(mlp_weights['fc.6.bias'])
    MLP_to_copy.state_dict()['fc4.0.bias'].copy_(mlp_weights['fc.9.bias'])
    MLP_to_copy.state_dict()['fc5.0.bias'].copy_(mlp_weights['fc.12.bias'])
    MLP_to_copy.state_dict()['fc6.0.bias'].copy_(mlp_weights['fc.15.bias'])
    MLP_to_copy.state_dict()['fc7.0.bias'].copy_(mlp_weights['fc.18.bias'])
    MLP_to_copy.state_dict()['fc8.bias'].copy_(mlp_weights['fc.20.bias'])

    # PReLU
    MLP_to_copy.state_dict()['fc1.1.weight'].copy_(mlp_weights['fc.1.weight'])
    MLP_to_copy.state_dict()['fc2.1.weight'].copy_(mlp_weights['fc.4.weight'])
    MLP_to_copy.state_dict()['fc3.1.weight'].copy_(mlp_weights['fc.7.weight'])
    MLP_to_copy.state_dict()['fc4.1.weight'].copy_(mlp_weights['fc.10.weight'])
    MLP_to_copy.state_dict()['fc5.1.weight'].copy_(mlp_weights['fc.13.weight'])
    MLP_to_copy.state_dict()['fc6.1.weight'].copy_(mlp_weights['fc.16.weight'])
    MLP_to_copy.state_dict()['fc7.1.weight'].copy_(mlp_weights['fc.19.weight'])
    return MLP_to_copy

def main(args):
    # Set this value to export models for continual learning or batch training

    system = _sst_module.PSOPTAcrobot()
    #dynamics = acrobot_obs.dynamics
    dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)
    enforce_bounds = acrobot_obs.enforce_bounds
    step_sz = 0.02
    num_steps = 20

    # Get the right architecture which was used for continual learning
    mlp = mlp_acrobot.MLP
    CAE = CAE_acrobot_voxel_2d
    # make the big model
    mpNet = KMPNet(args.total_input_size, args.AE_input_size, args.mlp_input_size, args.output_size,
                   CAE, mlp)
    # The model that performed well originally, load into the big end2end model
    model_dir = args.model_dir
    model_dir = model_dir+"acrobot_obs_lr%f_%s/" % (args.learning_rate, args.opt)
    model_path='kmpnet_epoch_%d_direction_%d.pkl' %(args.start_epoch, args.direction)
    load_net_state(mpNet, os.path.join(model_dir, model_path))



    # Get the weights from this model and create a copy of the weights in mlp_weights (to be copied over)
    MLP2 = mpNet.mlp
    MLP2.cuda()
    mlp_weights = MLP2.state_dict()

    # Save a copy of the encoder's state_dict() for loading into the annotated encoder later on
    encoder_to_copy = mpNet.encoder
    encoder_to_copy.cuda()
    #encoder_to_copy.cuda()
    torch.save(encoder_to_copy.state_dict(), 'acrobot_encoder_save.pkl')

    # do everything for the MLP on the GPU
    device = torch.device('cuda:%d'%(args.device))

    encoder = Encoder_acrobot_Annotated(args.AE_input_size, args.mlp_input_size-args.total_input_size)
    encoder.cuda()
    #encoder.cuda()
    # Create the annotated model
    MLP = MLP_acrobot_Annotated(args.mlp_input_size,args.output_size)
    MLP.cuda()

    # Create the python model with the new layer names
    MLP_to_copy = MLP_acrobot(args.mlp_input_size,args.output_size)
    MLP_to_copy.cuda()

    # Copy over the mlp_weights into the Python model with the new layer names
    MLP_to_copy = copyMLP(MLP_to_copy, mlp_weights)

    print("Saving models...")

    # Load the encoder weights onto the gpu and then save the Annotated model
    encoder.load_state_dict(torch.load('acrobot_encoder_save.pkl', map_location=device))
    encoder.save("acrobot_encoder_annotated_test_cpu.pt")

    # Save the Python model with the weights copied over and the new layer names in a temp file
    torch.save(MLP_to_copy.state_dict(), 'acrobot_mlp_no_dropout.pkl')

    # Because the layer names now match, can immediately load this state_dict() into the annotated model and then save it
    #MLP.load_state_dict(torch.load('mlp_no_dropout.pkl', map_location=device))
    MLP.load_state_dict(torch.load('acrobot_mlp_no_dropout.pkl', map_location=device))

    MLP.save("acrobot_mlp_annotated_test_gpu.pt")

    """
    # Everything from here below just tests both models to see if the outputs match
    obs, waypoint_dataset, waypoint_targets, env_indices, \
    _, _, _, _ = data_loader.load_train_dataset(N=1, NP=1,
                                                data_folder=args.data_path, obs_f=True,
                                                direction=1,
                                                dynamics=dynamics, enforce_bounds=enforce_bounds,
                                                system=system, step_sz=step_sz, num_steps=args.num_steps)


    # write test case
    obs_test = np.array([0.1,1.2,3.0,2.5,1.4,5.2,3.4,-1.])
    #obs_test = obs_test.reshape((1,2,2,2))
    np.savetxt('obs_voxel_test.txt', obs_test, delimiter='\n', fmt='%f')

    # write obstacle to flattened vector representation, then later be loaded in the C++
    obs_out = obs.flatten()
    np.savetxt('obs_voxel.txt', obs_out, delimiter='\n', fmt='%f')


    obs = torch.from_numpy(obs).type(torch.FloatTensor)
    obs = Variable(obs)
    # h = mpNet.encoder(obs)
    h = encoder(obs)
    path_data = np.array([-0.08007369,  0.32780212, -0.01338363,  0.00726194, 0.00430644, -0.00323558,
                       0.18593094,  0.13094018, 0.18499476, 0.3250918, 0.52175426, 0.07388325, -0.49999127, 0.52322733])
    path_data = np.array([path_data])
    path_data = torch.from_numpy(path_data).type(torch.FloatTensor)

    test_input = torch.cat((path_data, h.data.cpu()), dim=1).cuda()  # for MPNet1.0
    test_input = Variable(test_input)
    for i in range(5):
        test_output = mpNet.mlp(test_input)
        test_output_save = MLP(test_input)
        print("output %d: " % i)
        print(test_output.data)
        print(test_output_save.data)

    """
parser = argparse.ArgumentParser()
# for training
parser.add_argument('--model_path', type=str, default='./results/',help='path for saving trained models')
parser.add_argument('--model_dir', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/',help='path for saving trained models')
parser.add_argument('--num_steps', type=int, default=20)

# Model parameters
parser.add_argument('--total_input_size', type=int, default=8, help='dimension of total input')
parser.add_argument('--AE_input_size', type=int, default=32, help='dimension of input to AE')
parser.add_argument('--mlp_input_size', type=int, default=136, help='dimension of the input vector')
parser.add_argument('--output_size', type=int, default=4, help='dimension of the input vector')

parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--device', type=int, default=0, help='cuda device')
parser.add_argument('--data_folder', type=str, default='./data/acrobot_obs/')
parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')

parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=100, help='rehersal on how many data (not path)')
parser.add_argument('--path_folder', type=str, default='../data/simple/')
parser.add_argument('--path_file', type=str, default='train')

parser.add_argument('--start_epoch', type=int, default=5000)
parser.add_argument('--opt', type=str, default='SGD')
parser.add_argument('--direction', type=int, default=0, help='0: forward, 1: backward')


args = parser.parse_args()
print(args)
main(args)
