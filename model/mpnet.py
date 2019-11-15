import torch.nn as nn
import torch
import numpy as np
import copy
"""
this defines the MPNet to be used, which will utilize MLP and AE.
"""
class KMPNet(nn.Module):
    def __init__(self, total_input_size, AE_input_size, mlp_input_size, output_size, CAE, MLP):
        super(KMPNet, self).__init__()
        if CAE is None:
            self.encoder = None
        else:
            self.encoder = CAE.Encoder(AE_input_size, mlp_input_size-(total_input_size-AE_input_size))
        self.mlp = MLP(mlp_input_size, output_size)
        self.mse = nn.MSELoss()
        self.opt = torch.optim.Adagrad(list(self.encoder.parameters())+list(self.mlp.parameters()))
        self.total_input_size = total_input_size
        self.AE_input_size = AE_input_size

    def set_opt(self, opt, lr=1e-2, momentum=None):
        # edit: can change optimizer type when setting
        if momentum is None:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr)
        else:
            self.opt = opt(list(self.encoder.parameters())+list(self.mlp.parameters()), lr=lr, momentum=momentum)

    def forward(self, x, obs):
        # xobs is the input to encoder
        # x is the input to mlp
        if obs is not None:
            z = self.encoder(obs)
            mlp_in = torch.cat((z,x), 1)
        else:
            mlp_in = x
        return self.mlp(mlp_in)

    def loss(self, pred, truth):
        return self.mse(pred, truth)

    def step(self, x, obs, y):
        # given a batch of data, optimize the parameters by one gradient descent step
        # assume here x and y are torch tensors, and have been
        self.zero_grad()
        loss = self.loss(self.forward(x, obs), y)
        loss.backward()
        self.opt.step()
