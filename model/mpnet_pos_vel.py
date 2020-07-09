import torch
import torch.nn as nn
class PosVelKMPNet(nn.Module):
    def __init__(self, total_input_size, PNet, VNet):
        super(PosVelKMPNet, self).__init__()
        self.pnet = PNet
        self.vnet = VNet

    def forward(self, x, obs):
        # xobs is the input to encoder
        # x is the input to mlp
        if obs is not None:
            z = self.encoder(obs)
            mlp_in = torch.cat((z,x), 1)
        else:
            mlp_in = x
        # for cartpole, the output is [pos0, vel0, pos1, vel1]
        pos = self.pnet(mlp_in)
        vel = self.vnet(mlp_in)
        return torch.stack([pos[:,0], vel[:,0], pos[:,1], vel[:,1]]
