import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.encoder = nn.Identity()
    def forward(self, x):
        out = self.encoder(x)
        return out