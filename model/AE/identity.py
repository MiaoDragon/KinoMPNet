import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    def forward(self, x):
        return x
