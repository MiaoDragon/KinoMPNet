import torch
import numpy as np
def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    if type(x) is np.ndarray:
        bound = np.array(bound)
    else:
        bound = torch.FloatTensor(bound)
    if len(x.shape) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] / bound
            x[:,-len(bound):] = x[:,-len(bound):] / bound
        else:
            x = x / bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] / bound
            x[-len(bound):] = x[-len(bound):] / bound
        else:
            x = x / bound
    return x
def unnormalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    if type(x) is np.ndarray:
        bound = np.array(bound)
    else:
        bound = torch.FloatTensor(bound)
    if len(x.shape) > 1:
        if len(x[0]) != len(bound):
            x[:,:-len(bound)] = x[:,:-len(bound)] * bound
            x[:,-len(bound):] = x[:,-len(bound):] * bound
        else:
            x = x * bound
    else:
        if len(x) != len(bound):
            x[:-len(bound)] = x[:-len(bound)] * bound
            x[-len(bound):] = x[-len(bound):] * bound
        else:
            x = x * bound
    return x

def IsInCollision(x, obc):
    return False
