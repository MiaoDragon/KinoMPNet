import torch

def normalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    #return x
    bound = torch.tensor(bound)
    if len(x[0]) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        x[:,:-2*len(bound)] = x[:,:-2*len(bound)] / bound[0]
        x[:,-2*len(bound):-len(bound)] = x[:,-2*len(bound):-len(bound)] / bound
        x[:,-len(bound):] = x[:,-len(bound):] / bound
    else:
        x = x / bound
    return x
def unnormalize(x, bound):
    # normalize to -1 ~ 1  (bound can be a tensor)
    # x only one dim
    #return x
    bound = torch.tensor(bound)
    if len(x) != len(bound):
        # then the proceding is obstacle
        # don't normalize obstacles
        x[:,:-2*len(bound)] = x[:,:-2*len(bound)] * bound[0]
        x[:,-2*len(bound):-len(bound)] = x[:,-2*len(bound):-len(bound)] * bound
        x[:,-len(bound):] = x[:,-len(bound):] * bound
    else:
        x = x * bound
    return x
