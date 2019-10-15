"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
def load_train_dataset(N, NP, folder):
    # obtain the generated paths, and transform into
    # (obs, dataset, targets, env_indices)
    ## TODO: add different folders for obstacle information and path
    file = open(folder+'train.pkl', 'rb')
    paths = pickle.load(file)
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}
    obs = np.empty((1,1))
    dataset = []
    targets = []
    env_indices = []
    for p in paths:
        for i in range(len(p)-1):
            dataset.append(np.concatenate([p[i], p[-1]]))
            targets.append(p[i+1])
            env_indices.append(0)
    dataset = np.array(dataset)
    targets = np.array(targets)
    env_indices = np.array(env_indices)
    return obs, dataset, targets, env_indices


#def load_test_dataset(N, NP, folder):
#    # obtain
