"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
def load_train_dataset(N, NP, p_folder, p_fname, obs_f=None, obc_f=None):
    # obtain the generated paths, and transform into
    # (obc, dataset, targets, env_indices)
    # return list NOT NUMPY ARRAY
    ## TODO: add different folders for obstacle information and path
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}

    # load obs and obc (obc: obstacle point cloud)
    if obs_f is None:
        obs = np.empty((1,1))
        obc = np.empty((1,1))
    else:
        file = open(obs_f, 'rb')
        obs = pickle.load(file)
        file = open(obc_f, 'rb')
        obc = pickle.load(file)
    dataset = []
    targets = []
    env_indices = []
    for i in range(N):
        file = open(p_folder+str(i)+'/'+p_fname, 'rb')
        paths = pickle.load(file)
        for p in paths:
            for i in range(len(p)-1):
                dataset.append(np.concatenate([p[i], p[-1]]))
                targets.append(p[i+1])
                env_indices.append(i)
    ## TODO: print out intermediate results to visualize
    
    #dataset = np.array(dataset)
    #targets = np.array(targets)
    #env_indices = np.array(env_indices)
    return obc, dataset, targets, env_indices


#def load_test_dataset(N, NP, folder):
#    # obtain
