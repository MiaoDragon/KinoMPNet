"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
def load_train_dataset(N, NP, p_folder, obs_f=None, obc_f=None):
    # obtain the generated paths, and transform into
    # (obc, dataset, targets, env_indices)
    # return list NOT NUMPY ARRAY
    ## TODO: add different folders for obstacle information and path
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}

    # load obs and obc (obc: obstacle point cloud)
    if obs_f is None:
        obs = None
        obc = None
    else:
        file = open(obs_f, 'rb')
        obs = pickle.load(file)
        file = open(obc_f, 'rb')
        obc = pickle.load(file)
    dataset = []
    targets = []
    env_indices = []


    for i in range(N):
        for j in range(NP):
            dir = p_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            file = open(path_file)
            p = pickle.load(file)
            for k in range(len(p)-1):
                for l in range(k+1, len(p)):
                    dataset.append(np.concatenate([p[k], p[l]]))
                    targets.append(p[k+1])
                    env_indices.append(i)
    ## TODO: print out intermediate results to visualize

    #dataset = np.array(dataset)
    #targets = np.array(targets)
    #env_indices = np.array(env_indices)
    return obs, dataset, targets, env_indices


#def load_test_dataset(N, NP, folder):
#    # obtain



def load_test_dataset(N, NP, p_folder, s=0, sp=0, obs_f=None, obc_f=None):
    # obtain the generated paths, and transform into
    # (obc, dataset, targets, env_indices)
    # return list NOT NUMPY ARRAY
    ## TODO: add different folders for obstacle information and path
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}

    # load obs and obc (obc: obstacle point cloud)
    if obs_f is None:
        obs = None
        obc = None
    else:
        file = open(obs_f, 'rb')
        obs = pickle.load(file)
        file = open(obc_f, 'rb')
        obc = pickle.load(file)
    path_env = []
    path_length_env = []
    for i in range(s,N+s):
        paths = []
        path_lengths = []
        for j in range(sp,NP+sp):
            dir = p_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            file = open(path_file)
            p = pickle.load(file)
            paths.append(p)
            path_lengths.append(len(p))
        path_env.append(paths)
        path_length_env.append(path_lengths)
    return obc, obs, path_env, path_length_env
