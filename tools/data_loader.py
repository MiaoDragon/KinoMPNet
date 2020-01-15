"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
def load_train_dataset(N, NP, data_folder, direction=0):
    # obtain the generated paths, and transform into
    # (obc, dataset, targets, env_indices)
    # return list NOT NUMPY ARRAY
    ## TODO: add different folders for obstacle information and path
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}
    # direction: 0 -- forward;  1 -- backward

    # load obs and obc (obc: obstacle point cloud)
    if obs_f is None:
        obs = None
        obc = None
        obs_list = None
        obc_list = None
    else:
        obs_list = []
        obc_list = []
        for i in range(N):
            file = open(data_folder+'obs_%d.pkl' % (i), 'rb')
            obs = pickle.load(file)
            file = open(data_folder+'obc_%d.pkl' % (i), 'rb')
            obc = pickle.load(file)
            obc = pointcloud_to_voxel(obc, voxel_size=[32,32]).reshape(-1,1,32,32)
            obs_list.append(obs)
            obc_list.append(obc)
    dataset = []
    targets = []
    env_indices = []


    for i in range(N):
        for j in range(NP):
            dir = data_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            file = open(path_file)
            p = pickle.load(file)
            if direction == 1:
                # backward
                p = np.flip(p, axis=0)
            for k in range(len(p)-1):
                for l in range(k+1, len(p)):
                    dataset.append(np.concatenate([p[k], p[l]]))
                    targets.append(p[k+1])
                    env_indices.append(i)
    ## TODO: print out intermediate results to visualize

    #dataset = np.array(dataset)
    #targets = np.array(targets)
    #env_indices = np.array(env_indices)
    return obc_list, dataset, targets, env_indices


#def load_test_dataset(N, NP, folder):
#    # obtain



def load_test_dataset(N, NP, data_folder, s=0, sp=0):
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
        obs_list = []
        obc_list = []
        for i in range(N):
            file = open(data_folder+'obs_%d.pkl' % (i), 'rb')
            obs = pickle.load(file)
            file = open(data_folder+'obc_%d.pkl' % (i), 'rb')
            obc = pickle.load(file)
            obc = pointcloud_to_voxel(obc, voxel_size=[32,32]).reshape(-1,1,32,32)
            obs_list.append(obs)
            obc_list.append(obc)
    path_env = []
    path_length_env = []
    for i in range(s,N+s):
        paths = []
        path_lengths = []
        for j in range(sp,NP+sp):
            dir = data_folder+str(i)+'/'
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



def pcd_to_voxel2d(points, voxel_size=(24, 24), padding_size=(32, 32)):
    voxels = [voxelize(points[i], voxel_size, padding_size) for i in range(len(points))]
    # return size: BxV*V*V
    return np.array(voxels)

def voxelize2d(points, voxel_size=(24, 24), padding_size=(32, 32), resolution=0.05):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.

    Args:
    `points`: pointcloud in 3D numpy.ndarray (shape: N * 3)
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters

    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    # calculate resolution based on boundary
    if abs(resolution) < sys.float_info.epsilon:
        print('error input, resolution should not be zero')
        return None, None

    """
    here the point cloud is centerized, and each dimension uses a different resolution
    """
    OCCUPIED = 1
    FREE = 0
    resolution = [(points[:,i].max() - points[:,i].min()) / voxel_size[i] for i in range(2)]
    resolution = np.array(resolution)
    #resolution = np.max(res)
    # remove all non-numeric elements of the said array
    points = points[np.logical_not(np.isnan(points).any(axis=1))]

    # filter outside voxel_box by using passthrough filter
    # TODO Origin, better use centroid?
    origin = (np.min(points[:, 0]), np.min(points[:, 1]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    #points[:, 2] -= origin[2]
    # logical condition index
    x_logical = np.logical_and((points[:, 0] < voxel_size[0] * resolution[0]), (points[:, 0] >= 0))
    y_logical = np.logical_and((points[:, 1] < voxel_size[1] * resolution[1]), (points[:, 1] >= 0))
    #z_logical = np.logical_and((points[:, 2] < voxel_size[2] * resolution[2]), (points[:, 2] >= 0))
    xy_logical = np.logical_and(x_logical, y_logical)
    #xyz_logical = np.logical_and(x_logical, np.logical_and(y_logical, z_logical))
    #inside_box_points = points[xyz_logical]
    inside_box_points = points[xy_logical]
    # init voxel grid with zero padding_to_size=(32*32*32) and set the occupany grid
    voxels = np.zeros(padding_size)
    # centerlize to padding box
    center_points = inside_box_points + (padding_size[0] - voxel_size[0]) * resolution / 2
    # TODO currently just use the binary hit grid
    x_idx = (center_points[:, 0] / resolution[0]).astype(int)
    y_idx = (center_points[:, 1] / resolution[1]).astype(int)
    #z_idx = (center_points[:, 2] / resolution[2]).astype(int)
    #voxels[x_idx, y_idx, z_idx] = OCCUPIED
    voxels[x_idx, y_idx] = OCCUPIED
    return voxels
    #return voxels, inside_box_points
