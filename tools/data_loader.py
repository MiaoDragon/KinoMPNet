"""
This implements data loader for both training and testing procedures.
"""
import pickle
import numpy as np
import sys
def preprocess(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz, num_steps):
    p_start = data_path[0]
    detail_paths = [p_start]
    detail_controls = []
    detail_costs = []
    state = [p_start]
    control = []
    cost = []
    for k in range(len(data_control)):
        #state_i.append(len(detail_paths)-1)
        max_steps = int(np.round(data_cost[k]/step_sz))
        accum_cost = 0.
        ## modify it because of small difference between data and actual propagation
        #p_start = data_path[k]
        #state[-1] = data_path[k]
        for step in range(1,max_steps+1):
            p_start = dynamics(p_start, data_control[k], step_sz)
            accum_cost += step_sz
            if (step % 1 == 0) or (step == max_steps):
                state.append(p_start)
                control.append(data_control[k])
                cost.append(accum_cost)
                accum_cost = 0.
    
    # new method: don't care if intermediate nodes have the same control or not
    # take every num_steps after the entire path is stored
    state = state[::num_steps]

    # if last node is not the same as goal, then add goal
    if np.linalg.norm(np.array(state[-1]) - np.array(data_path[-1])) > 1e-3:
        state.append(data_path[-1])

    #control = control[::num_steps]  # this data is wrong
    #cost = cost[::num_steps]   # this data is wrong
    
    #state[-1] = data_path[-1]
    return state, control, cost

def load_train_dataset(N, NP, data_folder, obs_f=None, direction=0, dynamics=None, enforce_bounds=None, system=None, step_sz=0.02, num_steps=20, multigoal=0):
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
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obs = p.load()
            #obs = pickle.load(file)
            file = open(data_folder+'obc_%d.pkl' % (i), 'rb')
            #obc = pickle.load(file)
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obc = p.load()
            # concatenate on the first direction (each obs has a different index)
            obc = obc.reshape(-1, 2)
            obs_list.append(obs)
            obc_list.append(obc)
        obc_list = np.array(obc_list)
        obc_list = pcd_to_voxel2d(obc_list, voxel_size=[32,32]).reshape(-1,1,32,32)

    waypoint_dataset = []
    waypoint_targets = []
    env_indices = []
    u_init_dataset = []  # (start, goal) -> control
    t_init_dataset = []  # (start, goal) -> dt
    u_init_targets = []
    t_init_targets = []

    for i in range(N):
        print('loading... env: %d' % (i))
        for j in range(NP):
            dir = data_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            sg_file = dir+'start_goal_%d' % (j) + '.pkl'
            file = open(sg_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_sg = p.load()
            file = open(path_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_path = p.load()
            file = open(control_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_control = p.load()
            file = open(cost_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_cost = p.load()


            data_path_before_preprocess = data_path
            
            if dynamics is not None:
                # use dense input
                data_path, data_control, data_cost = preprocess(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz, num_steps)
            p = data_path
            #print('before flip:')
            #print(p)
            if direction == 1:
                # backward
                p = np.flip(p, axis=0)
            #print('after flip:')
            #print(p)
            for k in range(len(p)-1):
                # If want to use all intermediate nodes as subgoal, then use the commented code
                if multigoal:
                    for l in range(k+1, len(p)):
                        waypoint_dataset.append(np.concatenate([p[k], p[l]]))
                        waypoint_targets.append(p[k+1])
                        env_indices.append(i)
                else:
                    # otherwise directly use the last node as goal
                    waypoint_dataset.append(np.concatenate([p[k], p[-1]]))
                    waypoint_targets.append(p[k+1])
                    env_indices.append(i)
                
                
                u_init_dataset.append(np.concatenate([p[k], p[k+1]]))
                u_init_targets.append(data_control[k])
                t_init_dataset.append(np.concatenate([p[k], p[k+1]]))
                t_init_targets.append(data_cost[k])
        #path_env.append(paths)
        #path_length_env.append(path_lengths)
        #control_env.append(controls)
        #cost_env.append(costs)
        #sg_env.append(sgs)
    ## TODO: print out intermediate results to visualize
    waypoint_dataset = np.array(waypoint_dataset)
    waypoint_targets = np.array(waypoint_targets)
    env_indices = np.array(env_indices)
    u_init_dataset = np.array(u_init_dataset)
    u_init_targets = np.array(u_init_targets)
    t_init_dataset = np.array(t_init_dataset)
    t_init_targets = np.array(t_init_targets)
    if obs_list is not None:
        obs_list = np.array(obs_list)
        obc_list = np.array(obc_list)
    return obc_list, waypoint_dataset, waypoint_targets, env_indices, \
           u_init_dataset, u_init_targets, t_init_dataset, t_init_targets


#def load_test_dataset(N, NP, folder):
#    # obtain



def load_test_dataset(N, NP, data_folder, obs_f=None, s=0, sp=0):
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
        obs_list = None
        obc_list = None
    else:
        obs_list = []
        obc_list = []
        for i in range(s,s+N):
            file = open(data_folder+'obs_%d.pkl' % (i), 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obs = p.load()
            #obs = pickle.load(file)
            file = open(data_folder+'obc_%d.pkl' % (i), 'rb')
            #obc = pickle.load(file)
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obc = p.load()
            # concatenate on the first direction (each obs has a different index)
            obc = obc.reshape(-1, 2)
            obs_list.append(obs)
            obc_list.append(obc)
        obc_list = np.array(obc_list)
        obc_list = pcd_to_voxel2d(obc_list, voxel_size=[32,32]).reshape(-1,1,32,32)

    path_env = []
    path_length_env = []
    control_env = []
    cost_env = []
    sg_env = []
    
    
    
    
    path_nums = []

    for i in range(s,N+s):
        paths = []
        path_lengths = []
        costs = []
        controls = []
        sgs = []
        
        correct_paths = []
        correct_costs = []
        correct_controls = []
        
        
        for j in range(sp,NP+sp):
            dir = data_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            sg_file = dir+'start_goal_%d' % (j) + '.pkl'
            file = open(sg_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            sg = p.load()
            sgs.append(sg)
            file = open(path_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            paths.append(p)
            path_lengths.append(len(p))
            file = open(control_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            controls.append(p)
            file = open(cost_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            costs.append(p)


            
            """
            path_file = dir+'path_corrected_%d' %(j) + ".pkl"
            control_file = dir+'control_corrected_%d' %(j) + ".pkl"
            cost_file = dir+'cost_corrected_%d' %(j) + ".pkl"
            file = open(path_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            correct_paths.append(p)
            file = open(control_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            correct_controls.append(p)
            file = open(cost_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            p = p.load()
            correct_costs.append(p)
            # compare cost
            path_nums.append(int(np.round(np.sum(costs[-1])/0.002)))
            print('correct cost: ', np.array(correct_costs[-1]).shape)
            print('cost: ', np.array(costs[-1]).shape)
            #print('correct cost: ', np.sum(correct_costs[-1]))
            #print('cost: ', np.sum(costs[-1]))
            #print('difference: ', np.sum(correct_costs[-1])-np.sum(costs[-1]))
            
            #if np.abs(np.sum(correct_costs[-1])-np.sum(costs[-1])) > 0.:
            #    return
            """
            
            
            
            
            
        path_env.append(paths)
        path_length_env.append(path_lengths)
        control_env.append(controls)
        cost_env.append(costs)
        sg_env.append(sgs)
    if obs_list is not None:
        obs_list = np.array(obs_list)
        obc_list = np.array(obc_list)
        
    return obc_list, obs_list, path_env, sg_env, path_length_env, control_env, cost_env



def pcd_to_voxel2d(points, voxel_size=(24, 24), padding_size=(32, 32)):
    voxels = [voxelize2d(points[i], voxel_size, padding_size) for i in range(len(points))]
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





def load_train_dataset_cost(N, NP, data_folder, obs_f=None, direction=0, dynamics=None, enforce_bounds=None, system=None, step_sz=0.02, num_steps=20):
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
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obs = p.load()
            #obs = pickle.load(file)
            file = open(data_folder+'obc_%d.pkl' % (i), 'rb')
            #obc = pickle.load(file)
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            obc = p.load()
            # concatenate on the first direction (each obs has a different index)
            obc = obc.reshape(-1, 2)
            obs_list.append(obs)
            obc_list.append(obc)
        obc_list = np.array(obc_list)
        obc_list = pcd_to_voxel2d(obc_list, voxel_size=[32,32]).reshape(-1,1,32,32)

    cost_dataset = []
    cost_targets = []
    env_indices = []
    u_init_dataset = []  # (start, goal) -> control
    t_init_dataset = []  # (start, goal) -> dt
    u_init_targets = []
    t_init_targets = []

    for i in range(N):
        print('loading... env: %d' % (i))
        for j in range(NP):
            dir = data_folder+str(i)+'/'
            path_file = dir+'path_%d' %(j) + ".pkl"
            control_file = dir+'control_%d' %(j) + ".pkl"
            cost_file = dir+'cost_%d' %(j) + ".pkl"
            time_file = dir+'time_%d' %(j) + ".pkl"
            sg_file = dir+'start_goal_%d' % (j) + '.pkl'
            file = open(sg_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_sg = p.load()
            file = open(path_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_path = p.load()
            file = open(control_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_control = p.load()
            file = open(cost_file, 'rb')
            p = pickle._Unpickler(file)
            p.encoding = 'latin1'
            data_cost = p.load()


            if dynamics is not None:
                # use dense input
                data_path, data_control, data_cost = preprocess(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz, num_steps)
            #print('cost:')
            #print(data_cost)
            data_cost = np.array(data_cost)
            p = data_path
            #print('before flip:')
            #print(p)
            if direction == 1:
                # backward
                p = np.flip(p, axis=0)
            #print('after flip:')
            #print(p)
            for k in range(len(p)-1):
                for l in range(k+1, len(p)):
                    cost_dataset.append(np.concatenate([p[k], p[l]]))
                    #print('start idx: %d, end idx: %d' % (k, l))
                    #print(data_cost[k:l].sum())
                    cost_targets.append(data_cost[k:l].sum())
                    env_indices.append(i)
                u_init_dataset.append(np.concatenate([p[k], p[k+1]]))
                u_init_targets.append(data_control[k])
                t_init_dataset.append(np.concatenate([p[k], p[k+1]]))
                t_init_targets.append(data_cost[k])
        #path_env.append(paths)
        #path_length_env.append(path_lengths)
        #control_env.append(controls)
        #cost_env.append(costs)
        #sg_env.append(sgs)
    ## TODO: print out intermediate results to visualize
    cost_dataset = np.array(cost_dataset)
    cost_targets = np.array(cost_targets).reshape(-1,1)
    env_indices = np.array(env_indices)
    u_init_dataset = np.array(u_init_dataset)
    u_init_targets = np.array(u_init_targets)
    t_init_dataset = np.array(t_init_dataset)
    t_init_targets = np.array(t_init_targets)
    if obs_list is not None:
        obs_list = np.array(obs_list)
        obc_list = np.array(obc_list)
    return obc_list, cost_dataset, cost_targets, env_indices, \
           u_init_dataset, u_init_targets, t_init_dataset, t_init_targets


#def load_test_dataset(N, NP, folder):
#    # obtain
