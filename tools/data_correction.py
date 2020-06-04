"""
This implements data loader for both training and testing procedures.
"""
import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('../')
import numpy as np
import random
import os
from sparse_rrt import _sst_module
import pickle

def process(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz):
    # propagate and calculate difference with the next waypoint, if the same, then ignore the rest
    # of control and cost
    correct_path = []
    correct_control = []
    correct_cost = []
    
    
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
        # modify it because of small difference between data and actual propagation
        #p_start = data_path[k]
        #state[-1] = data_path[k]
        
        correct_path.append(p_start)
        correct_control.append(data_control[k])
        
        for step in range(1,max_steps+1):
            p_start = dynamics(p_start, data_control[k], step_sz)
            accum_cost += step_sz
            # calculate if this is the next waypoint
            if np.linalg.norm(p_start - data_path[k+1]) <= 1e-6:
                # end here
                correct_cost.append(accum_cost)
                accum_cost = 0.
                if step != max_steps:
                    print('step:', step)
                    print('max_step:', max_steps)
                    print('not using all the steps')
                    
                break
    #state[-1] = data_path[-1]
    correct_path.append(p_start)
    return correct_path, correct_control, correct_cost

def correct_dataset(N, NP, data_folder, obs_f=None, direction=0, dynamics=None, enforce_bounds=None, system=None, step_sz=0.02):
    # obtain the generated paths, and transform into
    # (obc, dataset, targets, env_indices)
    # return list NOT NUMPY ARRAY
    ## TODO: add different folders for obstacle information and path
    # transform paths into dataset and targets
    # (xt, xT), x_{t+1}
    # direction: 0 -- forward;  1 -- backward

    # load obs and obc (obc: obstacle point cloud)

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



            correct_path, correct_control, correct_cost = process(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz)
            correct_path_file = dir+'path_corrected_%d' %(j) + ".pkl"
            correct_control_file = dir+'control_corrected_%d' %(j) + ".pkl"
            correct_cost_file = dir+'cost_corrected_%d' %(j) + ".pkl"
            
            
            # store the corrected file
            file = open(correct_path_file, 'wb')
            pickle.dump(correct_path, file)
            file = open(correct_control_file, 'wb')
            pickle.dump(correct_control, file)
            file = open(correct_cost_file, 'wb')
            pickle.dump(correct_cost, file)
            

            
def rename_dataset(N, NP, data_folder, obs_f=None, direction=0, dynamics=None, enforce_bounds=None, system=None, step_sz=0.02):
    # load corrected and then copy to original data
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



            correct_path, correct_control, correct_cost = process(data_path, data_control, data_cost, dynamics, enforce_bounds, system, step_sz)
            correct_path_file = dir+'path_%d' %(j) + ".pkl"
            correct_control_file = dir+'control_%d' %(j) + ".pkl"
            correct_cost_file = dir+'cost_%d' %(j) + ".pkl"
            
            
            # store the corrected file
            file = open(correct_path_file, 'wb')
            pickle.dump(np.array(correct_path), file)
            file = open(correct_control_file, 'wb')
            pickle.dump(np.array(correct_control), file)
            file = open(correct_cost_file, 'wb')
            pickle.dump(np.array(correct_cost), file)
            
            
            


            
cpp_propagator = _sst_module.SystemPropagator()
acrobot_system = _sst_module.PSOPTAcrobot()
acrobot_dynamics = lambda x, u, t: cpp_propagator.propagate(acrobot_system, x, u, t)
cartpole_system = _sst_module.PSOPTCartPole()
cartpole_dynamics = lambda x, u, t: cpp_propagator.propagate(cartpole_system, x, u, t)


# use the following if don't want to modify the original dataset, but create new ones
#correct_dataset(N=10, NP=1000, data_folder='../data/acrobot_obs/', obs_f=True, direction=0, dynamics=acrobot_dynamics, enforce_bounds=None, system=None, step_sz=0.02)

#correct_dataset(N=10, NP=1000, data_folder='../data/cartpole_obs/', obs_f=True, direction=0, dynamics=cartpole_dynamics, enforce_bounds=None, system=None, step_sz=0.002)


# use the following if want to change the original dataset
rename_dataset(N=10, NP=1000, data_folder='../data/acrobot_obs/', obs_f=True, direction=0, dynamics=acrobot_dynamics, enforce_bounds=None, system=None, step_sz=0.02)

rename_dataset(N=10, NP=1000, data_folder='../data/cartpole_obs/', obs_f=True, direction=0, dynamics=cartpole_dynamics, enforce_bounds=None, system=None, step_sz=0.002)
