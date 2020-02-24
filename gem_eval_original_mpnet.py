import sys
sys.path.append('.')

import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_utility.plan_general_original_mpnet import *

def eval_tasks(mpNet1, mpNet2, test_data, folder, filename, IsInCollision, normalize_func, unnormalize_func, informer, init_informer, system, dynamics, enforce_bounds, traj_opt, step_sz, num_steps):
    obc, obs, paths, sgs, path_lengths, controls, costs = test_data
    obc = obc.astype(np.float32)
    obc = torch.from_numpy(obc)
    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        # feasible paths for each env
        for j in range(len(paths[0])):
            time0 = time.time()
            time_norm = 0.
            fp = 0 # indicator for feasibility
            print ("step: i="+str(i)+" j="+str(j))
            p1_ind=0
            p2_ind=0
            p_ind=0
            if path_lengths[i][j]<2:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>=2:
                state_i = []
                state = paths[i][j]
                # obtain the sequence
                p_start = paths[i][j][0]
                detail_paths = [p_start]
                detail_controls = []
                detail_costs = []
                state = [p_start]
                control = []
                cost = []
                for k in range(len(controls[i][j])):
                    #state_i.append(len(detail_paths)-1)
                    max_steps = int(costs[i][j][k]/step_sz)
                    accum_cost = 0.
                    print('p_start:')
                    print(p_start)
                    print('data:')
                    print(paths[i][j][k])
                    # modify it because of small difference between data and actual propagation
                    p_start = paths[i][j][k]
                    state[-1] = paths[i][j][k]
                    for step in range(1,max_steps+1):
                        p_start = dynamics(p_start, controls[i][j][k], step_sz)
                        p_start = enforce_bounds(p_start)          
                        detail_paths.append(p_start)
                        detail_controls.append(controls[i][j])
                        detail_costs.append(step_sz)
                        accum_cost += step_sz
                        if (step % 1 == 0) or (step == max_steps):
                            state.append(p_start)
                            print('control')
                            print(controls[i][j])
                            control.append(controls[i][j][k])
                            cost.append(accum_cost)
                            accum_cost = 0.
                print('p_start:')
                print(p_start)
                print('data:')
                print(paths[i][j][-1])
                state[-1] = paths[i][j][-1]
                
                
                
                fp = 0
                valid_path.append(1)
                start_node = Node(paths[i][j][0])
                #goal_node = Node(sgs[i][j][1])
                goal_node = Node(paths[i][j][-1])
                print(goal_check(goal_node, Node(sgs[i][j][1])))
                #goal_node.S0 = np.diag([1.,1.,0,0])
                #goal_node.rho0 = 1.
                path = [start_node, goal_node]
                #step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 21
                for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                # 1.2, 0.5, 0.1 are for simple env
                # 0.04, 0.03, 0.02 are for home env
                    if (t == 2):
                        #step_sz = 1.2
                        step_sz = 0.02
                    elif (t == 3):
                            #step_sz = 0.5
                        step_sz = 0.02
                    elif (t > 3):
                        #step_sz = 0.1
                        step_sz = 0.02
                        #num_steps = num_steps * 2
                    path = neural_replan(mpNet1, mpNet2, path, Node(sgs[i][j][1]), obc[i], obs[i], IsInCollision, \
                                        normalize_func, unnormalize_func, t==0, step_sz, num_steps, \
                                        informer, init_informer, system, dynamics, enforce_bounds, traj_opt, state)
                    if feasibility_check(path, obc[i], IsInCollision, system):
                        fp = 1
                        print('feasible, ok!')
                        break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time1 -= time_norm
                time_path.append(time1)
                print('test time: %f' % (time1))
            # write the path
            #print('planned path:')
            #print(path)
            path = [p.numpy() for p in path]
            path = np.array(path)
            np.savetxt('path_%d.txt' % (j), path, fmt='%f')
            fes_path.append(fp)
            print('env %d accuracy up to now: %f' % (i, (float(np.sum(fes_path))/ np.sum(valid_path))))
        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (float(np.sum(fes_env)) / np.sum(valid_env)))
    if filename is not None:
        pickle.dump(time_env, open(filename, "wb" ))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)
