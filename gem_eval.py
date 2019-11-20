import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from plan_general import *

import matplotlib.pyplot as plt
fig = plt.figure()

def eval_tasks(mpNet, bvp_solver, test_data, filename, IsInCollision, normalize_func = lambda x:x, unnormalize_func=lambda x: x, time_flag=False):
    obc, obs, paths, path_lengths = test_data
    if obs is not None:
        obs = obs.astype(np.float32)
        obs = torch.from_numpy(obs)
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
            if path_lengths[i][j]==0:
                # invalid, feasible = 0, and path count = 0
                fp = 0
                valid_path.append(0)
            if path_lengths[i][j]>0:
                fp = 0
                valid_path.append(1)
                #paths[i][j][0][1] = 0.
                #paths[i][j][path_lengths[i][j]-1][1] = 0.
                path = [paths[i][j][0], paths[i][j][path_lengths[i][j]-1]]
                # plot the entire path
                plt.plot(paths[i][j][:,0], paths[i][j][:,1])



                control = []
                time_step = []
                step_sz = DEFAULT_STEP
                MAX_NEURAL_REPLAN = 11
                if obs is None:
                    obs_i = None
                    obc_i = None
                else:
                    obs_i = obs[i]
                    obc_i = obc[i]
                for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                    if (t == 2):
                        step_sz = 1.2
                    elif (t == 3):
                        step_sz = 0.5
                    elif (t > 3):
                        step_sz = 0.1
                    if time_flag:
                        res, path, control, time_step, time_norm = neural_replan(mpNet, bvp_solver, path, control, time_step, obc_i, obs_i, IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    else:
                        res, path, control, time_step = neural_replan(mpNet, bvp_solver, path, control, time_step, obc_i, obs_i, IsInCollision, \
                                            normalize_func, unnormalize_func, t==0, step_sz=step_sz, time_flag=time_flag)
                    #print('after neural replan:')
                    #print(path)
                    #path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    #print('after lvc:')
                    #print(path)
                    if res:
                        fp = 1
                        print('feasible ok!')
                        break
                    #if feasibility_check(bvp_solver, path, obc_i, IsInCollision, step_sz=0.01):
                    #    fp = 1
                    #    print('feasible, ok!')
                    #    break
            if fp:
                # only for successful paths
                time1 = time.time() - time0
                time1 -= time_norm
                time_path.append(time1)
                print('test time: %f' % (time1))
                # write the path
                print('planned path:')
                print(path)
                path = np.array(path)
                np.savetxt('results/path_%d.txt' % (j), path)
                np.savetxt('results/control_%d.txt' % (j), np.array(control))
                np.savetxt('results/timestep_%d.txt' % (j), np.array(time_step))

            fes_path.append(fp)

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
