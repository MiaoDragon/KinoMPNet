import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.autograd import Variable
import math
import time
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module

#import matplotlib.pyplot as plt
#fig = plt.figure()

def eval_tasks(mpNet0, mpNet1, env_type, test_data, save_dir, data_type, normalize_func = lambda x:x, unnormalize_func=lambda x: x, dynamics=None, jac_A=None, jac_B=None, enforce_bounds=None):
    # data_type: seen or unseen
    obc, obs, paths, path_lengths = test_data
    if obs is not None:
        obc = obc.astype(np.float32)
        obc = torch.from_numpy(obc)
    if torch.cuda.is_available():
        obc = obc.cuda()
    def informer(env, x0, xG, direction):
        x0 = torch.from_numpy(x0.x).type(torch.FloatTensor)
        xG = torch.from_numpy(xG.x).type(torch.FloatTensor)
        if torch.cuda.is_available():
            x0 = x0.cuda()
            xG = xG.cuda()
        if direction == 0:
            x = torch.cat([x0,xG], dim=0)
            mpNet = mpNet0
        else:
            x = torch.cat([xG,x0], dim=0)
            mpNet = mpNet1
        if torch.cuda.is_available():
            x = x.cuda()
        res = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data.numpy()[0]
        res = Node(res)
        return res

    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    for i in range(len(paths)):
        time_path = []
        fes_path = []   # 1 for feasible, 0 for not feasible
        valid_path = []      # if the feasibility is valid or not
        # save paths to different files, indicated by i
        #print(obs, flush=True)

        # feasible paths for each env
        if env_type == 'pendulum':
            system = standard_cpp_systems.PSOPTPendulum()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 20, step_sz)

        elif env_type == 'cartpole_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
            system = _sst_module.CartPole()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, step_sz)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0
        elif env_type == 'acrobot_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            system = _sst_module.PSOPTAcrobot()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, step_sz)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0
        elif args.env_type == 'acrobot_obs_2':
            system = _sst_module.PSOPTAcrobot()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, step_szs)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0
        elif args.env_type == 'acrobot_obs_3':
            system = _sst_module.PSOPTAcrobot()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
            step_sz = 0.002
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, step_sz)
            goal_S0 = np.identity(4)
            goal_rho0 = 1.0

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
                #plt.plot(paths[i][j][:,0], paths[i][j][:,1])

                start = Node(path[0])
                goal = Node(path[-1])
                goal.S0 = goal_S0
                goal.rho0 = goal_rho0    # change this later

                control = []
                time_step = []
                step_sz = step_sz
                MAX_NEURAL_REPLAN = 11
                if obs is None:
                    obs_i = None
                    obc_i = None
                else:
                    obs_i = obs[i]
                    obc_i = obc[i]
                for t in range(MAX_NEURAL_REPLAN):
                    # adaptive step size on replanning attempts
                    res, path_list = plan(obc[i], start, goal, informer, system, dynamics, \
                               enforce_bounds, traj_opt, jac_A, jac_B, step_sz=0.002, MAX_LENGTH=1000)
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
                #print('planned path:')
                #print(path)
                #path = np.array(path)
                #np.savetxt('results/path_%d.txt' % (j), path)
                #np.savetxt('results/control_%d.txt' % (j), np.array(control))
                #np.savetxt('results/timestep_%d.txt' % (j), np.array(time_step))

            fes_path.append(fp)

        time_env.append(time_path)
        time_total += time_path
        print('average test time up to now: %f' % (np.mean(time_total)))
        fes_env.append(fes_path)
        valid_env.append(valid_path)
        print('accuracy up to now: %f' % (float(np.sum(fes_env)) / np.sum(valid_env)))
        time_path = save_dir + 'mpnet_%s_time.pkl' % (data_type)
        pickle.dump(time_env, open(time_path, "wb" ))
        #print(fp/tp)
    return np.array(fes_env), np.array(valid_env)
