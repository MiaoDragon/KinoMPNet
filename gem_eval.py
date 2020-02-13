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
import sys
sys.path.append('..')
from plan_utility.data_structure import *
from plan_utility.informed_path import plan
from plan_utility.informed_path_only_mpnet import plan_mpnet
from sparse_rrt.systems.acrobot import Acrobot

import matplotlib.pyplot as plt
from visual.acrobot_vis import *
from visual.vis_tools import *
import matplotlib.pyplot as plt
#fig = plt.figure()
def eval_tasks(mpNet0, mpNet1, env_type, test_data, save_dir, data_type, normalize_func = lambda x:x, unnormalize_func=lambda x: x, dynamics=None, jac_A=None, jac_B=None, enforce_bounds=None, IsInCollision=None):
    # data_type: seen or unseen
    obc, obs, paths, sgs, path_lengths, controls, costs = test_data
    if obs is not None:
        obc = obc.astype(np.float32)
        obc = torch.from_numpy(obc)
    if torch.cuda.is_available():
        obc = obc.cuda()

    if env_type == 'pendulum':
        system = standard_cpp_systems.PSOPTPendulum()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
        step_sz = 0.002
        num_steps = 20
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, num_steps, 1, 20, step_sz)

    elif env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
        system = _sst_module.CartPole()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.002
        num_steps = 20
        traj_opt = lambda x0, x1, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 500, num_steps, step_sz*1, step_sz*50, x_init, u_init, t_init)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4']:
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 6.0
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 500, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        #step_sz = 0.002
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0


    circular = system.is_circular_topology()
    def informer(env, x0, xG, direction):
        x0_x = torch.from_numpy(x0.x).type(torch.FloatTensor)
        xG_x = torch.from_numpy(xG.x).type(torch.FloatTensor)
        x0_x = normalize_func(x0_x)
        xG_x = normalize_func(xG_x)
        if torch.cuda.is_available():
            x0_x = x0_x.cuda()
            xG_x = xG_x.cuda()
        if direction == 0:
            x = torch.cat([x0_x,xG_x], dim=0)
            mpNet = mpNet0
            if torch.cuda.is_available():
                x = x.cuda()
            next_state = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data
            next_state = unnormalize_func(next_state).numpy()[0]
            delta_x = next_state - x0.x
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1 and np.abs(delta_x[i]) >= np.pi*0.5:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        if delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            res = Node(next_state)
            x_init = np.linspace(x0.x, x0.x+delta_x, num_steps)
            ## TODO: : change this to general case
            u_init_i = np.random.uniform(low=[-4.], high=[4], size=(num_steps,1))
            u_init = u_init_i
            #u_init_i = control[max_d_i]
            cost_i = (num_steps-1)*step_sz  #TOEDIT
            #u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            #u_init = u_init + np.random.normal(scale=1., size=u_init.shape)
            t_init = np.linspace(0, cost_i, num_steps)
        else:
            x = torch.cat([xG_x,x0_x], dim=0)
            mpNet = mpNet1
            next_state = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data
            next_state = unnormalize_func(next_state).numpy()[0]
            delta_x = x0.x - next_state
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1 and np.abs(delta_x[i]) >= np.pi*0.5:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        elif delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            #next_state = state[max_d_i] + delta_x
            res = Node(next_state)
            # initial: from max_d_i to max_d_i+1
            x_init = np.linspace(next_state, next_state + delta_x, num_steps) + rand_x_init
            # action: copy over to number of steps
            u_init_i = np.random.uniform(low=[-4.], high=[4], size=(num_steps,1))
            u_init = u_init_i
            cost_i = (num_steps-1)*step_sz
            #u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            #u_init = u_init + np.random.normal(scale=1., size=u_init.shape)
            t_init = np.linspace(0, cost_i, num_steps)
        return res, x_init, u_init, t_init

    def init_informer(env, x0, xG, direction):
        if direction == 0:
            next_state = xG.x
            delta_x = next_state - x0.x
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1 and np.abs(delta_x[i]) >= np.pi*0.5:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        if delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            res = Node(next_state)
            x_init = np.linspace(x0.x, x0.x+delta_x, num_steps)
            ## TODO: : change this to general case
            u_init_i = np.random.uniform(low=[-4.], high=[4], size=(num_steps,1))
            u_init = u_init_i
            #u_init_i = control[max_d_i]
            cost_i = (num_steps-1)*step_sz
            #u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            #u_init = u_init + np.random.normal(scale=1., size=u_init.shape)
            t_init = np.linspace(0, cost_i, num_steps)
        else:
            next_state = xG.x
            delta_x = x0.x - next_state
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1 and np.abs(delta_x[i]) >= np.pi*0.5:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        elif delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            #next_state = state[max_d_i] + delta_x
            res = Node(next_state)
            # initial: from max_d_i to max_d_i+1
            x_init = np.linspace(next_state, next_state + delta_x, num_steps) + rand_x_init
            # action: copy over to number of steps
            u_init_i = np.random.uniform(low=[-4.], high=[4], size=(num_steps,1))
            u_init = u_init_i
            cost_i = (num_steps-1)*step_sz
            #u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            #u_init = u_init + np.random.normal(scale=1., size=u_init.shape)
            t_init = np.linspace(0, cost_i, num_steps)
        return x_init, u_init, t_init

    
    
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
        for j in range(len(paths[0])):
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
                    if (step % 20 == 0) or (step == max_steps):
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
            #############################
            
            
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
                goal = Node(sgs[i][j][1])
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
                    # convert obs_i center to points
                    new_obs_i = []
                    for k in range(len(obs_i)):
                        obs_pt = []
                        obs_pt.append(obs_i[k][0]-obs_width/2)
                        obs_pt.append(obs_i[k][1]-obs_width/2)
                        obs_pt.append(obs_i[k][0]-obs_width/2)
                        obs_pt.append(obs_i[k][1]+obs_width/2)
                        obs_pt.append(obs_i[k][0]+obs_width/2)
                        obs_pt.append(obs_i[k][1]+obs_width/2)
                        obs_pt.append(obs_i[k][0]+obs_width/2)
                        obs_pt.append(obs_i[k][1]-obs_width/2)
                        new_obs_i.append(obs_pt)
                    #obs_i = new_obs_i
                collision_check = lambda x: IsInCollision(x, new_obs_i)
                for t in range(MAX_NEURAL_REPLAN):
                    # adaptive step size on replanning attempts
                    res, path_list = plan(obs_i, obc_i, start, goal, detail_paths, informer, init_informer, system, dynamics, \
                               enforce_bounds, collision_check, traj_opt, jac_A, jac_B, step_sz=step_sz, MAX_LENGTH=1000)
                    #print('after neural replan:')
                    #print(path)
                    #path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                    #print('after lvc:')
                    #print(path)

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    # after plan, generate the trajectory, and check if it is within the region
                    xs = plot_trajectory(ax, start, goal, dynamics, enforce_bounds, collision_check, step_sz)

                    params = {}
                    params['obs_w'] = 6.
                    params['obs_h'] = 6.
                    params['integration_step'] = step_sz
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    animator = AcrobotVisualizer(Acrobot(), params)
                    animation_acrobot(fig, ax, animator, xs, obs_i)
                    plt.waitforbuttonpress()

                    
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







#########################################
def eval_tasks_mpnet(mpNet0, mpNet1, env_type, test_data, save_dir, data_type, normalize_func = lambda x:x, unnormalize_func=lambda x: x, dynamics=None, jac_A=None, jac_B=None, enforce_bounds=None, IsInCollision=None):
    # data_type: seen or unseen
    obc, obs, paths, sgs, path_lengths, controls, costs = test_data
    if obs is not None:
        obc = obc.astype(np.float32)
        obc = torch.from_numpy(obc)
    if torch.cuda.is_available():
        obc = obc.cuda()

    if env_type == 'pendulum':
        system = standard_cpp_systems.PSOPTPendulum()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
        step_sz = 0.002
        num_steps = 20

    elif env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
        system = _sst_module.CartPole()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.002
        num_steps = 20
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4']:
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 6.0
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 20
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0


    circular = system.is_circular_topology()
    def informer(env, x0, xG, direction):
        x0_x = torch.from_numpy(x0.x).type(torch.FloatTensor)
        xG_x = torch.from_numpy(xG.x).type(torch.FloatTensor)
        x0_x = normalize_func(x0_x)
        xG_x = normalize_func(xG_x)
        if torch.cuda.is_available():
            x0_x = x0_x.cuda()
            xG_x = xG_x.cuda()
        if direction == 0:
            x = torch.cat([x0_x,xG_x], dim=0)
            mpNet = mpNet0
            if torch.cuda.is_available():
                x = x.cuda()
            next_state = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data
            print('next state:')
            print(next_state)
            next_state = unnormalize_func(next_state).numpy()[0]
            print('after unnormalize:')
            print(next_state)
            delta_x = next_state - x0.x
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        if delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            res = Node(next_state)
            x_init = np.linspace(x0.x, x0.x+delta_x, num_steps)
            ## TODO: : change this to general case
            u_init_i = np.random.uniform(low=[-4.], high=[4])
            #u_init_i = control[max_d_i]
            cost_i = num_steps*step_sz
            u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            u_init = u_init + np.random.normal(scale=1.)
            t_init = np.linspace(0, cost_i, num_steps)
        else:
            x = torch.cat([xG_x,x0_x], dim=0)
            mpNet = mpNet1
            next_state = mpNet(x.unsqueeze(0), env.unsqueeze(0)).cpu().data
            print('next state:')
            print(next_state)
            next_state = unnormalize_func(next_state).numpy()[0]
            print('after unnormalize:')
            print(next_state)
            delta_x = x0.x - next_state
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        elif delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            #next_state = state[max_d_i] + delta_x
            res = Node(next_state)
            # initial: from max_d_i to max_d_i+1
            x_init = np.linspace(next_state, next_state + delta_x, num_steps) + rand_x_init
            # action: copy over to number of steps
            u_init_i = np.random.uniform(low=[-4.], high=[4])
            cost_i = num_steps*step_sz
            u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            u_init = u_init + np.random.normal(scale=1.)
            t_init = np.linspace(0, cost_i, num_steps)
        return res, x_init, u_init, t_init

    def init_informer(env, x0, xG, direction):
        if direction == 0:
            next_state = xG.x
            delta_x = next_state - x0.x
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        if delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            res = Node(next_state)
            x_init = np.linspace(x0.x, x0.x+delta_x, num_steps)
            ## TODO: : change this to general case
            u_init_i = np.random.uniform(low=[-4.], high=[4])
            #u_init_i = control[max_d_i]
            cost_i = num_steps*step_sz
            u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            u_init = u_init + np.random.normal(scale=1.)
            t_init = np.linspace(0, cost_i, num_steps)
        else:
            next_state = xG.x
            delta_x = x0.x - next_state
            # can be either clockwise or counterclockwise, take shorter one
            for i in range(len(delta_x)):
                if circular[i]:
                    delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    # randomly pick either direction
                    rand_d = np.random.randint(2)
                    if rand_d < 1:
                        if delta_x[i] > 0.:
                            delta_x[i] = delta_x[i] - 2*np.pi
                        elif delta_x[i] <= 0.:
                            delta_x[i] = delta_x[i] + 2*np.pi
            #next_state = state[max_d_i] + delta_x
            res = Node(next_state)
            # initial: from max_d_i to max_d_i+1
            x_init = np.linspace(next_state, next_state + delta_x, num_steps) + rand_x_init
            # action: copy over to number of steps
            u_init_i = np.random.uniform(low=[-4.], high=[4])
            cost_i = num_steps*step_sz
            u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
            u_init = u_init + np.random.normal(scale=1.)
            t_init = np.linspace(0, cost_i, num_steps)
        return x_init, u_init, t_init

    
    
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
        for j in range(len(paths[0])):
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
                    if (step % 20 == 0) or (step == max_steps):
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
            #############################
            
            
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
                #goal = Node(path[-1])
                goal = Node(sgs[i][j][1])
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
                    # convert obs_i center to points
                    new_obs_i = []
                    for k in range(len(obs_i)):
                        obs_pt = []
                        obs_pt.append(obs_i[k][0]-obs_width/2)
                        obs_pt.append(obs_i[k][1]-obs_width/2)
                        obs_pt.append(obs_i[k][0]-obs_width/2)
                        obs_pt.append(obs_i[k][1]+obs_width/2)
                        obs_pt.append(obs_i[k][0]+obs_width/2)
                        obs_pt.append(obs_i[k][1]+obs_width/2)
                        obs_pt.append(obs_i[k][0]+obs_width/2)
                        obs_pt.append(obs_i[k][1]-obs_width/2)
                        new_obs_i.append(obs_pt)
                    #obs_i = new_obs_i
                collision_check = lambda x: IsInCollision(x, new_obs_i)
                for t in range(MAX_NEURAL_REPLAN):
                    # adaptive step size on replanning attempts
                    res, path_list = plan_mpnet(obs_i, obc_i, start, goal, detail_paths, informer, init_informer, system, dynamics, \
                               enforce_bounds, collision_check, None, jac_A, jac_B, step_sz=step_sz, MAX_LENGTH=1000)

                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    # after plan, generate the trajectory, and check if it is within the region
                    xs = plot_trajectory(ax, start, goal, dynamics, enforce_bounds, collision_check, step_sz)

                    params = {}
                    params['obs_w'] = 6.
                    params['obs_h'] = 6.
                    params['integration_step'] = step_sz
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    animator = AcrobotVisualizer(Acrobot(), params)
                    animation_acrobot(fig, ax, animator, xs, obs_i)
                    plt.waitforbuttonpress()

                    
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



