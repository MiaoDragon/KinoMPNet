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

import jax
from tvlqr.python_lyapunov import *
from visual.acrobot_vis import *
#from visual.vis_tools import *
import matplotlib.pyplot as plt



def plot_ellipsoid(ax, S, rho, x0, alpha=1.0):
    theta = np.linspace(0, np.pi*2, 100)
    U = [np.cos(theta), np.sin(theta), np.zeros(100), np.zeros(100)]
    U = np.array(U).T
    tmp = np.linalg.pinv(S)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    S_invsqrt = scipy.linalg.sqrtm(tmp)
    X = U @ S_invsqrt  # 100x2
    X = np.sqrt(rho)*X + x0
    ax.plot(X[:,0],X[:,1], alpha=alpha)

def animation_acrobot(fig, ax, animator, xs, obs):
    animator.obs = obs
    animator._init(ax)
    for i in range(0,len(xs)):
        animator._animate(xs[i], ax)
        animator.draw_update_line(fig, ax)

def plot_trajectory(ax, start, goal, dynamics, enforce_bounds, IsInCollision, step_sz):

    plot_ellipsoid(ax, goal.S0, goal.rho0, goal.x, alpha=0.1)

    # plot funnel
    # rho_t = rho0+(rho1-rho0)/(t1-t0)*t
    node = start
    while node.edge is not None:
        if node.edge.S is not None:
            rho0s = node.edge.rho0s[node.edge.i0:]
            rho1s = node.edge.rho1s[node.edge.i0:]
            time_knot = node.edge.time_knot[node.edge.i0:]
            S = node.edge.S
            for i in range(len(rho0s)):
                rho0 = rho0s[i]
                rho1 = rho1s[i]
                t0 = time_knot[i]
                t1 = time_knot[i+1]
                rho_t = rho0
                S_t = S(t0).reshape(len(node.x),len(node.x))
                x_t = node.edge.xtraj(t0)
                u_t = node.edge.utraj(t0)
                # plot
                plot_ellipsoid(ax, S_t, rho_t, x_t, alpha=0.1)
                rho_t = rho1
                S_t = S(t1).reshape(len(node.x),len(node.x))
                x_t = node.edge.xtraj(t1)
                u_t = node.edge.utraj(t1)
                # plot
                plot_ellipsoid(ax, S_t, rho_t, x_t, alpha=0.1)
        node = node.next
    node = start
    actual_x = node.x
    xs = []
    us = []
    valid = True
    while node.edge is not None:
        # printout which node it is
        print('steering node...')
        print('node.x:')
        print(node.x)
        print('node.next.x:')
        print(node.next.x)
        # if node does not have controller defined, we use open-loop traj
        if node.edge.S is None:
            xs += node.edge.xs.tolist()
            actual_x = np.array(xs[-1])
        else:
            # then we use the controller
            # see if it can go to the goal region starting from start
            dt = node.edge.dts[node.edge.i0:]
            num = np.sum(dt)/step_sz
            time_span = np.linspace(node.edge.t0, node.edge.t0+np.sum(dt), num+1)
            delta_t = step_sz
            xs.append(actual_x)
            controller = node.edge.controller
            print('number of time knots: %d' % (len(time_span)))
            # plot data
            for i in range(len(time_span)):
                u = controller(time_span[i], actual_x)
                actual_x = dynamics(actual_x, u, step_sz)
                xs.append(actual_x)
                actual_x = enforce_bounds(actual_x)
                print('actual x:')
                print(actual_x)
                if IsInCollision(actual_x):
                    print('In Collision Booooo!!')
                    valid = False
        node = node.next
    xs = np.array(xs)
    ax.plot(xs[:,0], xs[:,1], 'black', label='using controller')
    plt.show()
    print('start:')
    print(start.x)
    print('goal:')
    print(goal.x)
    if not valid:
        print('in Collision Boommm!!!')

    plt.waitforbuttonpress()
    return xs




def eval_tasks(mpNet1, mpNet2, test_data, folder, filename, IsInCollision, normalize_func, unnormalize_func, informer, init_informer, system, dynamics, xdot, jax_dynamics, enforce_bounds, traj_opt, step_sz, num_steps):
    obc, obs, paths, sgs, path_lengths, controls, costs = test_data
    obc = obc.astype(np.float32)
    obc = torch.from_numpy(obc)
    fes_env = []   # list of list
    valid_env = []
    time_env = []
    time_total = []
    jac_A = jax.jacfwd(jax_dynamics, argnums=0)
    jac_B = jax.jacfwd(jax_dynamics, argnums=1)

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
                    if feasibility_check(path, Node(sgs[i][j][1]), obc[i], IsInCollision, system):
                        fp = 1
                        print('feasible, ok!')
                        break
            if fp:
                # only for successful paths

                # goal compute the stability region
                path[-1].x = sgs[i][j][1]  # change to real goal
                path[-1].S0 = np.diag([1.,1.,0.,0.])
                path[-1].rho0 = 1.0
                # reversely construct funnel

                lazyFunnel(path[0], path[-1], xdot, enforce_bounds, jac_A, jac_B, traj_opt, system=system, step_sz=step_sz)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                # after plan, generate the trajectory, and check if it is within the region
                xs = plot_trajectory(ax, path[0], path[-1], dynamics, enforce_bounds, collision_check, step_sz)

                params = {}
                params['obs_w'] = 6.
                params['obs_h'] = 6.
                params['integration_step'] = step_sz
                fig = plt.figure()
                ax = fig.add_subplot(111)
                animator = AcrobotVisualizer(Acrobot(), params)
                animation_acrobot(fig, ax, animator, xs, obs_i)
                plt.waitforbuttonpress()

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
