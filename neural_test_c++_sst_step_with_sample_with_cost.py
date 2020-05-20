'''
This is the main file to run gem_end2end network.
It simulates the real scenario of observing a data, puts it inside the memory (or not),
and trains the network using the data
after training at each step, it will output the R matrix described in the paper
https://arxiv.org/abs/1706.08840
and after sevral training steps, it needs to store the parameter in case emergency
happens
To make it work in a real-world scenario, it needs to listen to the observer at anytime,
and call the network to train if a new data is available
(this thus needs to use multi-process)
here for simplicity, we just use single-process to simulate this scenario
'''
from __future__ import print_function
import sys
sys.path.append('deps/sparse_rrt')
sys.path.append('.')

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


from sparse_rrt import _sst_module
import sparse_rrt.planners as vis_planners
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt.visualization import show_image_opencv
#import model.AE.identity as cae_identity
#from model.AE import CAE_acrobot_voxel_2d, CAE_acrobot_voxel_2d_2, CAE_acrobot_voxel_2d_3
#from model import mlp, mlp_acrobot
#from model.mlp import MLP
#from model.mpnet import KMPNet
import numpy as np
import argparse
import os
#import torch

#from gem_eval_original_mpnet import eval_tasks
#from iterative_plan_and_retreat.gem_eval import eval_tasks
#from torch.autograd import Variable
import copy
import os
import gc
import random
#from tools.utility import *
#from plan_utility import pendulum, acrobot_obs
#from sparse_rrt.systems import standard_cpp_systems
#from sparse_rrt import _sst_module
from multiprocessing import Process, Queue

from iterative_plan_and_retreat.data_structure import *
#from iterative_plan_and_retreat.plan_general import propagate

#from plan_utility.data_structure import *
#from plan_utility.plan_general_original_mpnet import propagate
from tools import data_loader
import jax
import time
import matplotlib.pyplot as plt

from plan_utility.line_line_cc import line_line_cc
def IsInCollision(x, obc, obc_width=6.):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi

    LENGTH = 20.
    m = 1.0
    lc = 0.5
    lc2 = 0.25
    l2 = 1.
    I1 = 0.2
    I2 = 1.0
    l = 1.0
    g = 9.81
    pole_x0 = 0.
    pole_y0 = 0.
    pole_x1 = LENGTH * np.cos(x[STATE_THETA_1] - np.pi / 2)
    pole_y1 = LENGTH * np.sin(x[STATE_THETA_1] - np.pi / 2)
    pole_x2 = pole_x1 + LENGTH * np.cos(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    pole_y2 = pole_y1 + LENGTH * np.sin(x[STATE_THETA_1] + x[STATE_THETA_2] - np.pi / 2)
    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x0, pole_y0, pole_x1, pole_y1, x1, y1, x2, y2):
                return True
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False

def enforce_bounds(state):
    STATE_THETA_1, STATE_THETA_2, STATE_V_1, STATE_V_2 = 0, 1, 2, 3
    MIN_V_1, MAX_V_1 = -6., 6.
    MIN_V_2, MAX_V_2 = -6., 6.
    MIN_TORQUE, MAX_TORQUE = -4., 4.

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    state = np.array(state)
    if state[0] < -np.pi:
        state[0] += 2*np.pi
    elif state[0] > np.pi:
        state[0] -= 2 * np.pi
    if state[1] < -np.pi:
        state[1] += 2*np.pi
    elif state[1] > np.pi:
        state[1] -= 2 * np.pi

    state[2:] = np.clip(
        state[2:],
        [MIN_V_1, MIN_V_2],
        [MAX_V_1, MAX_V_2])
    return state


def main(args):
    # set seed
    print(args.model_path)
    torch_seed = np.random.randint(low=0, high=1000)
    np_seed = np.random.randint(low=0, high=1000)
    py_seed = np.random.randint(low=0, high=1000)
    #torch.manual_seed(torch_seed)
    np.random.seed(np_seed)
    random.seed(py_seed)
    # Build the models
    #if torch.cuda.is_available():
    #    torch.cuda.set_device(args.device)

    # setup evaluation function and load function
    if args.env_type == 'acrobot_obs':
        obs_file = None
        obc_file = None
        #cpp_propagator = _sst_module.SystemPropagator()
        #dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)

        obs_f = True
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0
        obs_file = None
        obc_file = None
        system = _sst_module.PSOPTAcrobot()
        cpp_propagator = _sst_module.SystemPropagator()
        dynamics = lambda x, u, t: cpp_propagator.propagate(system, x, u, t)

        obs_f = True
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 21
        traj_opt = lambda x0, x1, step_sz, num_steps, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 200, num_steps, step_sz*1, step_sz*(num_steps-1), x_init, u_init, t_init)
        goal_S0 = np.diag([1.,1.,0,0])
        #goal_S0 = np.identity(4)
        goal_rho0 = 1.0

        


    if args.env_type == 'pendulum':
        step_sz = 0.002
        num_steps = 20

    elif args.env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
        step_sz = 0.002
        num_steps = 20
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif args.env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4', 'acrobot_obs_8']:
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 6.0
        step_sz = 0.02
        num_steps = 21
        goal_radius=10.0
        random_seed=0
        delta_near=1.0
        delta_drain=0.5


    # load previously trained model if start epoch > 0
    #model_path='kmpnet_epoch_%d_direction_0_step_%d.pkl' %(args.start_epoch, args.num_steps)
    mlp_path = os.path.join(os.getcwd()+'/c++/','acrobot_obs_MLP_epoch_5000.pt')
    encoder_path = os.path.join(os.getcwd()+'/c++/','acrobot_obs_encoder_epoch_5000.pt')
    cost_mlp_path = os.path.join(os.getcwd()+'/c++/','costnet_acrobot_obs_MLP_epoch_1200_step_10.pt')
    cost_encoder_path = os.path.join(os.getcwd()+'/c++/','costnet_acrobot_obs_encoder_epoch_1200_step_10.pt')
    
    print('mlp_path:')
    print(mlp_path)
    #####################################################
    def plan_one_path(obs_i, obs, obc, start_state, goal_state, goal_inform_state, cost_i, max_iteration, out_queue):
        if args.env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4', 'acrobot_obs_8']:
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            obs_width = 6.0
            psopt_system = _sst_module.PSOPTAcrobot()
            propagate_system = standard_cpp_systems.RectangleObs(obs, 6., 'acrobot')
            distance_computer = propagate_system.distance_computer()
            #distance_computer = _sst_module.euclidean_distance(np.array(propagate_system.is_circular_topology()))
            step_sz = 0.02
            num_steps = 21
            goal_radius=2.0
            random_seed=0
            delta_near=1.0
            delta_drain=0.5
            device = 3
            pick_goal_init_threshold = 0.1
            goal_linear_inc_start_rate = 0.4
            pick_goal_end_threshold = 0.95
            num_sample = 20
            bvp_min_time_steps = 5
            bvp_max_time_steps = 100
        #print('creating planner...')
        planner = vis_planners.DeepSMPWrapper(mlp_path, encoder_path, cost_mlp_path, cost_encoder_path, \
                                              200, num_steps, step_sz, propagate_system, device)
        #cost_threshold = cost_i * 1.1
        cost_threshold = 100000000.
        # generate a path by using SST to plan for some maximal iterations
        time0 = time.time()
        res_x, res_u, res_t = planner.plan_tree_SMP_cost("sst", propagate_system, psopt_system, obc.flatten(), start_state, goal_inform_state, goal_inform_state, \
                                goal_radius, max_iteration, distance_computer, \
                                delta_near, delta_drain, cost_threshold, \
                                pick_goal_init_threshold, goal_linear_inc_start_rate, pick_goal_end_threshold,
                                num_sample, bvp_min_time_steps, bvp_max_time_steps)
        plan_time = time.time() - time0

        """
        # visualization
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.set_autoscale_on(True)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        hl, = ax.plot([], [], 'b')
        #hl_real, = ax.plot([], [], 'r')
        hl_for, = ax.plot([], [], 'g')
        hl_back, = ax.plot([], [], 'r')
        hl_for_mpnet, = ax.plot([], [], 'lightgreen')
        hl_back_mpnet, = ax.plot([], [], 'salmon')

        #print(obs)
        def update_line(h, ax, new_data):
            new_data = wrap_angle(new_data, propagate_system)
            h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
            #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
            #h.set_ydata(np.append(h.get_ydata(), new_data[1]))

        def remove_last_k(h, ax, k):
            h.set_data(h.get_xdata()[:-k], h.get_ydata()[:-k])

        def draw_update_line(ax):
            #ax.relim()
            #ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            #plt.show()

        def wrap_angle(x, system):
            circular = system.is_circular_topology()
            res = np.array(x)
            for i in range(len(x)):
                if circular[i]:
                    # use our previously saved version
                    res[i] = x[i] - np.floor(x[i] / (2*np.pi))*(2*np.pi)
                    if res[i] > np.pi:
                        res[i] = res[i] - 2*np.pi
            return res
        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*np.pi/dtheta)


        for i in range(imin, imax):
            for j in range(imin, imax):
                x = np.array([dtheta*i-np.pi, dtheta*j-np.pi, 0., 0.])
                if IsInCollision(x, obs_i):
                    infeasible_points.append(x)
                else:
                    feasible_points.append(x)
        feasible_points = np.array(feasible_points)
        infeasible_points = np.array(infeasible_points)
        print('feasible points')
        print(feasible_points)
        print('infeasible points')
        print(infeasible_points)
        ax.scatter(feasible_points[:,0], feasible_points[:,1], c='yellow')
        ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c='pink')
        #for i in range(len(data)):
        #    update_line(hl, ax, data[i])
        draw_update_line(ax)
        #state_t = start_state
                
        if len(res_u):
            # propagate data
            p_start = res_x[0]
            detail_paths = [p_start]
            detail_controls = []
            detail_costs = []
            state = [p_start]
            control = []
            cost = []
            for k in range(len(res_u)):
                #state_i.append(len(detail_paths)-1)
                max_steps = int(res_t[k]/step_sz)
                accum_cost = 0.
                #print('p_start:')
                #print(p_start)
                #print('data:')
                #print(paths[i][j][k])
                # modify it because of small difference between data and actual propagation
                p_start = res_x[k]
                state[-1] = res_x[k]
                for step in range(1,max_steps+1):
                    p_start = dynamics(p_start, res_u[k], step_sz)
                    p_start = enforce_bounds(p_start)
                    detail_paths.append(p_start)
                    accum_cost += step_sz
                    if (step % 1 == 0) or (step == max_steps):
                        state.append(p_start)
                        #print('control')
                        #print(controls[i][j])
                        cost.append(accum_cost)
                        accum_cost = 0.
            #print('p_start:')
            #print(p_start)
            #print('data:')
            #print(paths[i][j][-1])
            state[-1] = res_x[-1]
            
            
            
            xs_to_plot = np.array(state)
            for i in range(len(xs_to_plot)):
                xs_to_plot[i] = wrap_angle(xs_to_plot[i], propagate_system)
                if IsInCollision(xs_to_plot[i], obs_i):
                    print('in collision')
            ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='green')
            # draw start and goal
            #ax.scatter(start_state[0], goal_state[0], marker='X')
            draw_update_line(ax)
            plt.waitforbuttonpress()
        """
        
        #im = planner.visualize_nodes(propagate_system)
        #sec = input('Let us wait for user input')
        #show_image_opencv(im, "planning_tree", wait=True)
        
        # validate if the path contains collision
        """
        if len(res_u):
            # propagate data
            p_start = res_x[0]
            detail_paths = [p_start]
            detail_controls = []
            detail_costs = []
            state = [p_start]
            control = []
            cost = []
            for k in range(len(res_u)):
                #state_i.append(len(detail_paths)-1)
                max_steps = int(res_t[k]/step_sz)
                accum_cost = 0.
                #print('p_start:')
                #print(p_start)
                #print('data:')
                #print(paths[i][j][k])
                # modify it because of small difference between data and actual propagation
                p_start = res_x[k]
                state[-1] = res_x[k]
                for step in range(1,max_steps+1):
                    p_start = dynamics(p_start, res_u[k], step_sz)
                    p_start = enforce_bounds(p_start)
                    detail_paths.append(p_start)
                    accum_cost += step_sz
                    if (step % 1 == 0) or (step == max_steps):
                        state.append(p_start)
                        #print('control')
                        #print(controls[i][j])
                        cost.append(accum_cost)
                        accum_cost = 0.
                        # check collision for the new state
                        assert not IsInCollision(p_start, obs_i)
                        
            #print('p_start:')
            #print(p_start)
            #print('data:')
            #print(paths[i][j][-1])
            state[-1] = res_x[-1]
        # validation end
        """
        
        print('plan time: %fs' % (plan_time))
        if len(res_x) == 0:
            print('failed.')
            out_queue.put(-1)
        else:
            print('path succeeded.')
            print('cost: %f' % (np.sum(res_t)))
            print('cost_threshold: %f' % (cost_threshold))
            print('data cost: %f' % (cost_i))
            out_queue.put(plan_time)
    ####################################################################################



    # load data
    print('loading...')
    if args.seen_N > 0:
        seen_test_data = data_loader.load_test_dataset(args.seen_N, args.seen_NP,
                                  args.data_folder, obs_f, args.seen_s, args.seen_sp)
    if args.unseen_N > 0:
        unseen_test_data = data_loader.load_test_dataset(args.unseen_N, args.unseen_NP,
                                  args.data_folder, obs_f, args.unseen_s, args.unseen_sp)
    # test
    # testing

    queue = Queue(1)
    print('testing...')
    seen_test_suc_rate = 0.
    unseen_test_suc_rate = 0.

    obc, obs, paths, sgs, path_lengths, controls, costs = seen_test_data
    obc = obc.astype(np.float32)
    #obc = torch.from_numpy(obc)
    #if torch.cuda.is_available():
    #    obc = obc.cuda()
    
    plan_res = []
    plan_times = []
    plan_res_all = []
    for i in range(len(paths)):
        new_obs_i = []
        obs_i = obs[i]
        plan_res_env = []
        plan_time_env = []
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
        obs_i = new_obs_i
        #print(obs_i)
        for j in range(len(paths[i])):
            start_state = sgs[i][j][0]
            goal_inform_state = paths[i][j][-1]
            goal_state = sgs[i][j][1]
            cost_i = costs[i][j].sum()
            #cost_i = 100000000.
            print('environment: %d/%d, path: %d/%d' % (i+1, len(paths), j+1, len(paths[i])))
            p = Process(target=plan_one_path, args=(obs_i, obs[i], obc[i], start_state, goal_state, goal_inform_state, cost_i, 300000, queue))
            #plan_one_path(obs_i, obs[i], obc[i], start_state, goal_state, goal_inform_state, cost_i, 300000, queue)
            p.start()
            p.join()
            res = queue.get()
            if res == -1:
                plan_res_env.append(0)
                plan_res_all.append(0)
            else:
                plan_res_env.append(1)
                plan_times.append(res)
                plan_res_all.append(1)
            print('average accuracy up to now: %f' % (np.array(plan_res_all).flatten().mean()))
            print('plan average time: %f' % (np.array(plan_times).mean()))
            print('plan time std: %f' % (np.array(plan_times).std()))
        plan_res.append(plan_res_env)
    print('plan accuracy: %f' % (np.array(plan_res).flatten().mean()))
    print('plan average time: %f' % (np.array(plan_times).mean()))
    print('plan time std: %f' % (np.array(plan_times).std()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_lr0.010000_SGD/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=10)
    parser.add_argument('--seen_NP', type=int, default=100)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=800)
    parser.add_argument('--unseen_N', type=int, default=0)
    parser.add_argument('--unseen_NP', type=int, default=0)
    parser.add_argument('--unseen_s', type=int, default=0)
    parser.add_argument('--unseen_sp', type=int, default=0)
    parser.add_argument('--grad_step', type=int, default=1, help='number of gradient steps in continual learning')
    # Model parameters
    parser.add_argument('--total_input_size', type=int, default=8, help='dimension of total input')
    parser.add_argument('--AE_input_size', type=int, default=32, help='dimension of input to AE')
    parser.add_argument('--mlp_input_size', type=int , default=136, help='dimension of the input vector')
    parser.add_argument('--output_size', type=int , default=4, help='dimension of the input vector')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=int, default=0, help='cuda device')
    parser.add_argument('--data_folder', type=str, default='./data/acrobot_obs/')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=5000)
    parser.add_argument('--env_type', type=str, default='acrobot_obs', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--num_steps', type=int, default=20)
    parser.add_argument('--plan_type', type=str, default='tree')

    args = parser.parse_args()
    print(args)
    main(args)
