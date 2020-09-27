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

def IsInCollision(x, obc, obc_width=4.):
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2

    
    if x[0] < MIN_X or x[0] > MAX_X:
        return True
    
    H = 0.5
    pole_x1 = x[0]
    pole_y1 = H
    pole_x2 = x[0] + L * np.sin(x[2])
    pole_y2 = H + L * np.cos(x[2])

    
    for i in range(len(obc)):
        for j in range(0, 8, 2):
            x1 = obc[i][j]
            y1 = obc[i][j+1]
            x2 = obc[i][(j+2) % 8]
            y2 = obc[i][(j+3) % 8]
            if line_line_cc(pole_x1, pole_y1, pole_x2, pole_y2, x1, y1, x2, y2):
                return True
    return False

def enforce_bounds(state):
    I = 10
    L = 2.5
    M = 10
    m = 5
    g = 9.8
    H = 0.5

    STATE_X = 0
    STATE_V = 1
    STATE_THETA = 2
    STATE_W = 3
    CONTROL_A = 0

    MIN_X = -30
    MAX_X = 30
    MIN_V = -40
    MAX_V = 40
    MIN_W = -2
    MAX_W = 2
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    if state[1]<MIN_V:
        state[1]=MIN_V
    elif state[1]>MAX_V:
        state[1]=MAX_V

    if state[2]<-np.pi:
        state[2]+=2*np.pi
    elif state[2]>np.pi:
        state[2]-=2*np.pi

    if state[3]<MIN_W:
        state[3]=MIN_W
    elif state[3]>MAX_W:
        state[3]=MAX_W
    return state



def init_informer(system, x0, xG, num_steps, step_sz):
    # here we find the nearest point to x0 in the data, and depending on direction, find the adjacent node
    circular = system.is_circular_topology()
    bound = [30.0, 40.0, 3.141592653589793, 2.0]
    # forward
    next_state = xG
    std = 0.1
    cov = np.diag([std*bound[0], std*bound[1], std*bound[2], std*bound[3]])
    #mean = next_state
    #next_state = np.random.multivariate_normal(mean=mean,cov=cov)
    mean = np.zeros(next_state.shape)
    rand_x_init = np.random.multivariate_normal(mean=mean, cov=cov, size=num_steps)
    rand_x_init[0] = rand_x_init[0] * 0.
    rand_x_init[-1] = rand_x_init[-1] * 0.
    # initial: from max_d_i to max_d_i+1
    delta_x = next_state - x0
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

    x_init = np.linspace(x0, x0+delta_x, num_steps) + rand_x_init
    #x_init = np.array(detail_paths[state_i[max_d_i]:state_i[next_idx]])
    # action: copy over to number of steps
    #u_init_i = np.random.uniform(low=[-4.], high=[4])
    #u_init_i = control[max_d_i]
    cost_i = step_sz*(num_steps-1)
    # add gaussian to u
    #u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
    u_init = np.random.uniform(low=[-300.], high=[300.], size=(num_steps,1))
    #u_init = u_init + np.random.normal(scale=1.)
    t_init = np.linspace(0, cost_i, num_steps)

    return x_init, u_init, t_init

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
    if args.env_type == 'pendulum':
        obs_file = None
        obc_file = None
        obs_f = False
        #system = standard_cpp_systems.PSOPTPendulum()
        #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
    elif args.env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 4.0
        step_sz = 0.002
        num_steps = 200
        goal_radius=2.0
        random_seed=0
        delta_near=0.01
        delta_drain=0.005
        obs_f = True
        psopt_system = _sst_module.PSOPTCartPole()
        cpp_propagator = _sst_module.SystemPropagator()

        #system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
        dynamics = lambda x, u, t: cpp_propagator.propagate(psopt_system, x, u, t)

    elif args.env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4', 'acrobot_obs_8']:
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 6.0
        step_sz = 0.02
        num_steps = 20
        goal_radius=2.0
        random_seed=0
        delta_near=0.1
        delta_drain=0.05

    # load previously trained model if start epoch > 0
    #model_path='kmpnet_epoch_%d_direction_0_step_%d.pkl' %(args.start_epoch, args.num_steps)
    mlp_path = os.path.join(os.getcwd()+'/c++/','acrobot_obs_MLP_lr0.010000_epoch_2850_step_20.pt')
    encoder_path = os.path.join(os.getcwd()+'/c++/','acrobot_obs_encoder_lr0.010000_epoch_2850_step_20.pt')
    cost_mlp_path = os.path.join(os.getcwd()+'/c++/','costnet_acrobot_obs_8_MLP_epoch_300_step_20.pt')
    cost_encoder_path = os.path.join(os.getcwd()+'/c++/','costnet_acrobot_obs_8_encoder_epoch_300_step_20.pt')

    print('mlp_path:')
    print(mlp_path)
    #####################################################
    def plan_one_path(obs_i, obs, obc, start_state, goal_state, goal_inform_state, max_iteration, data, out_queue):
        if args.env_type == 'pendulum':
            system = standard_cpp_systems.PSOPTPendulum()
            bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
            step_sz = 0.002
            num_steps = 20
            traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 200, num_steps, 1, 20, step_sz)

        elif args.env_type == 'cartpole_obs':
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            obs_width = 4.0
            psopt_system = _sst_module.PSOPTAcrobot()
            propagate_system = standard_cpp_systems.RectangleObs(obs, 4., 'cartpole')
            distance_computer = propagate_system.distance_computer()
            #distance_computer = _sst_module.euclidean_distance(np.array(propagate_system.is_circular_topology()))
            bvp_wrapper = _sst_module.PSOPTBVPWrapper(psopt_system, 4, 1, 0)
            step_sz = 0.002
            num_steps = 20
            psopt_num_steps = 20
            psopt_step_sz = 0.002
            goal_radius=2
            random_seed=0
            #delta_near=1.0
            #delta_drain=0.5
            delta_near=0.01
            delta_drain=0.005        
        elif args.env_type in ['acrobot_obs','acrobot_obs_2', 'acrobot_obs_3', 'acrobot_obs_4', 'acrobot_obs_8']:
            #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
            obs_width = 6.0
            psopt_system = _sst_module.PSOPTAcrobot()
            propagate_system = standard_cpp_systems.RectangleObs(obs, 6., 'acrobot')
            distance_computer = propagate_system.distance_computer()
            #distance_computer = _sst_module.euclidean_distance(np.array(propagate_system.is_circular_topology()))
            bvp_wrapper = _sst_module.PSOPTBVPWrapper(psopt_system, 4, 1, 0)
            step_sz = 0.02
            num_steps = 21
            psopt_num_steps = 21
            psopt_step_sz = 0.02
            goal_radius=2
            random_seed=0
            #delta_near=1.0
            #delta_drain=0.5
            delta_near=0.1
            delta_drain=0.05
        #print('creating planner...')
        planner = vis_planners.DeepSMPWrapper(mlp_path, encoder_path, 
                                              cost_mlp_path, cost_encoder_path, 
                                              20, psopt_num_steps, psopt_step_sz, step_sz, propagate_system, args.device)
        # generate a path by using SST to plan for some maximal iterations
        time0 = time.time()
        #print('obc:')
        #print(obc.shape)
        #print(delta_near)
        #print(delta_drain)
        #print('start_state:')
        #print(start_state)
        #print('goal_state:')
        #print(goal_state)

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.set_autoscale_on(True)
        ax.set_xlim(-30, 30)
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
        dx = 1
        dtheta = 0.1
        feasible_points = []
        infeasible_points = []
        imin = 0
        imax = int(2*30./dx)
        jmin = 0
        jmax = int(2*np.pi/dtheta)

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                x = np.array([dx*i-30, 0., dtheta*j-np.pi, 0.])
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
        ax.scatter(feasible_points[:,0], feasible_points[:,2], c='yellow')
        ax.scatter(infeasible_points[:,0], infeasible_points[:,2], c='pink')
        #for i in range(len(data)):
        #    update_line(hl, ax, data[i])
        
        data = np.array(data)
        ax.scatter(data[:,0], data[:,2], c='lightblue', s=10)
        ax.scatter(data[-1,0], data[-1,2], c='red', s=10, marker='*')

        draw_update_line(ax)
        state_t = start_state

        state_t = data[0]
        for data_i in range(num_steps,len(data),num_steps):
            print('iteration: %d' % (data_i))
            print('state_t:')
            print(state_t)    

            x_init, u_init, t_init = init_informer(propagate_system, state_t, data[data_i], psopt_num_steps+1, psopt_step_sz)
            print('x_init:')
            print(x_init)
            bvp_x, bvp_u, bvp_t = bvp_wrapper.solve(state_t, x_init[-1], 20, psopt_num_steps+1, 0.8*step_sz*num_steps, 2*psopt_step_sz*psopt_num_steps, \
                                                    x_init, u_init, t_init)
            print('bvp_x:')
            print(bvp_x)
            print('bvp_u:')
            print(bvp_u)
            print('bvp_t:')
            print(bvp_t)
            if len(bvp_u) != 0:# and bvp_t[0] > 0.01:  # turn bvp_t off if want to use step_bvp
                
                
                # propagate data
                #p_start = bvp_x[0]
                p_start = state_t
                detail_paths = [p_start]
                detail_controls = []
                detail_costs = []
                state = [p_start]
                control = []
                cost = []
                for k in range(len(bvp_t)):
                    #state_i.append(len(detail_paths)-1)
                    max_steps = int(np.round(bvp_t[k]/step_sz))
                    accum_cost = 0.
                    for step in range(1,max_steps+1):
                        p_start = dynamics(p_start, bvp_u[k], step_sz)
                        p_start = enforce_bounds(p_start)
                        detail_paths.append(p_start)
                        accum_cost += step_sz
                        if (step % 1 == 0) or (step == max_steps):
                            state.append(p_start)
                            cost.append(accum_cost)
                            accum_cost = 0.

                xs_to_plot = np.array(state)
                for i in range(len(xs_to_plot)):
                    xs_to_plot[i] = wrap_angle(xs_to_plot[i], propagate_system)
                #ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='green')
                ax.scatter(xs_to_plot[:,0], xs_to_plot[:,2], c='green', s=10.0)
                
                print('solution: x')
                print(bvp_x)
                print('solution: u')
                print(bvp_u)
                print('solution: t')
                print(bvp_t)
                # draw start and goal
                #ax.scatter(start_state[0], goal_state[0], marker='X')
                draw_update_line(ax)
                #state_t = state[-1]
                # try using mpnet_res as new start
                state_t = data[data_i]
                #state_t = xs_to_plot[-1]
                print('data_i:')
                
                print(data[data_i])
            else:
                # in incollision
                state_t = data[data_i]
        #if len(res_x) == 0:
        #    print('failed.')
        out_queue.put(0)
        #else:
        #    print('path succeeded.')
        #    out_queue.put(1)
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
    for i in range(len(paths)):
        new_obs_i = []
        obs_i = obs[i]
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
            #p = Process(target=plan_one_path, args=(obs[i], obc[i], start_state, goal_state, 500, queue))
            
            # propagate data
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
                #print('p_start:')
                #print(p_start)
                #print('data:')
                #print(paths[i][j][k])
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
                        #print('control')
                        #print(controls[i][j])
                        control.append(controls[i][j][k])
                        cost.append(accum_cost)
                        accum_cost = 0.
            #print('p_start:')
            #print(p_start)
            #print('data:')
            #print(paths[i][j][-1])
            state[-1] = paths[i][j][-1]
            data = state

            plan_one_path(obs_i, obs[i], obc[i], start_state, goal_state, goal_inform_state, 1000, data, queue)
            #p.start()
            #p.join()
            #res = queue.get()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # for training
    parser.add_argument('--model_path', type=str, default='/media/arclabdl1/HD1/YLmiao/results/KMPnet_res/acrobot_obs_lr0.010000_SGD/',help='path for saving trained models')
    parser.add_argument('--seen_N', type=int, default=10)
    parser.add_argument('--seen_NP', type=int, default=2)
    parser.add_argument('--seen_s', type=int, default=0)
    parser.add_argument('--seen_sp', type=int, default=801)
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
    parser.add_argument('--data_folder', type=str, default='./data/cartpole_obs/')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    parser.add_argument('--start_epoch', type=int, default=2850)
    parser.add_argument('--env_type', type=str, default='cartpole_obs', help='s2d for simple 2d, c2d for complex 2d')
    parser.add_argument('--world_size', nargs='+', type=float, default=[3.141592653589793, 3.141592653589793, 6.0, 6.0], help='boundary of world')
    parser.add_argument('--opt', type=str, default='Adagrad')
    parser.add_argument('--num_steps', type=int, default=20)

    args = parser.parse_args()
    print(args)
    main(args)
