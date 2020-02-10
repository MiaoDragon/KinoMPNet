import sys
import jax
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
from tools import data_loader
from sparse_rrt.planners import SST
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import pickle
import scipy
from plan_utility.informed_path import *
from plan_utility.plan_general import *
from plan_utility.data_structure import *
from plan_utility import pendulum, acrobot_obs
from sparse_rrt.systems.acrobot import Acrobot

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
                xdot = dynamics(actual_x, u)
                actual_x = actual_x + xdot * delta_t
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

env_type = 'acrobot_obs'
data_folder = '../data/acrobot_obs/'
# setup evaluation function and load function
if env_type == 'pendulum':
    IsInCollision =pendulum.IsInCollision
    normalize = pendulum.normalize
    unnormalize = pendulum.unnormalize
    obs_file = None
    obc_file = None
    dynamics = pendulum.dynamics
    jax_dynamics = pendulum.jax_dynamics
    enforce_bounds = pendulum.enforce_bounds

    obs_f = False
    #system = standard_cpp_systems.PSOPTPendulum()
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
elif env_type == 'cartpole_obs':
    IsInCollision =cartpole.IsInCollision
    normalize = cartpole.normalize
    unnormalize = cartpole.unnormalize
    obs_file = None
    obc_file = None
    dynamics = cartpole.dynamics
    jax_dynamics = cartpole.jax_dynamics
    enforce_bounds = cartpole.enforce_bounds

    obs_f = True
    #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'cartpole')
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
elif env_type == 'acrobot_obs':
    IsInCollision =acrobot_obs.IsInCollision
    normalize = acrobot_obs.normalize
    unnormalize = acrobot_obs.unnormalize
    obs_file = True
    obc_file = True
    dynamics = acrobot_obs.dynamics
    jax_dynamics = acrobot_obs.jax_dynamics
    enforce_bounds = acrobot_obs.enforce_bounds
    obs_f = True
    #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)

jac_A = jax.jacfwd(jax_dynamics, argnums=0)
jac_B = jax.jacfwd(jax_dynamics, argnums=1)


test_data = data_loader.load_test_dataset(1, 5, data_folder, sp=51, obs_f=obs_f)

# data_type: seen or unseen
obc, obs, paths, sgs, path_lengths, controls, costs = test_data


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
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 20, 0.002)
    elif env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
        system = _sst_module.CartPole()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type == 'acrobot_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        obs_width = 6.0
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        step_sz = 0.02
        num_steps = 20
        traj_opt = lambda x0, x1, x_init, u_init, t_init: bvp_solver.solve(x0, x1, 500, num_steps, 0.02*1, 0.02*5*num_steps, x_init, u_init, t_init)
        #goal_S0 = np.identity(4)
        goal_S0 = np.diag([1.,1.,0.,0.])
        goal_rho0 = 0.5
    elif env_type == 'acrobot_obs_2':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type == 'acrobot_obs_3':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0

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
    collision_check = lambda x: IsInCollision(x, new_obs_i)
        
        
        
    for j in range(len(paths[0])):
        #paths[i][j] = paths[i][j].astype(np.float32)
        #controls[i][j] = controls[i][j].astype(np.float32)
        #costs[i][j] = costs[i][j].astype(np.float32)
        state_i = []
        state = paths[i][j]
        # collision check for waypoint data
        print('checking data waypoint for collision...')
        for k in range(len(state)):
            print('InCollision: ')
            print(collision_check(state[k]))
        
        
        # obtain the sequence
        p_start = paths[i][j][0]
        detail_paths = [p_start]
        detail_controls = []
        detail_costs = []
        state = [p_start]
        control = []
        cost = []
        data_step_sz = 0.02
        for k in range(len(controls[i][j])):
            #state_i.append(len(detail_paths)-1)
            print('before int: %f' %(costs[i][j][k]/data_step_sz))
            max_steps = int(np.round(costs[i][j][k]/data_step_sz))
            accum_cost = 0.
            print('p_start:')
            print(p_start)
            print('data:')
            print(paths[i][j][k])
            # modify it because of small difference between data and actual propagation
            p_start = paths[i][j][k]   #uncomment this to allow smaller discrepency
            state[-1] = paths[i][j][k]
            for step in range(1,max_steps+1):
                p_start = p_start + data_step_sz*dynamics(p_start, controls[i][j][k])
                p_start = enforce_bounds(p_start)          
                detail_paths.append(p_start)
                detail_controls.append(controls[i][j])
                detail_costs.append(data_step_sz)
                accum_cost += data_step_sz
                if (step % 20 == 0) or (step == max_steps):  #TOEDIT: 200->20
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
        print('checking dense waypoint for collision...')
        for k in range(len(state)):
            print('InCollision: ')
            print(collision_check(state[k]))

        #detail_paths.append(paths[i][j][-1])
        #state = detail_paths[::200]
        #state = paths[i][j]
        #control = controls[i][j]
        #cost = costs[i][j]
        #state = detail_paths
        #control = detail_controls
        #cost = detail_costs
        print('cost:')
        print(cost)
        max_ahead = 1
        def informer(env, x0, xG, direction):
            # here we find the nearest point to x0 in the data, and depending on direction, find the adjacent node
            dis = x0.x - state
            circular = system.is_circular_topology()
            for i in range(len(x0.x)):
                if circular[i]:
                    # if it is angle, should map to -pi to pi
                    # map to [-pi, pi]
                    dis[:,i] = dis[:,i] - np.floor(dis[:,i] / (2*np.pi))*(2*np.pi)
                    # should not change the "sign" of the delta_x
                    dis[:,i] = (dis[:,i] > np.pi) * (dis[:,i] - 2*np.pi) + (dis[:,i] <= np.pi) * dis[:,i]
            #S = np.identity(len(x0.x))
            S = np.diag([1.,1.,0.,0.])
            #S = np.diag([1/30./30., 1/40./40., 1., 1.])
            #dif = np.sqrt(dis.T@S@dis)
            dif = []
            for i in range(len(dis)):
                dif.append(np.sqrt(dis[i].T@S@dis[i]))
            dif = np.array(dif)
            #dif = np.linalg.norm(dis, axis=1)
            max_d_i = np.argmin(dif)
            #print('current state: ')
            #print(x0.x)
            #print('chosen data:')
            #print(state[max_d_i])

            if direction == 0:
                # forward
                next_indices = np.minimum(np.arange(start=max_d_i+1, stop=max_d_i+max_ahead+1, step=1, dtype=int), len(state)-1)
                next_idx = np.random.choice(next_indices)      
                next_state = np.array(state[next_idx])
                cov = np.diag([0.01,0.01,0.0,0.0])
                #mean = next_state
                #next_state = np.random.multivariate_normal(mean=mean,cov=cov)
                mean = np.zeros(next_state.shape)
                rand_x_init = np.random.multivariate_normal(mean=mean, cov=cov, size=num_steps)
                
                # initial: from max_d_i to max_d_i+1
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
                            elif delta_x[i] <= 0.:
                                delta_x[i] = delta_x[i] + 2*np.pi
                res = Node(next_state)
                x_init = np.linspace(x0.x, x0.x+delta_x, num_steps) + rand_x_init
                #x_init = np.array(detail_paths[state_i[max_d_i]:state_i[next_idx]])
                # action: copy over to number of steps
                if max_d_i < len(control):
                    u_init_i = np.random.uniform(low=[-4.], high=[4])
                    #u_init_i = control[max_d_i]
                    cost_i = cost[max_d_i]
                    print(cost_i)
                else:
                    u_init_i = np.array(control[max_d_i-1])*0.
                    cost_i = step_sz
                # add gaussian to u
                u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
                u_init = u_init + np.random.normal(scale=1.)
                t_init = np.linspace(0, cost_i, num_steps)
            else:
                if max_d_i-1 == -1:
                    next_idx = max_d_i
                else:
                    next_idx = max_d_i-1
                next_indices = np.maximum(np.arange(start=max_d_i-1, stop=max_d_i-max_ahead-1, step=-1, dtype=int), 0)
                next_idx = np.random.choice(next_indices)                          
                next_state = np.array(state[next_idx])
                cov = np.diag([0.01,0.01,0.0,0.0])
                #mean = next_state
                #next_state = np.random.multivariate_normal(mean=mean,cov=cov)
                mean = np.zeros(next_state.shape)
                rand_x_init = np.random.multivariate_normal(mean=mean,cov=cov, size=num_steps)
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
                if max_d_i > 0:
                    u_init_i = control[max_d_i-1]
                    cost_i = cost[max_d_i-1]
                else:
                    u_init_i = np.array(control[max_d_i])*0.
                    cost_i = step_sz               
                u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
                u_init = u_init + np.random.normal(scale=1.)                
                t_init = np.linspace(0, cost_i, num_steps)
                #t_init = np.linspace(0, step_sz*(num_steps-1), num_steps)
            return res, x_init, u_init, t_init

        def init_informer(env, x0, xG, direction):
            # here we find the nearest point to x0 in the data, and depending on direction, find the adjacent node
            circular = system.is_circular_topology()
            if direction == 0:
                # forward
                next_state = xG.x
                cov = np.diag([0.01,0.01,0.0,0.0])
                #mean = next_state
                #next_state = np.random.multivariate_normal(mean=mean,cov=cov)
                mean = np.zeros(next_state.shape)
                rand_x_init = np.random.multivariate_normal(mean=mean, cov=cov, size=num_steps)
                # initial: from max_d_i to max_d_i+1
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
                            elif delta_x[i] <= 0.:
                                delta_x[i] = delta_x[i] + 2*np.pi
                x_init = np.linspace(x0.x, x0.x+delta_x, num_steps) + rand_x_init
                #x_init = np.array(detail_paths[state_i[max_d_i]:state_i[next_idx]])
                # action: copy over to number of steps
                u_init_i = np.random.uniform(low=[-4.], high=[4])
                #u_init_i = control[max_d_i]
                cost_i = step_sz*num_steps*2
                # add gaussian to u
                u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
                u_init = u_init + np.random.normal(scale=1.)
                t_init = np.linspace(0, cost_i, num_steps)
            else:
                next_state = xG.x
                cov = np.diag([0.01,0.01,0.0,0.0])
                #mean = next_state
                #next_state = np.random.multivariate_normal(mean=mean,cov=cov)
                mean = np.zeros(next_state.shape)
                rand_x_init = np.random.multivariate_normal(mean=mean,cov=cov, size=num_steps)
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
                            if delta_x[i] <= 0.:
                                delta_x[i] = delta_x[i] + 2*np.pi
                # initial: from max_d_i to max_d_i+1
                x_init = np.linspace(next_state, next_state + delta_x, num_steps) + rand_x_init
                u_init_i = np.random.uniform(low=[-4.], high=[4])
                cost_i = step_sz*num_steps*2           
                u_init = np.repeat(u_init_i, num_steps, axis=0).reshape(-1,len(u_init_i))
                u_init = u_init + np.random.normal(scale=1.)                
                t_init = np.linspace(0, cost_i, num_steps)
                #t_init = np.linspace(0, step_sz*(num_steps-1), num_steps)
            return x_init, u_init, t_init

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        state_pre, control_pre, cost_pre = data_loader.preprocess(paths[i][j], controls[i][j], costs[i][j], dynamics, enforce_bounds, step_sz, num_steps)
        xs_to_plot = np.array(state_pre)
        for i in range(len(xs_to_plot)):
            xs_to_plot[i] = wrap_angle(xs_to_plot[i], system)
        ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='red')
        
        xs_to_plot = np.array(state)
        for i in range(len(xs_to_plot)):
            xs_to_plot[i] = wrap_angle(xs_to_plot[i], system)
        ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='blue')
        plt.waitforbuttonpress()
        """
       
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
            #goal = Node(sgs[i][j][1])  # using the start and goal read from data
            print('detailed distance: %f' % (node_h_dist(state[-1], sgs[i][j][1], goal_S0, goal_rho0, system)))
            print("data distance: %f" % (node_h_dist(paths[i][j][-1], sgs[i][j][1], goal_S0, goal_rho0, system)))  # should be <1
            #goal_rho0 = np.sqrt(node_h_dist(paths[i][j][-1], sgs[i][j][1], goal_S0, goal_rho0, system)) * goal_rho0*1.05
            #print("data distance: %f" % (node_h_dist(paths[i][j][-1], sgs[i][j][1], goal_S0, goal_rho0, system)))  # should be <1
            print('endpoint:')
            print(goal.x)
            print('detail[-1]:')
            print(state[-1])
            #goal = Node(paths[i][j][-2])
            #goal = Node(path[-1])
            goal.S0 = goal_S0
            goal.rho0 = goal_rho0    # change this later

            #control = []
            time_step = []

            MAX_NEURAL_REPLAN = 11
            
            for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                res, path_list = plan(obs_i, obc_i, start, goal, detail_paths, informer, init_informer, system, dynamics, \
                           enforce_bounds, collision_check, traj_opt, jac_A, jac_B, step_sz=step_sz, MAX_LENGTH=1000)
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
