import sys
sys.path.append('..')

import numpy as np
#from plan_utility.plan_general import *
import matplotlib.pyplot as plt
from visual.acrobot_vis import *
from sparse_rrt.systems.acrobot import Acrobot
from iterative_plan_and_retreat.data_structure import *

import torch
from torch.autograd import Variable
#from utility import *
import time


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)



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

def goal_check(node, goal):
    LENGTH = 20.
    point1 = node.x
    point2 = goal.x
    x = np.cos(point1[0] - np.pi / 2)+np.cos(point1[0] + point1[1] - np.pi / 2)
    y = np.sin(point1[0] - np.pi / 2)+np.sin(point1[0] + point1[1] - np.pi / 2)
    x2 = np.cos(point2[0] - np.pi / 2)+np.cos(point2[0] + point2[1] - np.pi / 2)
    y2 = np.sin(point2[0] - np.pi / 2)+np.sin(point2[0] + point2[1] - np.pi / 2)
    dist = LENGTH*np.sqrt((x-x2)**2+(y-y2)**2)
    goal_radius = 2.0
    #print('goal endpoint distance: %f' % (dist))
    if dist <= goal_radius:
        return 1
    else:
        return 0

def node_nearby(x0, x1, S, rho, system):
    # state x0 to state x1
    delta_x = x0 - x1
    circular = system.is_circular_topology()
    for i in range(len(delta_x)):
        if circular[i]:
            # if it is angle
            # should not change the "sign" of the delta_x
            # map to [-pi, pi]
            delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
            # should not change the "sign" of the delta_x
            if delta_x[i] > np.pi:
                delta_x[i] = delta_x[i] - 2*np.pi
    #lam_S = np.linalg.eigvals(S).max()
    xTSx = delta_x.T@S@delta_x# / lam_S
    # here we use a safe threshold (0.9)
    if xTSx <= rho*rho:
        return True
    else:
        pass
    return False



def propagate(x, us, dts, dynamics, enforce_bounds, IsInCollision, system=None, step_sz=None):
    # use the dynamics to interpolate the state x
    # can implement different interpolation method for this
    # ADDED: notice now for circular cases, we use the unmapped angle to ensure smoothness

    # change propagation: maybe only using step_sz but not smaller is better (however, round for accuracy)

    # try to round according to control up to difference being some threshold
    new_xs = [x]
    new_us = []
    new_dts = []
    valid = True  # collision free
    for i in range(len(us)):
        dt = dts[i]
        u = us[i]
        num_steps = int(np.floor(dt / step_sz))
        last_step = dt - num_steps*step_sz
        print('propagating...')
        print('num_steps: %d' % (num_steps))
        print('last_step: %f' % (last_step))
        for k in range(num_steps):
            x = dynamics(x, u, step_sz)
            if IsInCollision(x):
                # the ccurrent state is in collision, abort
                #print('collision, i=%d, num_steps=%d' % (i, k))
                valid = False
                print('in collision')
                break
                #break
            new_xs.append(x)
            new_us.append(u)
            new_dts.append(step_sz)
            #print('appended, i=%d' % (i))
        #if not valid:
        #    break
        # here we apply round to last_step as in SST we use this method
        #if last_step > step_sz/2:
        if True:
            #last_step = step_sz
            #x = x + last_step*dynamics(x, u)
            x = dynamics(x, u, last_step)
            new_xs.append(x)
            new_us.append(u)
            new_dts.append(last_step)
        if IsInCollision(x):
            #print('collision, i=%d' % (i))
            valid = False
            print('in collision')
            break
            #break
    new_xs = np.array(new_xs)
    new_us = np.array(new_us)
    new_dts = np.array(new_dts)
    print('propagation output:')
    print('xs:')
    print(new_xs)
    print('us:')
    print(new_us)
    print('dts:')
    print(new_dts)
    return new_xs, new_us, new_dts, valid


def pathSteerTo(x0, x1, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, traj_opt, direction, system=None, step_sz=0.002, num_steps=21, propagating=False, endpoint=False):
    # direciton 0 means forward from x0 to x1
    # direciton 1 means backward from x0 to x1
    # jac_A: given x, u -> linearization A
    # jac_B: given x, u -> linearization B
    # traj_opt: a function given two endpoints x0, x1, compute the optimal trajectory
    xs, us, dts = traj_opt(x0.x, x1.x, step_sz, num_steps, x_init, u_init, t_init)
    print('trajopt output:')
    print('xs:')
    print(xs)
    print('us:')
    print(us)
    print('dts:')
    print(dts)
    # ensure us and dts have length 1 less than xs
    if len(us) == len(xs):
        us = us[:-1]
    if propagating:
        xs, us, dts, valid = propagate(x0.x, us, dts, dynamics=dynamics, enforce_bounds=enforce_bounds, IsInCollision=IsInCollision, system=system, step_sz=step_sz)
    else:
        # check collision for the trajopt endpoint
        valid = True
        for i in range(len(xs)):
            if IsInCollision(xs[i]):
                valid = False
                print('in collision')
                break
    edge_dt = np.sum(dts)
    start = x0
    goal = Node(wrap_angle(xs[-1], system))
    x1 = goal

    # after trajopt, make actions of dimension 2
    us = us.reshape(len(us), -1)
    # notice that controller time starts from 0, hence locally need to shift the time by minusing t0_edges
    # start from 0
    time_knot = np.cumsum(dts)
    time_knot = np.insert(time_knot, 0, 0.)
    # can also change the resolution by the following function (for instance, every 10)
    if len(us) == 0:
        edge = None
    else:
        edge = Edge(xs, us, dts, time_knot, edge_dt)
        edge.next = goal
        start.edge = edge
        start.next = goal
        goal.prev = start
    return x1, edge, valid


def explored(x, explored_nodes, system):
    for i in range(len(explored_nodes)):
        S = np.identity(4)
        rho = 1.0
        if node_nearby(x.x, explored_nodes[i].x, S, rho, system):
            return True
    return False

def plan(obs, env, x0, xG, data, informer, init_informer, system, dynamics, enforce_bounds, IsInCollision, traj_opt, step_sz=0.02, MAX_LENGTH=1000):
    """
    For each node xt, we record how many explorations have been made from xt. We do planning according to the following rules:
    1. if n_explored >= n_max_explore:
        if not the root, backtrace to the previous node

    hat(x)_t+1 = informer(xt, G)
    x_t+1, tau = BVP(xt, hat(x)_t+1)
    n_exlored(x_t) += 1
    2. if explored(x_t+1) (w.r.t. some regions as defined in SST paper)
        ignore x_t+1
        # may also update the "score" using the already explored region
    3. if too_far(x_t+1, hat(x)_t+1)
        ignore x_t+1
    4. if inCollision(x_t+1)
        ignore x_t+1
        # may also update the "score" of points

    # otherwise it is safe
    establish the connections between x_t and x_t+1,
    move to x_t+1
    """

    # visualization
    # visualization
    print('step_sz: %f' % (step_sz))
    params = {}
    params['obs_w'] = 6.
    params['obs_h'] = 6.
    params['integration_step'] = step_sz
    vis = AcrobotVisualizer(Acrobot(), params)
    vis.obs = obs
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    #hl_real, = ax.plot([], [], 'r')
    hl_for, = ax.plot([], [], 'g')
    hl_back, = ax.plot([], [], 'r')
    hl_for_mpnet, = ax.plot([], [], 'lightgreen')
    hl_back_mpnet, = ax.plot([], [], 'salmon')

    ax_ani = fig.add_subplot(122)
    vis._init(ax_ani)

    print(obs)
    def update_line(h, ax, new_data):
        new_data = wrap_angle(new_data, system)
        h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
        #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        #h.set_ydata(np.append(h.get_ydata(), new_data[1]))

    def remove_last_k(h, ax, k):
        h.set_data(h.get_xdata()[:-k], h.get_ydata()[:-k])

    def draw_update_line(ax):
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.show()

    def animation(states, actions):
        vis._animate(states[-1], ax_ani)
        draw_update_line(ax_ani)
    dtheta = 0.1
    feasible_points = []
    infeasible_points = []
    goal_region = []
    imin = 0
    imax = int(2*np.pi/dtheta)
    obs_width = 6.
    new_obs_i = []
    for k in range(len(obs)):
        obs_pt = []
        obs_pt.append(obs[k][0]-obs_width/2)
        obs_pt.append(obs[k][1]-obs_width/2)
        obs_pt.append(obs[k][0]-obs_width/2)
        obs_pt.append(obs[k][1]+obs_width/2)
        obs_pt.append(obs[k][0]+obs_width/2)
        obs_pt.append(obs[k][1]+obs_width/2)
        obs_pt.append(obs[k][0]+obs_width/2)
        obs_pt.append(obs[k][1]-obs_width/2)
        new_obs_i.append(obs_pt)

    IsInCollision = lambda x: IsInCollisionWithObs(x, new_obs_i)
    for i in range(imin, imax):
        for j in range(imin, imax):
            x = np.array([dtheta*i-np.pi, dtheta*j-np.pi, 0., 0.])
            if IsInCollision(x):
                infeasible_points.append(x)
            else:
                feasible_points.append(x)
                if goal_check(Node(x), real_goal_node):
                    goal_region.append(x)
    feasible_points = np.array(feasible_points)
    infeasible_points = np.array(infeasible_points)
    goal_region = np.array(goal_region)
    ax.scatter(feasible_points[:,0], feasible_points[:,1], c='yellow')
    ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c='pink')
    ax.scatter(goal_region[:,0], goal_region[:,1], c='green')
    for i in range(len(data)):
        update_line(hl, ax, data[i])
    draw_update_line(ax)
    update_line(hl_for, ax, start_node.x)
    draw_update_line(ax)
    update_line(hl_back, ax, goal_node.x)
    draw_update_line(ax)





    explored_nodes = [x0]
    xt = x0
    max_explore = 10
    xt.n_explored = 0
    xw_scat = None
    while True:
        if xw_scat is not None:
            xw_scat.remove()
        if xt.n_exlored >= max_explore and xt is not x0:
            xt = xt.prev
            continue
        xw, x_init, u_init, t_init = informer(env, xt, xG, direction=0)
        xw_scat = ax.scatter(xw.x[0], xw.x[1], c='lightgreen')
        draw_update_line(ax)

        x_t_1, edge, valid = pathSteerTo(xt, xw, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True)

        for i in range(len(edge.xs)):
            update_line(hl_for, ax, edge.xs[i])
        xs_to_plot = np.array(edge.xs[::10])
        for i in range(len(xs_to_plot)):
            xs_to_plot[i] = wrap_angle(xs_to_plot[i], system)
        ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='g')
        draw_update_line(ax)
        animation(edge.xs, edge.us)
        # stop at the last node that is not in collision

        xt.n_explored += 1
        if edge is None:  # when the immediate next node is in collision
            continue
        if explored(x_t_1, explored_nodes, system):
            # based on some defined distance
            continue
        # if too far...

        # establish connections to x_t+1
        xt.next = x_t_1
        x_t_1.prev = xt
        xt.edge = edge
        xt.edge.next = x_t_1
        x_t_1.n_explored = 0
        explored_nodes.append(x_t_1)
        xt = x_t_1

        # check if the new node is near goal
        if goal_check(xt, xG):
            print("success")
            break
