import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
import matplotlib.pyplot as plt
from visual.acrobot_vis import *
from sparse_rrt.systems.acrobot import Acrobot
from plan_utility.data_structure import *

import torch
from torch.autograd import Variable
#from utility import *
import time
DEFAULT_STEP = 2.

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
        num_steps = int(np.round(dt / step_sz))
        last_step = dt - num_steps*step_sz

        for k in range(num_steps):
            x = dynamics(x, u, step_sz)
            if IsInCollision(x):
                # the ccurrent state is in collision, abort
                #print('collision, i=%d, num_steps=%d' % (i, k))
                valid = False
                print('in collision')
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
            #break
    new_xs = np.array(new_xs)
    new_us = np.array(new_us)
    new_dts = np.array(new_dts)
    return new_xs, new_us, new_dts, valid

def pathSteerTo(x0, x1, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, traj_opt, direction, system=None, step_sz=0.002, num_steps=21, propagating=False, endpoint=False):
    # direciton 0 means forward from x0 to x1
    # direciton 1 means backward from x0 to x1
    # jac_A: given x, u -> linearization A
    # jac_B: given x, u -> linearization B
    # traj_opt: a function given two endpoints x0, x1, compute the optimal trajectory
    if direction == 0:
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
    else:
        xs, us, dts = traj_opt(x1.x, x0.x, step_sz, num_steps, x_init, u_init, t_init)
        if len(us) == len(xs):
            us = us[:-1]
        if len(dts) == len(xs):
            dts = dts[:-1]
        if propagating:
            us = np.flip(us, axis=0)
            dts = np.flip(dts, axis=0)
            # reversely propagate the system
            xs, us, dts, valid = propagate(x0.x, us, dts, dynamics=lambda x, u, t: -dynamics(x, u, t), enforce_bounds=enforce_bounds, IsInCollision=IsInCollision, system=system, step_sz=step_sz)
            xs = np.flip(xs, axis=0)
            us = np.flip(us, axis=0)
            dts = np.flip(dts, axis=0)
        else:
            # check collision for the trajopt endpoint
            valid = True
            for i in range(len(xs)):
                if IsInCollision(xs[i]):
                    valid = False
                    print('in collision')
                    break            
        edge_dt = np.sum(dts)
        start = Node(wrap_angle(xs[0], system))  # after flipping, the first in xs is the start
        goal = x0
        x1 = start
    # after trajopt, make actions of dimension 2
    us = us.reshape(len(us), -1)
    # notice that controller time starts from 0, hence locally need to shift the time by minusing t0_edges
    # start from 0
    time_knot = np.cumsum(dts)
    time_knot = np.insert(time_knot, 0, 0.)
    # can also change the resolution by the following function (for instance, every 10)
    edge = Edge(xs, us, dts, time_knot, edge_dt)
    edge.next = goal
    start.edge = edge
    start.next = goal
    goal.prev = start
    return x1, edge, valid

#def MPNetSteerTo(

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


def node_d(x0, x1, S, system):
    # given two nodes, S and rho, check if x0 is near x1
    delta_x = x0 - x1
    # this is pendulum specific. For other envs, need to do similar things
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
    xTSx = delta_x.T@S@delta_x# / lam_S
    return xTSx





def feasibility_check(path, obc, IsInCollision, system):
    # check endpoint if near
    return goal_check(path[-2], path[-1])
    
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
    print('goal endpoint distance: %f' % (dist))
    if dist <= goal_radius:
        return 1
    else:
        return 0
    
    
def removeCollision(path, obc, IsInCollision):
    return path
    new_path = []
    # rule out nodes that are already in collision
    for i in range(0,len(path)):
        if not IsInCollision(path[i].numpy(),obc):
            new_path.append(path[i])
    return new_path

def neural_replan(mpNet1, mpNet2, path, goal_node, obc, obs, IsInCollision, normalize, unnormalize, init_plan_flag, step_sz, num_steps, informer, init_informer, system, dynamics, enforce_bounds, traj_opt, data):
    if init_plan_flag:
        # if it is the initial plan, then we just do neural_replan
        MAX_LENGTH = 30
        #MAX_LENGTH = 3000
        mini_path = neural_replanner(mpNet1, mpNet2, path[0], path[-1], goal_node, obc, obs, IsInCollision, \
                                    normalize, unnormalize, MAX_LENGTH, step_sz, num_steps, \
                                    informer, init_informer, system, dynamics, enforce_bounds, traj_opt, data)
        return removeCollision(mini_path, obc, IsInCollision)
    MAX_LENGTH = 30
    #MAX_LENGTH = 3000
    # replan segments of paths
    new_path = [path[0]]
    # directly connecting the endpoint
    mini_path = neural_replanner(mpNet1, mpNet2, path[-2], path[-1], goal_node, obc, obs, IsInCollision, \
                                 normalize, unnormalize, MAX_LENGTH, step_sz, num_steps, \
                                informer, init_informer, system, dynamics, enforce_bounds, traj_opt, data)
    path = path[:-2] + mini_path
    return path

def neural_replanner(mpNet1, mpNet2, start_node, goal_node, real_goal_node, obc, obs, IsInCollisionWithObs, normalize, unnormalize, MAX_LENGTH, step_sz, num_steps, informer, init_informer, system, dynamics, enforce_bounds, traj_opt, data):
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
    feasible_points = np.array(feasible_points)
    infeasible_points = np.array(infeasible_points)
    ax.scatter(feasible_points[:,0], feasible_points[:,1], c='yellow')
    ax.scatter(infeasible_points[:,0], infeasible_points[:,1], c='pink')
    for i in range(len(data)):
        update_line(hl, ax, data[i])
    draw_update_line(ax)
    update_line(hl_for, ax, start_node.x)
    draw_update_line(ax)
    update_line(hl_back, ax, goal_node.x)
    draw_update_line(ax)
    
    
    # plan a mini path from start to goal
    # obs: tensor
    start = start_node.x
    goal = goal_node.x
    itr=0
    pA=[]
    pA.append(start)
    pB=[]
    pB.append(goal)
    target_reached=0
    tree=0
    new_path = []
    
    # extract the state for the node
    start = torch.from_numpy(start_node.x).type(torch.FloatTensor)
    goal = torch.from_numpy(goal_node.x).type(torch.FloatTensor)
    #--- compute MPNet waypoints connecting start and goal
    while target_reached==0 and itr<MAX_LENGTH*4:
        itr=itr+1  # prevent the path from being too long
        if tree==0:
            ip1 = torch.cat((start, goal)).unsqueeze(0)
            ob1 = torch.FloatTensor(obc).unsqueeze(0)
            #ip1=torch.cat((obs,start,goal)).unsqueeze(0)
            time0 = time.time()
            ip1=normalize(ip1)
            ip1=to_var(ip1)
            ob1=to_var(ob1)
            waypoint=mpNet1(ip1,ob1).squeeze(0)
            # unnormalize to world size
            waypoint=waypoint.data.cpu()
            time0 = time.time()
            waypoint = unnormalize(waypoint).numpy()
            waypoint = wrap_angle(waypoint, system)
            waypoint = torch.from_numpy(waypoint).type(torch.FloatTensor)
            if IsInCollision(waypoint.numpy()):
                #tree=1
                continue
            start = waypoint
            pA.append(start.numpy())
            tree=1
        else:
            #tree=0
            #continue
            ip2 = torch.cat((goal, start)).unsqueeze(0)
            ob2 = torch.FloatTensor(obc).unsqueeze(0)
            #ip2=torch.cat((obs,goal,start)).unsqueeze(0)
            time0 = time.time()
            ip2=normalize(ip2)
            ip2=to_var(ip2)
            ob2=to_var(ob2)
            waypoint=mpNet2(ip2,ob2).squeeze(0)
            # unnormalize to world size
            waypoint=waypoint.data.cpu()
            time0 = time.time()
            waypoint = unnormalize(waypoint).numpy()
            waypoint = wrap_angle(waypoint, system)
            waypoint = torch.from_numpy(waypoint).type(torch.FloatTensor)
            if IsInCollision(waypoint.numpy()):
                #tree=1
                continue
            goal = waypoint
            pB.append(goal.numpy())
            tree=0
        #target_reached=MPNetSteerTo(start, goal, )
        target_reached=node_nearby(start.numpy(), goal.numpy(), np.diag([1.,1.,0.1,0.1]), 2., system)
        # might have some terminal conditions for MPNet endpoints

    #--- after MPNet waypoints, compute actual connection
    pB.reverse()
    # visualize pA and pB
    pA = np.array(pA)
    pB = np.array(pB)
    print('pa:')
    print(pA)
    print('pb:')
    print(pB)
    pA_scat = ax.scatter(pA[:,0], pA[:,1], c='lightgreen')
    pB_scat = ax.scatter(pB[:,0], pB[:,1], c='red')
    draw_update_line(ax)

    
    mpnet_path = np.concatenate([pA,pB], axis=0)
    
    # each time find nearest waypoint, and connect them
    node = start_node
    current_idx = 0
    node_list = [node]
    while current_idx < len(mpnet_path)-1:
        # try connecting to goal
        x_init, u_init, t_init = init_informer(obs, node, goal_node, direction=0)
        next_node, edge, cf = pathSteerTo(node, goal_node, x_init, u_init, t_init, \
                                       dynamics, enforce_bounds, IsInCollision, traj_opt, 0, system,
                                       step_sz=step_sz, num_steps=num_steps, propagating=True, endpoint=True)        
        if cf and goal_check(next_node, real_goal_node):
            node.next = next_node
            next_node.prev = node
            node.edge = edge
            edge.next = next_node
            node_list.append(next_node)
            break
        
        
        # find the min node
        min_d = 1e8
        min_i = -1
        for i in range(current_idx, len(mpnet_path)):
            if node_d(node.x, mpnet_path[i], np.diag([1.,1.,0.5,0.5]), system) <= min_d:
                min_d = node_d(node.x, mpnet_path[i], np.diag([1.,1.,0.5,0.5]), system)
                min_i = i
        if min_i == len(mpnet_path)-1:
            min_i = min_i - 1
        if min_d > 1.:
            print('too far')
            print(min_d)
            node_list = node_list[:-3]
            break            
        print('min_i:%d, len(mpnet_path):%d' % (min_i, len(mpnet_path)))
        # connect to min_i+1
        x_init, u_init, t_init = init_informer(obs, node, Node(mpnet_path[min_i+1]), direction=0)
        next_node, edge, cf = pathSteerTo(node, Node(mpnet_path[min_i+1]), x_init, u_init, t_init, \
                                       dynamics, enforce_bounds, IsInCollision, traj_opt, 0, system,
                                       step_sz=step_sz, num_steps=num_steps, propagating=False, endpoint=False)
        if not node_nearby(edge.xs[0], node.x, np.identity(len(node.x)), 1e-2, system):
            next_node, edge, cf = pathSteerTo(node, Node(mpnet_path[min_i+1]), x_init, u_init, t_init, \
                                           dynamics, enforce_bounds, IsInCollision, traj_opt, 0, system,
                                           step_sz=step_sz, num_steps=num_steps, propagating=True, endpoint=False)            
        for i in range(len(edge.xs)):
            update_line(hl_for, ax, edge.xs[i])
        xs_to_plot = np.array(edge.xs[::10])
        for i in range(len(xs_to_plot)):
            xs_to_plot[i] = wrap_angle(xs_to_plot[i], system)
        ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='g')
        draw_update_line(ax)
        animation(edge.xs, edge.us)
        
        # might add some terminal condition: if too faraway OR collision
        if not cf:
            print('in collision as cf is false')
            node_list = node_list[:-3]  # delete several nodes
            break
        if min_i+1 == len(mpnet_path)-1 and not goal_check(next_node, real_goal_node):
            print('endpoint not at goal')
            node_list = node_list[:-1]  # remove last node
            break
        #next_node = Node(next_x)
        node.edge = edge
        node.next = next_node
        edge.next = next_node
        next_node.prev = node
        node = next_node
        current_idx = min_i+1
        node_list.append(node)
    print('neural replanner over')
    print('len of node_list: %d' % (len(node_list)))
    node_list.append(goal_node)
    plt.waitforbuttonpress()
    pA_scat.remove()
    pB_scat.remove()
    return node_list

