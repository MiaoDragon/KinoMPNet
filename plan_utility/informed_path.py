import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
import matplotlib.pyplot as plt
# this one predicts one individual path using informer and trajopt
def plan(env, x0, xG, data, informer, system, dynamics, enforce_bounds, traj_opt, jac_A, jac_B, step_sz=0.02, MAX_LENGTH=1000):
    # informer: given (xt, x_desired) ->  x_t+1
    # jac_A: given (x, u) -> linearization A
    # jac B: given (x, u) -> linearization B
    # traj_opt: given (x0, x1) -> (xs, us, dts
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    #hl_real, = ax.plot([], [], 'r')
    hl_for, = ax.plot([], [], 'g')
    hl_back, = ax.plot([], [], 'r')
    hl_for_mpnet, = ax.plot([], [], 'lightgreen')
    hl_back_mpnet, = ax.plot([], [], 'salmon')

    def update_line(h, ax, new_data):
        h.set_data(np.append(h.get_xdata(), new_data[0]), np.append(h.get_ydata(), new_data[1]))
        #h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        #h.set_ydata(np.append(h.get_ydata(), new_data[1]))


    def draw_update_line(ax):
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
        #plt.show()
    #update_line(hl, ax, x0.x)
    #draw_update_line(ax)
    for i in range(len(data)):
        update_line(hl, ax, data[i])
    draw_update_line(ax)
    update_line(hl_for, ax, x0.x)
    draw_update_line(ax)
    update_line(hl_back, ax, xG.x)
    draw_update_line(ax)
    plt.waitforbuttonpress()

    itr=0
    target_reached=0
    tree=0
    time_norm = 0.
    start = x0
    goal = xG
    funnel_node = goal
    while target_reached==0 and itr<MAX_LENGTH:
        itr=itr+1  # prevent the path from being too long
        print('iter: %d' % (itr))
        if tree==0:
            # since we ensure each step we can steer to the next waypoint
            # the edge connecting the two nodes will store the trajectory
            # information, TVLQR and the funnel size factors
            # the edge information is stored at the endpoint
            # here direciton=0 means we are computing forward steer, and 1 means
            # we are computing backward
            xw = informer(env, x0, xG, direction=0)
            x, e = pathSteerTo(x0, xw, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz=step_sz, direction=0)
            for i in range(len(e.xs)):
                update_line(hl_for, ax, e.xs[i])
            ax.scatter(e.xs[::10,0], e.xs[::10,1], c='g')
            draw_update_line(ax)
            plt.waitforbuttonpress()
            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            tree=1
        else:
            xw = informer(env, xG, x0, direction=1)
            x, e = pathSteerTo(xG, xw, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz=step_sz, direction=1)
            for i in range(len(e.xs)):
                update_line(hl_back, ax, e.xs[i])
            ax.scatter(e.xs[::10,0], e.xs[::10,1], c='r')
            draw_update_line(ax)
            plt.waitforbuttonpress()
            x.next = xG
            xG.prev = x
            e.next = xG
            x.edge = e
            xG = x
            tree=0

        # steer endpoint
        xG_, e = pathSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz=step_sz, direction=0)
        # add xG_ to the start tree
        x0.next = xG_
        xG_.prev = x0
        x0.edge = e
        x0.edge.next = xG_
        

        print('endpoint steering...')
        print('x0:')
        print(x0.x)
        print('xG:')
        print(xG.x)
        print('xG_:')
        print(xG_.x)
        ax.scatter(e.xs[::10,0], e.xs[::10,1], c='salmon')
        draw_update_line(ax)
        # find the nearest point from xG_ to points on the goal tree
        node = xG
        min_node = node
        min_d = 1e6
        # min_d: normalized distance xTSx / (rho^2)
        while node is not None:
            dis = h_dist(xG_, node, goal.S0, goal.rho0, system)
            if dis < min_d:
                min_d = dis
                min_node = node
            node = node.next
        if min_d > 1.0:
            itr += 1
            continue
        # otherwise it is a nearby node
        if min_node.S0 is None:
            lazyFunnel(min_node, funnel_node, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, system=system, step_sz=step_sz)
            funnel_node = min_node

        reached, node_i0, node_i1 = nearby(xG_, min_node, system)
        if reached:
            target_reached = True
            # since the nearby nodes are on the edge, need to extract the node index
            if min_node.edge is not None:
                # need to extract subset of x1.edge
                # change the t0 of edge starting from node to be time_knot[node_i] (use subset of edge)
                edge = min_node.edge
                edge.t0 = edge.time_knot[node_i1]
                edge.i0 = node_i1
                # change the node to be xs[node_i], S0 to be S(time_knot[node_i]), rho0 to be rho0s[node_i]
                new_node = Node(edge.xs[node_i1])
                new_node.S0 = edge.S(edge.t0).reshape((len(edge.xs[node_i1]),len(edge.xs[node_i1])))
                new_node.rho0 = edge.rho0s[node_i1]
                new_node.edge = edge
                new_node.next = min_node.next
                min_node = new_node
            xG = min_node
            x0 = xG_
            break
        itr += 1

    if target_reached:
        print('target reached.')
        # it is near enough, so we connect in the node data structure from x0 to xG, although the endpoint of x0.edge
        # in state is still xG_
        x0 = x0.prev  # since the x0 can directly connect to xG, we only need to set the next state of the previous x to xG
        x0.next = xG  # update endpoint (or should I?)
        x0.edge.next = xG
        xG.prev = x0
        # connect the lsat node
        # construct the funnel later
        # connect from x0 to xG, the endpoint of x0 is xG_, but it is near xG
        print('before funnelsteerto')
        funnelSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction=0, system=system, step_sz=step_sz)
        print('after funnelsteerto')

        #xG_.next = xG
        #e_.next = xG
        #xG_.edge = e_
    else:
        x0.next = None
        x0.edge = None

    # construct a list of the path
    path_list = []
    node = start
    while node is not None:
        path_list.append(node.x)
        node = node.next
    if not target_reached:
        # xG is the first in the goal tree
        while xG is not None:
            path_list.append(xG.x)
            xG = xG.next
    plt.cla()
    return target_reached, path_list
