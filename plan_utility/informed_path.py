import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
import matplotlib.pyplot as plt
from visual.acrobot_vis import *
from sparse_rrt.systems.acrobot import Acrobot
# this one predicts one individual path using informer and trajopt
MAX_INVALID_THRESHOLD = 1.5
invalid_mat = np.diag([1.,1.,0.1,0.1])
def plan(obs, env, x0, xG, data, informer, init_informer, system, dynamics, enforce_bounds, IsInCollision, traj_opt, jac_A, jac_B, step_sz=0.02, MAX_LENGTH=1000):
    # informer: given (xt, x_desired) ->  x_t+1
    # jac_A: given (x, u) -> linearization A
    # jac B: given (x, u) -> linearization B
    # traj_opt: given (x0, x1) -> (xs, us, dts
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



    #update_line(hl, ax, x0.x)
    #draw_update_line(ax)
    for i in range(len(data)):
        update_line(hl, ax, data[i])
    draw_update_line(ax)
    update_line(hl_for, ax, x0.x)
    draw_update_line(ax)
    update_line(hl_back, ax, xG.x)
    draw_update_line(ax)
    #plt.waitforbuttonpress()

    itr=0
    target_reached=0
    tree=0
    time_norm = 0.
    start = x0
    goal = xG
    funnel_node = goal
    BVP_TOLERANCE = 1e-3
   
    for_global_scatter = []
    back_global_scatter = []
    for_local_scatter = []
    back_local_scatter = []

    while target_reached==0 and itr<MAX_LENGTH:
        # global waypoint plan
        for_global_xs = []
        back_global_xs = []
        global_for = [x0]
        global_back = [xG]
        for i in range(100):
            xw, x_init, u_init, t_init = informer(env, global_for[-1], global_back[0], direction=0)
            if not IsInCollision(xw.x):
                print('forward not in collision')
                global_for.append(xw)
                for_global_xs.append(xw.x)
                # if endpoint are close enough, then stop
                if node_nearby(global_for[-1].x, global_back[-1].x, invalid_mat, MAX_INVALID_THRESHOLD, system):
                    break
            xw, x_init, u_init, t_init = informer(env, global_back[-1], global_for[0], direction=1)
            if not IsInCollision(xw.x):
                print('backward not in collision')
                global_back.append(xw)
                back_global_xs.append(xw.x)
                # if endpoint are close enough, then stop
                if node_nearby(global_for[-1].x, global_back[-1].x, invalid_mat, MAX_INVALID_THRESHOLD, system):
                    break
        for_global_xs = np.array(for_global_xs)
        print(for_global_xs)
        back_global_xs = np.array(back_global_xs)
        for_global_scatter = ax.scatter(for_global_xs[:,0], for_global_xs[:,1], c='lightgreen')
        back_global_scatter = ax.scatter(back_global_xs[:,0], back_global_xs[:,1], c='red')
        draw_update_line(ax)
        # local plan and steerTo
        global_back.reverse()

        global_waypoints = global_for + global_back
        for_local_scatter = []
        
        local_for = [x0]
        for i in range(len(global_waypoints)):
            xw = global_waypoints[i]
            x, e = pathSteerToBothDir(x0, xw, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True)        
            if e is not None and not node_nearby(e.xs[0], x0.x, np.identity(len(x0.x)), BVP_TOLERANCE, system):           
                x, e = pathSteerToBothDir(x0, xw, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True)
            if e is None:
                # if collision happens, have two options: 1). replan from the x0 2). continue
                print('invalid...')
                # try replanning
                if len(local_for) >= 5:
                    x0 = local_for[-5]  # retrieve 5 positions
                    for j in range(1,5,1):
                        for_local_scatter[-j].remove()
                    for j in range(2,6,1):
                        remove_last_k(hl_for, ax, len(local_for[-j].edge.xs))
                else:
                    x0 = local_for[0]
                    for j in range(len(for_local_scatter)):
                        for_local_scatter[j].remove()
                    for j in range(0,len(for_local_scatter)-1):
                            remove_last_k(hl_for, ax, len(local_for[j].edge.xs))
                break
            # update
            for i in range(len(e.xs)):
                update_line(hl_for, ax, e.xs[i])
            xs_to_plot = np.array(e.xs[::10])
            for i in range(len(xs_to_plot)):
                xs_to_plot[i] = wrap_angle(xs_to_plot[i], system)
            for_prev_scat = ax.scatter(xs_to_plot[:,0], xs_to_plot[:,1], c='g')
            for_local_scatter.append(for_prev_scat)

            draw_update_line(ax)
            animation(e.xs, e.us)
            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            local_for.append(x0)
            # if reached the target
            if node_nearby(x.x, goal.x, goal.S0, goal.rho0, system):
                target_reached = True
                break
        # remove global scatter
        for_global_scatter.remove()
        back_global_scatter.remove()
        draw_update_line(ax)
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
        #lazyFunnel(start, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, system=system, step_sz=step_sz)
        #print(start.edge.rho0s)
        #funnelSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction=0, system=system, step_sz=step_sz)
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
