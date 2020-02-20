import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
import matplotlib.pyplot as plt
from visual.acrobot_vis import *
from sparse_rrt.systems.acrobot import Acrobot
# this one predicts one individual path using informer and trajopt
def plan_mpnet(obs, env, x0, xG, data, informer, init_informer, system, dynamics, enforce_bounds, IsInCollision, traj_opt, jac_A, jac_B, step_sz=0.02, MAX_LENGTH=1000):
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
        print(data[i])
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
    BVP_TOLERANCE = 1e-6

    for_in_collision_nums = [0]  # forward in collision number
    for_prev_scatter = []
    back_in_collision_nums = [0]
    back_prev_scatter = []
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

            # the informed initialization is in the forward direction
            xw, x_init, u_init, t_init = informer(env, x0, goal, direction=0)
            
            ax.scatter(xw.x[0], xw.x[1], c='lightgreen')
            draw_update_line(ax)
            plt.waitforbuttonpress()
            x0 = Node(xw.x)
            tree=1
        else:
            xw, x_init, u_init, t_init = informer(env, xG, start, direction=1)
            ax.scatter(xG.x[0], xG.x[1], c='black')
            ax.scatter(start.x[0], start.x[1], c='blue')
            
            ax.scatter(xw.x[0], xw.x[1], c='red')
            print(xw.x)
            draw_update_line(ax)
            plt.waitforbuttonpress()
            xG = Node(xw.x)
            tree=0
