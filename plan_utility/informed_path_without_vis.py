import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
from sparse_rrt.systems.acrobot import Acrobot
# this one predicts one individual path using informer and trajopt
MAX_INVALID_THRESHOLD = .5
invalid_mat = np.diag([1.,1.,0.,0.])
def plan(obs, env, x0, xG, data, informer, init_informer, system, dynamics, enforce_bounds, IsInCollision, traj_opt, jac_A, jac_B, step_sz=0.02, MAX_LENGTH=1000):
    # informer: given (xt, x_desired) ->  x_t+1
    # jac_A: given (x, u) -> linearization A
    # jac B: given (x, u) -> linearization B
    # traj_opt: given (x0, x1) -> (xs, us, dts

    itr=0
    target_reached=0
    tree=0
    time_norm = 0.
    start = x0
    goal = xG
    funnel_node = goal
    BVP_TOLERANCE = 1e-3

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
            
            # try to connect from x0 to one node on the goal tree (nearest)
            node = xG
            min_node = node
            min_d = 1e6
            # min_d: normalized distance xTSx / (rho^2)
            while node is not None:
                dis = h_dist(x0, node, np.identity(len(node.x)), 2., system)
                if dis < min_d:
                    min_d = dis
                    min_node = node
                node = node.next
            
            xw, x_init, u_init, t_init = informer(env, x0, goal, direction=0)
            if IsInCollision(xw.x):
                tree = 1
                itr += 1
                continue
            x, e = pathSteerToBothDir(x0, xw, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=False)
            # if the bvp solver solution is too faraway, then ignore it
            # this is useful if we only use trajopt result without propagation
            if e is not None and not node_nearby(e.xs[0], x0.x, np.identity(len(x0.x)), BVP_TOLERANCE, system):
            #if e is not None and np.linalg.norm(e.xs[0] - x0.x) > BVP_TOLERANCE:
            #    # ignore it
                #print('forward searching bvp not successful.')
                #tree = 1
                #itr += 1
                #continue
                # then propagate it to obtain the result
                #x_init = e.xs
                #x_init[0] = x0.x
           
                x, e = pathSteerToBothDir(x0, xw, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True)

            if for_in_collision_nums[-1] >= 2 and x0.prev is not None:
                # backtrace, this include direct incollision nodes and indirect ones (parent)
                print('too many collisions... backtracing')
                # pop the last collision num
                for_in_collision_nums.pop(-1)
                # since the next state has collision, increase one for its parent
                for_in_collision_nums[-1] += 1
                # remove line as well
                x0 = x0.prev
                tree = 1
                itr += 1
                continue
            if e is None:
                # in collision
                for_in_collision_nums[-1] += 1
                tree = 1
                itr += 1
                continue

            
            
                

            # otherwise, create a new collision_num
            for_in_collision_nums.append(0)

            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            # if connected to the min_node, then reset goal tree
            if node_nearby(x.x, min_node.x, invalid_mat, MAX_INVALID_THRESHOLD, system):
                print('start tree is connected to one node in goal tree')
                if min_node.next is not None:
                    xG = min_node.next
                #plt.waitforbuttonpress()
                if min_node.next is None and node_nearby(x.x, min_node.x, min_node.S0, min_node.rho0, system):
                    target_reached = True
                    continue
                        
            xG = goal
            tree=1
        else:
            #tree=0
            # skip directly
            #continue
            # the informed initialization is in the forward direction
            #xG = goal
            xw, x_init, u_init, t_init = informer(env, xG, x0, direction=1)
            if IsInCollision(xw.x):
                tree = 0
                itr += 1
                continue
            xw.next = xG
            xG.prev = xw
            xG = xw
            node = xG
            min_node = node
            min_d = 1e6
            # min_d: normalized distance xTSx / (rho^2)
            while node is not None:
                # we need to make sure that current x0 is not near the node of interest,
                # to encourage exploration
                dis = h_dist(x0, node, np.identity(len(node.x)), MAX_INVALID_THRESHOLD, system)
                if dis < min_d and dis > 1.0:
                    min_d = dis
                    min_node = node
                node = node.next
            
            x_init, u_init, t_init = init_informer(env, x0, min_node, direction=0)
            x, e = pathSteerToBothDir(x0, min_node, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=False)
            if e is not None and not node_nearby(e.xs[0], x0.x, np.identity(len(x0.x)), BVP_TOLERANCE, system):
            #if e is not None and np.linalg.norm(e.xs[0] - x0.x) > BVP_TOLERANCE:
            #    # ignore it
                print('forward searching bvp not successful.')
                tree = 0
                itr += 1
                continue
                # then propagate it to obtain the result
                x, e = pathSteerToBothDir(x0, min_node, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True)

            if e is None:
                # in collision
                tree = 0
                itr += 1
                continue
            if h_dist(x, min_node, invalid_mat, MAX_INVALID_THRESHOLD, system) <= 1.0:
                print('start tree is connected to one node in goal tree')
                #if min_node.next is not None:
                #    xG = min_node.next
                xG = goal
                #plt.waitforbuttonpress()
                for_in_collision_nums.append(0)
                x0.next = x
                x.prev = x0
                e.next = x
                x0.edge = e
                x0 = x                
                xG = goal   
                print('endpoint distance to goal: %f' % (np.linalg.norm(min_node.x - goal.x)))
                if min_node.next is None and node_nearby(x.x, min_node.x, min_node.S0, min_node.rho0, system):
                #if np.linalg.norm(min_node.x - goal.x) <= 1e-4:
                    target_reached = True
                    continue
            tree=0

            
        # try connecting to goal
        x_init, u_init, t_init = init_informer(env, x0, goal, direction=0)
        xG_, e = pathSteerToBothDir(x0, goal, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, \
                                propagating=False, endpoint=True)
        if e is not None:
            # check if the BVP is successful
            if not node_nearby(e.xs[0], x0.x, np.identity(len(x0.x)), BVP_TOLERANCE, system):
            #if np.linalg.norm(e.xs[0] - x0.x) > BVP_TOLERANCE:
                print('forward searching bvp not successful.')
                #itr += 1
                #continue
                # try propagating
                xG_, e = pathSteerToBothDir(x0, goal, x_init, u_init, t_init, dynamics, enforce_bounds, IsInCollision, \
                                    jac_A, jac_B, traj_opt, step_sz=step_sz, system=system, direction=0, propagating=True, \
                                    endpoint=True)
                if e is None:
                    itr += 1
                    continue


            # add xG_ to the start tree
            x0.next = xG_
            xG_.prev = x0
            x0.edge = e
            x0.edge.next = xG_


            print('endpoint steering...')
            print('x0:')
            print(x0.x)
            print('goal:')
            print(goal.x)
            print('xG_:')
            print(xG_.x)
            # find the nearest point from xG_ to points on the goal tree
            if h_dist(xG_, goal, goal.S0, goal.rho0, system) <= 1.0:
                # otherwise it is a nearby node
                #if min_node.S0 is None:
                #    lazyFunnel(min_node, funnel_node, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, system=system, step_sz=step_sz)
                #    #funnel_node = min_node
                min_node = goal
                reached, node_i0, node_i1 = nearby(xG_, min_node, system)
                if reached:
                    target_reached = True
                    #ax.scatter(e.xs[::10,0], e.xs[::10,1], c='salmon')
                    if min_node.edge is not None:
                        # need to extract subset of x1.edge
                        # change the t0 of edge starting from node to be time_knot[node_i] (use subset of edge)
                        edge = min_node.edge
                        edge.t0 = edge.time_knot[node_i1]
                        edge.i0 = node_i1
                        # change the node to be xs[node_i], S0 to be S(time_knot[node_i]), rho0 to be rho0s[node_i]
                        new_node = Node(wrap_angle(edge.xs[node_i1], system))
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
        #print('target reached.')
        # it is near enough, so we connect in the node data structure from x0 to xG, although the endpoint of x0.edge
        # in state is still xG_
        x0 = x0.prev  # since the x0 can directly connect to xG, we only need to set the next state of the previous x to xG
        x0.next = xG  # update endpoint (or should I?)
        x0.edge.next = xG
        xG.prev = x0
        # connect the lsat node
        # construct the funnel later
        # connect from x0 to xG, the endpoint of x0 is xG_, but it is near xG
        #print('before funnelsteerto')
        #lazyFunnel(start, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, system=system, step_sz=step_sz)
        #print(start.edge.rho0s)
        #funnelSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction=0, system=system, step_sz=step_sz)
        #print('after funnelsteerto')

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
    return target_reached, path_list
