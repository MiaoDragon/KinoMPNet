import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
# this one predicts one individual path using informer and trajopt
def plan(env, x0, xG, informer, dynamics, enforce_bounds, traj_opt, jac_A, jac_B, step_sz=0.02, MAX_LENGTH=1000):
    # informer: given (xt, x_desired) ->  x_t+1
    # jac_A: given (x, u) -> linearization A
    # jac B: given (x, u) -> linearization B
    # traj_opt: given (x0, x1) -> (xs, us, dts)
    itr=0
    target_reached=0
    tree=0
    time_norm = 0.
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
            x, e = pathSteerTo(x0, informer(env, x0, xG, direction=0), dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=0, compute_funnel=True)
            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            tree=1
            print('after forward steering:')
            print('state:')
            print(x.x)
            node = xG
            while node is not None:
                target_reached = nearby(x0, node)
                if target_reached:
                    xG = node
                    break
                node = node.next

        else:
            x, e = pathSteerTo(xG, informer(env, xG, x0, direction=1), dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=1, compute_funnel=True)
            x.next = xG
            xG.prev = x
            e.next = xG
            x.edge = e
            xG = x
            tree=0
            print('after backward steering:')
            print('state:')
            print(x.x)
            node = x0
            while node is not None:
                target_reached = nearby(node, xG)
                if target_reached:
                    x0 = node
                    break
                node = node.prev


        #xG_, e_ = pathSteerTo(x0, xG, dynamics=dynamics, enforce_bounds=enforce_bounds, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, step_sz=step_sz, direction=0, compute_funnel=False)
        # check if x0 can connect to one node in the backward tree directly, if so, no need to construct a controller from x0 to the node
        # version one: only check endpoint
        #target_reached = nearby(x0, xG)  # check the funnel if can connect
        # version two: new node in start tree: check all goal tree, and otherwise conversely
    if target_reached:
        # it is near enough, so we connect in the node data structure from x0 to xG, although the endpoint of x0.edge
        # in state is still xG_
        x0 = x0.prev  # since the x0 can directly connect to xG, we only need to set the next state of the previous x to xG
        x0.next = xG  # update endpoint (or should I?)
        x0.edge.next = xG
        xG.prev = x0
        # connect the lsat node
        # construct the funnel later
        # connect from x0 to xG, the endpoint of x0 is xG_, but it is near xG
        funnelSteerTo(x0, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direciton=0, step_sz=step_sz)
        #xG_.next = xG
        #e_.next = xG
        #xG_.edge = e_
    else:
        x0.next = None
        x0.edge = None
    return target_reached
