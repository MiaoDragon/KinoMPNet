import sys
sys.path.append('..')

import numpy as np
from plan_utility.plan_general import *
# this one predicts one individual path using informer and trajopt
def plan(env, x0, xG, informer, dynamics, traj_opt, jac_A, jac_B, MAX_LENGTH=1000):
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
        if tree==0:
            # since we ensure each step we can steer to the next waypoint
            # the edge connecting the two nodes will store the trajectory
            # information, TVLQR and the funnel size factors
            # the edge information is stored at the endpoint
            # here direciton=0 means we are computing forward steer, and 1 means
            # we are computing backward
            x, e = pathSteerTo(x0, informer(env, x0, xG, direction=0), dynamics=dynamics, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, direction=0)
            x0.next = x
            x.prev = x0
            e.next = x
            x0.edge = e
            x0 = x
            tree=1
        else:
            x, e = pathSteerTo(xG, informer(env, xG, x0, direction=1), dynamics=dynamics, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, direction=1)
            x.next = xG
            xG.prev = x
            e.next = xG
            x.edge = e
            xG = x
            tree=0
        xG_, e_ = pathSteerTo(x0, xG, dynamics=dynamics, traj_opt=traj_opt, jac_A=jac_A, jac_B=jac_B, direction=0)
        target_reached = nearby(xG_, xG)  # check the funnel if can connect
    if target_reached:
        # connect the lsat node
        xG_.next = xG
        e_.next = xG
        xG_.edge = e_
    return target_reached
