import sys
sys.path.append('..')

import numpy as np
from tvlqr.python_tvlqr import tvlqr
from tvlqr.python_lyapunov import sample_tv_verify
from plan_utility.data_structure import *
def propagate(x, us, dts, dynamics, step_sz=None):
    # use the dynamics to interpolate the state x
    # can implement different interpolation method for this
    xs = [x]
    for i in range(len(us)):
        dt = dts[i]
        u = us[i]
        num_steps = int(dt / step_sz)
        last_step = dt - num_steps*step_sz
        for k in range(len(num_steps)):
            x = x + step_sz*dynamics(x, u)
            xs.append(x)
            us.append(u)
            dts.append(dt)
        x = x + last_step*dynamics(x, u)
        xs.append(x)
        us.append(u)
        dts.append(last_step)
    xs = np.array(xs)
    us = np.array(us)
    dts = np.array(dts)

def traj_opt(x0, x1, solver):
    # use trajectory optimization method to compute trajectory between x0 and x1
    # load the dynamics function corresponding to the envname
    xs, us, ts = solver.solve(x0, x1)
    return xs, us, ts

def pathSteerTo(x0, x1, dynamics, jac_A, jac_B, traj_opt, direction, step_sz=0.02):
    # direciton 0 means forward from x0 to x1
    # direciton 1 means backward from x0 to x1
    # jac_A: given x, u -> linearization A
    # jac_B: given x, u -> linearization B
    # traj_opt: a function given two endpoints x0, x1, compute the optimal trajectory
    if direction == 0:
        xs, us, dts = traj_opt(x0.x, x1.x)
        xs, us, dts = propagate(x0.x, us, dts, dynamics=dynamics, step_sz=step_sz)
        edge_dt = np.sum(dts)
        start = x0
        goal = Node(xs[-1])
        x1 = goal
    else:
        xs, us, dts = traj_opt(x1.x, x0.x)
        us.reverse()
        dts.reverse()
        # reversely propagate the system
        xs, us, dts = propagate(x0.x, us, dts, dynamics=lambda x, u: -dynamics(x, u), interpolation=interpolation)
        xs.reverse()
        us.reverse()
        dts.reverse()
        edge_dt = np.sum(dts)
        start = Node(xs[-1])
        goal = x0
        x1 = start

    controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B)

    # notice that controller time starts from 0, hence locally need to shift the time by minusing t0_edges
    # start from 0
    time_knot = np.cumsum(dts)
    time_knot = np.insert(time_knot, 0, 0.)

    # can also change the resolution by the following function (for instance, every 10)
    #indices = np.arange(0, len(time_knot), 10)
    #time_knot = time_knot[indices]
    #print(time_knot)

    edge = Edge(xtraj, utraj, time_knot, edge_dt, S, controller)
    edge.next = goal
    start.edge = edge
    start.next = goal
    goal.prev = start
    # if the upper is defined, then we can backpropagate the funnel computation
    if goal.S0 is None:
        return x1, edge
    res_x = x1
    res_edge = edge
    # otherwise, recursively backpropagate the funnel computation
    while start is not None:
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        S = start.edge.S
        start.S0 = S
        # reversely construct the funnel
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            u1 = utraj(t1)
            A0 = jac_A(x0, u0)
            B0 = jac_B(x0, u0)
            A1 = jac_A(x1, u1)
            B1 = jac_B(x1, u1)
            A0 = np.asarray(A0)
            B0 = np.asarray(B0)
            A1 = np.asarray(A1)
            B1 = np.asarray(B1)
            S0 = S(t0).reshape(len(x0),len(x0))
            S1 = S(t1).reshape(len(x0),len(x0))
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50)
            upper_rho = rho0
            upper_x = x0
            upper_S = S0
            #tvlqr_rhos.append([rho0, rho1, t0, t1])
            if i == len(time_knot)-1:
                # the endpoint
                start.edge.rho1 = rho1
                goal.rho1 = rho1
                goal.S1 = S1
        start.edge.rho0 = rho0
        start.rho0 = rho0
        start = start.prev
        goal = goal.prev
    return res_x1, res_edge

def funnelSteerTo(x0, x1, dynamics, jac_A, jac_B, traj_opt, direciton, step_sz=0.02):
    start = x0
    goal = x1
    # recursively backpropagate the funnel computation
    while start is not None:
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        S = start.edge.S
        start.S0 = S
        # reversely construct the funnel
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            u1 = utraj(t1)
            A0 = jac_A(x0, u0)
            B0 = jac_B(x0, u0)
            A1 = jac_A(x1, u1)
            B1 = jac_B(x1, u1)
            A0 = np.asarray(A0)
            B0 = np.asarray(B0)
            A1 = np.asarray(A1)
            B1 = np.asarray(B1)
            S0 = S(t0).reshape(len(x0),len(x0))
            S1 = S(t1).reshape(len(x0),len(x0))
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50)
            upper_rho = rho0
            upper_x = x0
            upper_S = S0
            #tvlqr_rhos.append([rho0, rho1, t0, t1])
            if i == len(time_knot)-1:
                # the endpoint
                start.edge.rho1 = rho1
                goal.rho1 = rho1
                goal.S1 = S1
        start.edge.rho0 = rho0
        start.rho0 = rho0
        start = start.prev
        goal = goal.prev


def nearby(x0, x1):
    # using the S and rho stored by the node to determine distance
    # if x0 lies in x1, and within the boundary of x1 (S, rho0)
    S = x1.S0
    if x0.x.T@S@x0.x <= x1.rho0:
        return True
    else:
        return False
