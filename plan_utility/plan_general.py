import sys
sys.path.append('..')

import numpy as np
from tvlqr.python_tvlqr import tvlqr
from tvlqr.python_lyapunov import sample_tv_verify
from plan_utility.data_structure import *
def propagate(x, us, dts, dynamics, enforce_bounds, step_sz=None):
    # use the dynamics to interpolate the state x
    # can implement different interpolation method for this
    new_xs = [x]
    new_us = []
    new_dts = []
    for i in range(len(us)):
        dt = dts[i]
        u = us[i]
        num_steps = int(dt / step_sz)
        last_step = dt - num_steps*step_sz
        for k in range(num_steps):
            x = x + step_sz*dynamics(x, u)
            x = enforce_bounds(x)
            new_xs.append(x)
            new_us.append(u)
            new_dts.append(step_sz)
        x = x + last_step*dynamics(x, u)
        x = enforce_bounds(x)
        new_xs.append(x)
        new_us.append(u)
        new_dts.append(last_step)
    new_xs = np.array(new_xs)
    new_us = np.array(new_us)
    new_dts = np.array(new_dts)
    return new_xs, new_us, new_dts
def traj_opt(x0, x1, solver):
    # use trajectory optimization method to compute trajectory between x0 and x1
    # load the dynamics function corresponding to the envname
    xs, us, ts = solver.solve(x0, x1)
    return xs, us, ts

def pathSteerTo(x0, x1, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction, step_sz=0.002, compute_funnel=True):
    # direciton 0 means forward from x0 to x1
    # direciton 1 means backward from x0 to x1
    # jac_A: given x, u -> linearization A
    # jac_B: given x, u -> linearization B
    # traj_opt: a function given two endpoints x0, x1, compute the optimal trajectory
    if direction == 0:
        xs, us, dts = traj_opt(x0.x, x1.x)
        """
            print('----------------forward----------------')
            print('trajectory opt:')
            print('start:')
            print(x0.x)
            print('end:')
            print(x1.x)
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        # ensure us and dts have length 1 less than xs
        if len(us) == len(xs):
            us = us[:-1]
        xs, us, dts = propagate(x0.x, us, dts, dynamics=dynamics, enforce_bounds=enforce_bounds, step_sz=step_sz)
        """
            print('propagation result:')
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        edge_dt = np.sum(dts)
        start = x0
        goal = Node(xs[-1])
        x1 = goal
    else:
        xs, us, dts = traj_opt(x1.x, x0.x)
        """
            print('----------------backward----------------')
            print('trajectory opt:')
            print('start:')
            print(x1.x)
            print('end:')
            print(x0.x)
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        if len(us) == len(xs):
            us = us[:-1]
        us = np.flip(us, axis=0)
        dts = np.flip(dts, axis=0)
        # reversely propagate the system
        xs, us, dts = propagate(x0.x, us, dts, dynamics=lambda x, u: -dynamics(x, u), enforce_bounds=enforce_bounds, step_sz=step_sz)
        xs = np.flip(xs, axis=0)
        us = np.flip(us, axis=0)
        dts = np.flip(dts, axis=0)
        """
            print('propagation result:')
            print('xs[0]:')
            print(xs[0])
            print('xs[-1]:')
            print(xs[-1])
            print('us:')
            print(us)
            print('dts:')
            print(dts)
        """
        edge_dt = np.sum(dts)
        start = Node(xs[0])  # after flipping, the first in xs is the start
        goal = x0
        x1 = start

    # notice that controller time starts from 0, hence locally need to shift the time by minusing t0_edges
    # start from 0
    time_knot = np.cumsum(dts)
    time_knot = np.insert(time_knot, 0, 0.)

    # can also change the resolution by the following function (for instance, every 10)
    #indices = np.arange(0, len(time_knot), 10)
    #time_knot = time_knot[indices]
    #print(time_knot)

    edge = Edge(xs, us, dts, time_knot, edge_dt)
    edge.next = goal
    start.edge = edge
    start.next = goal
    goal.prev = start

    if goal.S0 is None or not compute_funnel:
        return x1, edge
    # the values to return: new node, new edge
    res_x = x1
    res_edge = edge    
    # if the upper is defined, then we can backpropagate the tvlqr and funnel computation    
    while start is not None:
        # assuming we haven't computed tvlqr for start->goal
        edge = start.edge
        xs = edge.xs
        us = edge.us
        dts = edge.dts
        Qf = goal.S0
        if Qf is not None:
            Qf = np.array(Qf)
        controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B, Qf=Qf)
        start.S0 = S(0).reshape((len(start.x),len(start.x)))
        edge.xtraj = xtraj
        edge.utraj = utraj
        edge.S = S
        edge.controller = controller

        # otherwise, recursively backpropagate the funnel computation
        #print("xs (which is used to construct xtraj):")
        #print(xs)
        #print('xtraj.x:')
        #print(xtraj.x)        
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        #print('time_knot:')
        #print(time_knot)
        #print('goal.x:')
        #print(goal.x)
        #print('xtraj(last_time):')
        #print(xtraj(time_knot[-1]))
        S = start.edge.S
        print('time_knot: %d' % (len(time_knot)))
        #todo: to add rho0s and rho1s list to edge
        # reversely construct the funnel
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            #if i == len(time_knot)-1:
            #    x1 = goal.x
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
            Q = np.identity(len(x0))
            R = np.identity(len(u0))
            ##TODO: check the output of sample_tv_verify
            #print('verifying...')
            #print('x0:')
            #print(x0)
            #print('x1:')
            #print(x1)
            #print('upper_x')
            #print(upper_x)
            #print("S1:")
            #print(S1)
            #print("upper_S:")
            #print(upper_S)
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=1000)
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
    return res_x, res_edge

def funnelSteerTo(x0, x1, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direciton, step_sz=0.02):
    start = x0
    goal = x1
    if direciton == 0:
        start = x0
        goal = x1
    else:
        start = x1
        goal = x0
    # recursively backpropagate the funnel computation
    while start is not None:
        # assuming we haven't computed tvlqr for start->goal
        edge = start.edge
        xs = edge.xs
        us = edge.us
        dts = edge.dts
        Qf = goal.S0
        if Qf is not None:
            Qf = np.array(Qf)
        controller, xtraj, utraj, S = tvlqr(xs, us, dts, dynamics, jac_A, jac_B, Qf=Qf)
        start.S0 = S(0).reshape((len(start.x),len(start.x)))
        edge.xtraj = xtraj
        edge.utraj = utraj
        edge.S = S
        edge.controller = controller

        # otherwise, recursively backpropagate the funnel computation
        #print("xs (which is used to construct xtraj):")
        #print(xs)
        #print('xtraj.x:')
        #print(xtraj.x)        
        upper_x = goal.x
        upper_S = goal.S0
        upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
        time_knot = start.edge.time_knot
        xtraj = start.edge.xtraj
        utraj = start.edge.utraj
        #print('time_knot:')
        #print(time_knot)
        #print('goal.x:')
        #print(goal.x)
        #print('xtraj(last_time):')
        #print(xtraj(time_knot[-1]))
        S = start.edge.S
        print('time_knot: %d' % (len(time_knot)))
        #todo: to add rho0s and rho1s list to edge
        # reversely construct the funnel
        for i in range(len(time_knot)-1, 0, -1):
            t0 = time_knot[i-1]
            t1 = time_knot[i]
            x0 = xtraj(t0)
            u0 = utraj(t0)
            x1 = xtraj(t1)
            #if i == len(time_knot)-1:
            #    x1 = goal.x
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
            Q = np.identity(len(x0))
            R = np.identity(len(u0))
            ##TODO: check the output of sample_tv_verify
            #print('verifying...')
            #print('x0:')
            #print(x0)
            #print('x1:')
            #print(x1)
            #print('upper_x')
            #print(upper_x)
            #print("S1:")
            #print(S1)
            #print("upper_S:")
            #print(upper_S)
            rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=1000)
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
    # notice that for circulating state, needs to map the angle
    S = x1.S0
    print('nearby')
    print(x0.x)
    print(S)
    print(x1.rho0)
    delta_x = x0.x - x1.x
    # this is pendulum specific. For other envs, need to do similar things
    if delta_x[0] > np.pi:
        delta_x[0] = delta_x[0] - 2*np.pi
    if delta_x[0] < -np.pi:
        delta_x[0] = delta_x[0] + 2*np.pi
    xTSx = delta_x.T@S@delta_x
    print('xTSx: %f' % (xTSx))
    # notice that we define rho to be ||S^{1/2}x||
    if xTSx <= x1.rho0*x1.rho0:
        return True
    else:
        return False