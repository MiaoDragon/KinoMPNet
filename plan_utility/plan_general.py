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
    # after trajopt, make actions of dimension 2
    us = us.reshape(len(us), -1)

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
    return x1, edge

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
    # build funnel for one step
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
    upper_x = goal.x
    upper_S = goal.S0
    upper_rho = goal.rho0  # the rho0 of goal will be come the upper_rho currently
    time_knot = start.edge.time_knot
    xtraj = start.edge.xtraj
    utraj = start.edge.utraj

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
        rho0, rho1 = sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func=dynamics, numSample=1000)
        upper_rho = rho0
        upper_x = x0
        upper_S = S0
        if i == len(time_knot)-1:
            # the endpoint
            start.edge.rho1 = rho1
            goal.rho1 = rho1
            goal.S1 = S1
    start.edge.rho0 = rho0
    start.rho0 = rho0



def lazyFunnel(xg, xG, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, step_sz=0.02):
    # compute funnel backward until xg
    # recursively backpropagate the funnel computation
    start = xG.prev
    goal = xG
    while start is not None:
        # already at xg
        if xg.prev is not None and np.linalg.norm(xg.prev.x - start.x) <= 1e-6:
            # xg already computed
            break
        funnelSteerTo(start, goal, dynamics, enforce_bounds, jac_A, jac_B, traj_opt, direction=0, step_sz=step_sz)
        start = start.prev
        goal = goal.prev



def node_nearby(x0, x1, S, rho, system):
    # state x0 to state x1
    delta_x = x0 - x1
    circular = system.is_circular_topology()
    for i in range(len(delta_x)):
        if circular[i]:
            # if it is angle
            if delta_x[i] > np.pi:
                delta_x[i] = delta_x[i] - 2*np.pi
            if delta_x[i] < -np.pi:
                delta_x[i] = delta_x[i] + 2*np.pi
    #lam_S = np.linalg.eigvals(S).max()
    xTSx = delta_x.T@S@delta_x# / lam_S
    if xTSx <= 4.:
        print('nearby:')
        print('S:')
        print(S)
        print('xTSx: %f' % (xTSx))
        print('rho^2: %f' % (rho*rho))
    if xTSx <= rho*rho:
         return True
    return False

def line_nearby(x0, x1, system):
    # state x0 to line starting from node x1
    e = x1.edge
    xs = e.xs
    us = e.us
    ts = e.time_knot
    for k in range(len(ts)-1):
        S = e.S(ts[k]).reshape((len(x0), len(x0)))
        rho = e.rho0s[k]
        if node_nearby(x0, xs[k], S, rho, system):
            return True, k
    return False, 0

def nearby(x0, x1, system):
    # using the S and rho stored by the node to determine distance
    # if x0 lies in x1, and within the boundary of x1 (S, rho0)
    # notice that for circulating state, needs to map the angle
    # if edge is defined on xG, then use edge
    if x0.edge is not None:
        e = x0.edge
        xs = e.xs
        us = e.us
        ts = e.time_knot
        for k in range(len(ts)-1):
            if x1.edge is not None:
                line_near, k1 = line_nearby(xs[k], x1, system)
                if line_near:
                    # near the line, with node index k1
                    return True, k, k1
            else:
                if node_nearby(xs[k], x1.x, x1.S0, x1.rho0, system):
                    return True, k, 0
    else:
        if x1.edge is not None:
            line_near, k1 = line_nearby(x0.x, x1, system)
            if line_near:
                return True, 0, k1
        else:
            if node_nearby(x0.x, x1.x, x1.S0, x1.rho0, system):
                return True, 0, 0
    return False, 0, 0

def node_h_nearby(x0, x1, S, rho, system):
    # given two nodes, S and rho, check if x0 is near x1
    delta_x = x0 - x1
    # this is pendulum specific. For other envs, need to do similar things
    circular = system.is_circular_topology()
    for i in range(len(delta_x)):
        if circular[i]:
            # if it is angle
            if delta_x[i] > np.pi:
                delta_x[i] = delta_x[i] - 2*np.pi
            if delta_x[i] < -np.pi:
                delta_x[i] = delta_x[i] + 2*np.pi
    #lam_S = np.linalg.eigvals(S).max()
    xTSx = delta_x.T@S@delta_x# / lam_S
    if xTSx <= 1.:
        print('delta_x:')
        print(delta_x)
        print('S:')
        print(S)
        print('xTSx: %f' % (xTSx))
        # notice that we define rho to be ||S^{1/2}x||
        print('rho^2: %f' % (rho*rho))
    return xTSx / (rho*rho)
def line_h_dist(x0, xG, S, rho, system):
    # check state x0 against line starting from node xG
    e2 = xG.edge
    xs2 = e2.xs
    us2 = e2.us
    ts2 = e2.time_knot
    res = 1e8
    for k2 in range(len(ts2)-1):
        if e2.S is not None:
            S = e2.S(ts2[k2]).reshape((len(x0), len(x0)))
            rho = e2.rho0s[k2]
        elif xG.S0 is not None:
            S = xG.S0
            rho = xG.rho0
        dist = node_h_nearby(x0, xs2[k2], S, rho, system)
        if dist < res:
            res = dist
    return res
def h_dist(node, xG, S, rho, system):
    # check if the edge starting from node and the edge starting from xG has intersection
    res = 1e8
    if node.edge is not None:
        e1 = node.edge
        xs1 = e1.xs
        us1 = e1.us
        ts1 = e1.time_knot
        for k1 in range(len(ts1)-1):
            if xG.edge is not None:
                dist = line_h_dist(xs1[k1], xG, S, rho, system)
            else:
                if xG.S0 is not None:
                    S = xG.S0
                    rho = xG.rho0
                dist = node_h_dist(xs1[k1], xG.x, S, rho, system)
            if dist < res:
                res = dist
    else:
        if xG.edge is not None:
            dist = line_h_dist(node.x, xG, S, rho, system)
        else:
            if xG.S0 is not None:
                S = xG.S0
                rho = xG.rho0
            dist = node_h_dist(node.x, xG.x, S, rho, system)
        if dist < res:
            res = dist
    return res
