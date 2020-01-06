class Node:
    def __init__(self, x):
        self.x = x
        self.next = None
        self.prev = None
        self.edge = None
        self.t = None
        self.rho0 = None
        self.rho1 = None
        self.S0 = None
        self.S1 = None
class Edge:
    ## TODO: function implmentation
    def __init__(self, xtraj, utraj, time_knot, dt, S, controller, rho0, rho1):
        # the time within Edge (for controller, etc.) starts from 0
        # so given the real time t, to obtain the control, need to t-startpoint.t
        self.xtraj = xtraj
        self.utraj = utraj
        self.time_knot = time_knot
        self.dt = dt
        self.S = S
        self.controller = controller
        self.rho0 = rho0
        self.rho1 = rho1
        self.next = None
    def __init__(self, xtraj, utraj, time_knot, dt, S, controller):
        self.dt = dt
        self.xtraj = xtraj
        self.utraj = utraj
        self.time_knot = time_knot
        self.dt = dt
        self.S = S
        self.controller = controller
        self.rho0 = None
        self.rho1 = None
        self.next = None