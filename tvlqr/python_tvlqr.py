"""
use numerical integration and interpolation method provided by scipy to solve
differential Ricardii equation for TVLQR
input: traj opt output:
    x[N], u[N], dt[N]
    func: the system dynamics, xdot
steps:
1. interpolation using CubicHermiteSpline method provided by scipy, to preserve
values and first-order derivative of knots
2. locally linearization to obtain matrices A and B
3. write down the differential Ricardii equation, and use ODE solver solve_ivp
to solve it backward. Then obtained the S
4. obtain the matrix K, and thus the feedback controller is
u(t) = u0(t) - K(t)(x(t)-x0(t))
"""
import scipy
from scipy.interpolate import CubicHermiteSpline, PPoly, interp1d
from scipy.integrate import solve_ivp, ode, odeint
import numpy as np
def tvlqr(x, u, dt, func, jac_A, jac_B, system=None, Qf=None):
    # len(x) = len(u)+1=len(dt)+1
    # interpolation of x
    t = [0.]
    # obtain the time step for each knot
    for i in range(len(dt)):
        t.append(t[-1] + dt[i])
    xdot = []
    for i in range(len(x)-1):
        xdot.append(func(x[i], u[i]))
    xdot.append(func(x[-1], u[-1]))
    # obtain the interpolation for x
    xtraj = CubicHermiteSpline(x=t, y=x, dydx=xdot, extrapolate=True)


    # interp1d zero-th order interpolation will throw away the last signal
    utraj = []
    for i in range(len(u)):
        utraj.append(u[i])
    utraj.append(np.zeros(u[-1].shape))
    utraj = interp1d(x=t, y=utraj, kind='zero', axis=0, fill_value='extrapolate')
    # then to compute the jacobian at x, just call jax.jacfwd(jaxfunc, argnum=0)(x, u)
    # for jacobian at u, call the same function with argnum=1
    # write down the differential Ricardii equation
    def ricartti_f(t, S_):
        # obtain A and B
        A = jac_A(xtraj(t), utraj(t))
        B = jac_B(xtraj(t), utraj(t))
        A = np.asarray(A)
        B = np.asarray(B)
        #I = np.identity(len(x[0]))
        Q = 1*np.identity(len(x[0]))
        #Q = np.diag([100.,100.,1.,1.])
        S_ = S_.reshape(Q.shape)
        res = -(Q - S_ @ B @ B.T @ S_ + S_ @ A + A.T @ S_)
        res = res.flatten()
        return res
    if Qf is None:
        S_0 = 1*np.identity(len(x[0])).flatten()
    else:
        # maybe we are using Qf to connect to the next funnel
        S_0 = Qf.flatten()
    t_0 = t[-1]
    # use solve_ivp
    sol = solve_ivp(fun=ricartti_f, t_span=[t[-1],0.], y0=S_0, dense_output=True)
    S = sol.sol
    def controller(t, x):
        B = jac_B(xtraj(t), utraj(t))
        B = np.asarray(B)
        K = B.T @ S(t).reshape((len(x),len(x)))
        delta_x = x - xtraj(t)
        if system is not None:
            circular = system.is_circular_topology()
            for i in range(len(delta_x)):
                if circular[i]:
                    # if it is angle
                    if delta_x[i] > np.pi:
                        delta_x[i] = delta_x[i] - 2*np.pi
                    if delta_x[i] < -np.pi:
                        delta_x[i] = delta_x[i] + 2*np.pi
        u = -K @ delta_x + utraj(t)
        return u
    return controller, xtraj, utraj, S
