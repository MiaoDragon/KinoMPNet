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
import jax
import pickle
import numpy as np
import matplotlib.pyplot as plt
def tvlqr(x, u, dt, func, jax_f):
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
    # local linearization
    def jaxfunc(x, u):
        return jax.numpy.asarray(jax_f(x, u))
    # then to compute the jacobian at x, just call jax.jacfwd(jaxfunc, argnum=0)(x, u)
    # for jacobian at u, call the same function with argnum=1
    # write down the differential Ricardii equation
    def ricartti_f(t, S_):
        # obtain A and B
        A = jax.jacfwd(jaxfunc, argnums=0)(xtraj(t), utraj(t))
        B = jax.jacfwd(jaxfunc, argnums=1)(xtraj(t), utraj(t))
        #I = np.identity(len(x[0]))
        Q = 1*np.identity(len(x[0]))
        S_ = S_.reshape(Q.shape)
        res = -(Q - S_ @ B @ B.T @ S_ + S_ @ A + A.T @ S_)
        res = res.flatten()
        return res
    S_0 = 1*np.identity(len(x[0])).flatten()
    t_0 = t[-1]
    """
    # Here is one way to do it
    r = ode(ricardii_f).set_integrator('vode', method='adams', with_jacobian=False)
    r.set_initial_value(S_0, t_0)
    for i in range(len(dt)-1, -1, -1):
        print('going to integrate to time:')
        print(r.t-dt[i])
        if r.t-dt[i] < 0:
            integrate_t = 0.
        else:
            integrate_t = r.t-dt[i]
        r.integrate(integrate_t)
        print('after integration:')
        print("time:")
        print(r.t)
        print('y:')
        print(r.y)
    """
    """
    # another way
    time_span = []
    for i in range(len(t)):
        time_span.append(t[len(t)-1-i])
    sol = odeint(ricardii_f, S_0, time_span)
    sol = [s.reshape((len(x[0]), len(x[0]))) for s in sol]
    S = interp1d(time_span, sol, kind='cubic', axis=0)
    """
    # use solve_ivp
    sol = solve_ivp(fun=ricartti_f, t_span=[t[-1],0.], y0=S_0, dense_output=True)
    S = sol.sol
    print(S(t[-1]))
    def controller(t, x):
        #print('tracking time: %f' % (t))
        #print('current state:')
        #print(x)
        #print('tracking state:')
        #print(xtraj(t))
        #print('tracking action:')
        #print(utraj(t))
        B = jax.jacfwd(jaxfunc, argnums=1)(xtraj(t), utraj(t))
        K = B.T @ S(t).reshape((len(x),len(x)))
        #print(S(t).reshape(len(x),len(x)))
        u = -K @ (x - xtraj(t)) + utraj(t)
        #print('result control:')
        #print(u)
        return u
    return controller, xtraj, utraj, S

def dynamics(x, u):
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    MIN_W, MAX_W = -7., 7

    MIN_TORQUE, MAX_TORQUE = -1., 1.

    LENGTH = 1.
    MASS = 1.
    DAMPING = .05
    gravity_coeff = MASS*9.81*LENGTH*0.5
    integration_coeff = 3. / (MASS*LENGTH*LENGTH)
    res = np.zeros(2)
    res[0] = x[1]
    res[1] = integration_coeff * (u[0] - gravity_coeff*np.cos(x[0]) - DAMPING*x[1])
    #if res[0] < -np.pi:
    #    res[0] += 2*np.pi
    #elif res[0] > np.pi:
    #    res[0] -= 2 * np.pi
    #res = np.clip(res, [MIN_ANGLE, MIN_W], [MAX_ANGLE, MAX_W])
    return res

def jax_dynamics(x, u):
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    MIN_W, MAX_W = -7., 7

    MIN_TORQUE, MAX_TORQUE = -1., 1.

    LENGTH = 1.
    MASS = 1.
    DAMPING = .05
    gravity_coeff = MASS*9.81*LENGTH*0.5
    integration_coeff = 3. / (MASS*LENGTH*LENGTH)
    #res = jax.numpy.zeros(2)
    #res[0] = x[1]
    #res[1] = integration_coeff * (u[0] - gravity_coeff*jax.numpy.cos(x[0]) - DAMPING*x[1])
    return jax.numpy.asarray([x[1],integration_coeff * (u[0] - gravity_coeff*jax.numpy.cos(x[0]) - DAMPING*x[1])])

if __name__ == "__main__":

    # read the data obtained from sparse_rrt
    f = open('../../data/pendulum/path_1800.pkl', 'rb')
    p = pickle._Unpickler(f)
    p.encoding = 'latin1'
    x = p.load()
    f = open('../../data/pendulum/control_1800.pkl', 'rb')
    p = pickle._Unpickler(f)
    p.encoding = 'latin1'
    u = p.load()
    u = u.reshape(len(u),1)
    f = open('../../data/pendulum/cost_1800.pkl', 'rb')
    p = pickle._Unpickler(f)
    p.encoding = 'latin1'
    dt = p.load()
    #print(x)
    #print(u)
    print(dt)
    #print(len(x))
    #print(len(u))
    #print(len(dt))

    new_x = []
    new_u = []
    new_dt = []
    new_x0 = x[0]
    new_x.append(new_x0)
    # what if we only look at the first segment

    """
    for i in range(int(dt[0]/0.02)):
        new_x0 = new_x0 + 0.02 * dynamics(new_x0, u[0])
        new_x.append(new_x0)
        new_u.append(u[0])
        new_dt.append(0.02)
    for i in range(int(dt[1]/0.02)):
        new_x0 = new_x0 + 0.02 * dynamics(new_x0, u[1])
        new_x.append(new_x0)
        new_u.append(u[1])
        new_dt.append(0.02)
    x = new_x
    u = new_u
    dt = new_dt
    """
    x0 = x[0]
    xT = x[-1]
    print('previous xT:')
    print(xT)

    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    MIN_W, MAX_W = -7., 7

    MIN_TORQUE, MAX_TORQUE = -1., 1.

    LENGTH = 1.
    MASS = 1.
    DAMPING = .05
    gravity_coeff = MASS*9.81*LENGTH*0.5
    integration_coeff = 3. / (MASS*LENGTH*LENGTH)

    for i in range(len(dt)):
        for j in range(int(dt[i]/0.002)):
            new_x0 = new_x0 + 0.002*dynamics(new_x0, u[i])
            if new_x0[0] < -np.pi:
                new_x0[0] += 2*np.pi
            if new_x0[0] > np.pi:
                new_x0[0] -= 2*np.pi
            new_x0 = np.clip(new_x0, [MIN_ANGLE, MIN_W], [MAX_ANGLE, MAX_W])
            new_x.append(new_x0)
            new_u.append(u[i])
            new_dt.append(0.002)
        #new_x.append(new_x0)
        #new_u.append(u[i])
        #new_dt.append(dt[i])
    x = new_x
    u = new_u
    dt = new_dt

    controller, xtraj, utraj, S = tvlqr(x, u, dt, dynamics, jax_dynamics)

    # visualize the tvlqr
    x0 = x[0]
    xT = x[-1]
    print('current xT:')
    print(xT)
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_autoscale_on(True)
    hl, = ax.plot([], [], 'b')
    hl_real, = ax.plot([], [], 'r')
    hl_bvp, = ax.plot([], [], 'g')
    def update_line(h, ax, new_data):
        h.set_xdata(np.append(h.get_xdata(), new_data[0]))
        h.set_ydata(np.append(h.get_ydata(), new_data[1]))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    def update_line_batch(h, ax, new_datas):
        for i in range(len(new_datas)):
            h.set_xdata(np.append(h.get_xdata(), new_datas[i][0]))
            h.set_ydata(np.append(h.get_ydata(), new_datas[i][1]))
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()


    # plot the tracked trajectory
    batch_sz = 50
    for i in range(0, len(x), batch_sz):
        #update_line(hl_real, ax, x[i])
        update_line_batch(hl_real, ax, x[i:i+batch_sz])
    # plot the traking trajectory
    num = 200
    time_span = np.linspace(0, np.sum(dt), num+1)
    dt = time_span[-1] / num
    x = np.array(x0)
    xs = []
    xs.append(x)
    real_xs = []
    for i in range(len(time_span)):
        u = controller(time_span[i], x)
        #print('current state:')
        #print(x)
        #print('tracked traj:')
        #print(x_traj(time_span[i]))
        xdot = dynamics(x, u)
        x = x + xdot * dt
        xs.append(x)
        real_xs.append(x_traj(time_span[i]))
        if x[0] < -np.pi:
            x[0] = x[0] + 2*np.pi
        if x[0] > np.pi:
            x[0] = x[0] - 2*np.pi
        x = np.clip(x, [MIN_ANGLE, MIN_W], [MAX_ANGLE, MAX_W])
    for i in range(0, len(xs), batch_sz):
        update_line_batch(hl, ax, xs[i:i+batch_sz])
        update_line_batch(hl_bvp, ax, real_xs[i:i+batch_sz])
    plt.waitforbuttonpress()
