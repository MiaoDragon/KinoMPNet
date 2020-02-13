import numpy as np
import scipy
import matplotlib.pyplot as plt
def plot_ellipsoid(ax, S, rho, x0, alpha=1.0):
    theta = np.linspace(0, np.pi*2, 100)
    U = [np.cos(theta), np.sin(theta), np.zeros(100), np.zeros(100)]
    U = np.array(U).T
    tmp = np.linalg.pinv(S)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    S_invsqrt = scipy.linalg.sqrtm(tmp)
    X = U @ S_invsqrt  # 100x2
    X = np.sqrt(rho)*X + x0
    ax.plot(X[:,0],X[:,1], alpha=alpha)

def animation_acrobot(fig, ax, animator, xs, obs):
    animator.obs = obs
    animator._init(ax)
    for i in range(len(xs)):
        animator._animate(xs[i], ax)
        animator.draw_update_line(fig, ax)
    
def plot_trajectory(ax, start, goal, dynamics, enforce_bounds, IsInCollision, step_sz):

    plot_ellipsoid(ax, goal.S0, goal.rho0, goal.x, alpha=0.1)

    # plot funnel
    # rho_t = rho0+(rho1-rho0)/(t1-t0)*t
    node = start
    while node.edge is not None:
        if node.edge.S is not None:
            rho0s = node.edge.rho0s[node.edge.i0:]
            rho1s = node.edge.rho1s[node.edge.i0:]
            time_knot = node.edge.time_knot[node.edge.i0:]
            S = node.edge.S
            for i in range(len(rho0s)):
                rho0 = rho0s[i]
                rho1 = rho1s[i]
                t0 = time_knot[i]
                t1 = time_knot[i+1]
                rho_t = rho0
                S_t = S(t0).reshape(len(node.x),len(node.x))
                x_t = node.edge.xtraj(t0)
                u_t = node.edge.utraj(t0)
                # plot
                plot_ellipsoid(ax, S_t, rho_t, x_t, alpha=0.1)
                rho_t = rho1
                S_t = S(t1).reshape(len(node.x),len(node.x))
                x_t = node.edge.xtraj(t1)
                u_t = node.edge.utraj(t1)
                # plot
                plot_ellipsoid(ax, S_t, rho_t, x_t, alpha=0.1)
        node = node.next
    node = start
    actual_x = node.x
    xs = []
    us = []
    valid = True
    while node.edge is not None:
        # printout which node it is
        print('steering node...')
        print('node.x:')
        print(node.x)
        print('node.next.x:')
        print(node.next.x)
        # if node does not have controller defined, we use open-loop traj
        if node.edge.S is None:
            xs += node.edge.xs.tolist()
            actual_x = np.array(xs[-1])
        else:
            # then we use the controller
            # see if it can go to the goal region starting from start
            dt = node.edge.dts[node.edge.i0:]
            num = np.sum(dt)/step_sz
            time_span = np.linspace(node.edge.t0, node.edge.t0+np.sum(dt), num+1)
            delta_t = step_sz
            xs.append(actual_x)
            controller = node.edge.controller
            print('number of time knots: %d' % (len(time_span)))
            # plot data
            for i in range(len(time_span)):
                u = controller(time_span[i], actual_x)
                xdot = dynamics(actual_x, u)
                actual_x = actual_x + xdot * delta_t
                xs.append(actual_x)
                actual_x = enforce_bounds(actual_x)
                print('actual x:')
                print(actual_x)
                if IsInCollision(actual_x):
                    print('In Collision Booooo!!')
                    valid = False
        node = node.next
    xs = np.array(xs)
    ax.plot(xs[:,0], xs[:,1], 'black', label='using controller')
    plt.show()
    print('start:')
    print(start.x)
    print('goal:')
    print(goal.x)
    if not valid:
        print('in Collision Boommm!!!')

    plt.waitforbuttonpress()
    return xs
