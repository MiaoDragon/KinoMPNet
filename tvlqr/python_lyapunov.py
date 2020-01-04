"""
this implements the lyapunov analysis for TVLQR using the following two methods:
1. linearization
2. sampling-based method
"""
from cvxopt import matrix, solvers
from scipy.optimize import NonlinearConstraint
import numpy as np
from scipy.optimize import minimize, Bounds, linprog
import jax
import control
import scipy
def linear_tv_verify(t0, t1, rho_upper, S0, S1, A0, A1, B0, B1, R, Q):
    print("RVLQR verification using linear method")
    # calculate the rho in the time range [t0, t1]
    # by iteratively optimizing rho, and solving a SDP feasibility problem
    # the outter loop will use nonlinear solver of scipy, by setting constraint
    # toward the minimum eigenvalue to ensure SDP
    # here we use linear approximation for the level set rho
    # specifically, rho(t) = a1 * t + a0

    # initialize a1 and a0 using rho_upper
    a1 = 0.
    a0 = rho_upper

    # SDP equation:
    # epsilon*S+lambda*S-lambda*rho(t)I<=SBR^{-1}B^TS+Q+rho_dot*I
    # -epsilon*I<=0
    tmp0 = S0@B0@(np.linalg.pinv(R))@B0.T@S0+Q
    tmp1 = S1@B1@(np.linalg.pinv(R))@B1.T@S1+Q
    idty = np.identity(len(S0))
    while True:
        # solving SDP to obtain lambda and epsilon
        c = matrix([1., 1.])
        G = [ matrix( [S0-(a1*t0+a0)*idty, S0] ) ]
        h = [ matrix(tmp0+Q+a1*idty) ]
        G += [ matrix( [S1-(a1*t1+a0)*idty, S1] ) ]
        h += [ matrix(tmp1+Q+a1*idty) ]
        # adding constraint: epsilon >= 0
        G += [ matrix( [[0],[-1]] ) ]
        h += [ matrix( [[0]] ) ]
        sol = solvers.sdp(c, Gs=G, hs=h)
        print('after solving SDP:')
        print(sol['x'])
        x = sol['x']
        lam = x[0]
        eps = x[1]

        # seperate rho and other part
        const0 = tmp0-2*eps*S0-lam*S0
        const1 = tmp1-2*eps*S1-lam*S1
        def rosen(x):
            # we want to maximize the rho at time t0
            return -(x[1]*t0+x[0])
        def rosen_der(x):
            der = np.zeros_like(x)
            der[0] = -1.0
            der[1] = -t0
            return der
        def rosen_hess(x):
            return np.zeros((2,2))
        def cons_f(x):
            # x: [a0, a1]
            # the eigenvalue function
            evec0, eval0 = np.linalg.eig( const0+x[1]*idty+lam*(x[1]*t0+x[0])*idty )
            evec1, eval1 = np.linalg.eig( const1+x[1]*idty+lam*(x[1]*t1+x[0])*idty )
            return [eval0.min(), eval1.min(), x[1]*t0+x[0], x[1]*t1+x[0]]
        def cons_J(x):
            # return 4*2 matrix
            J = np.zeros((4,2))
            evec0, eval0 = np.linalg.eig( const0+x[1]*idty+lam*(x[1]*t0+x[0])*idty )
            evec1, eval1 = np.linalg.eig( const1+x[1]*idty+lam*(x[1]*t1+x[0])*idty )
            # sort
            idx = eval0.argsort()[::-1]
            eval0 = eval0[idx]
            evec0 = evec0[:,idx]
            idx = eval1.argsort()[::-1]
            eval1 = eval1[idx]
            evec1 = evec1[:,idx]
            # use eigenvalue derivative: d lambda/d A = vv^T
            dv0dx1 = lam*t0+1
            dv0dx0 = lam
            dv1dx1 = lam*t1+1
            dv1dx0 = lam
            # the other two derivative
            drho0dx0 = 1
            drho0dx1 = t0
            drho1dx0 = 1
            drho1dx1 = t1
            J = np.array([[dv0dx0, dv0dx1],
                          [dv1dx0, dv1dx1],
                          [drho0dx0, drho0dx1,
                          [drho1dx0, drho1dx1]]])
            return J
        def cons_H(x, v):
            return np.zeros((2,2))
        lower = np.array([0., 0, 0, 0])
        upper = np.array([np.inf, np.inf, np.inf, rho_upper])
        nonlinear_constraint = NonlinearConstraint(cons_f, lower, upper, jac=cons_J, hess=cons_H)
        x0 = np.array([a0, a1])
        res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
                constraints=[nonlinear_constraint],
                options={'verbose': 1}, bounds=bounds)
        x = res.x
        a0 = x[0]
        a1 = x[1]
        print('a0, a1: ')
        print(x)

def linear_ti_verify(S, A, B, R, Q):
    print("LQR verification using linear method")
    # calculate the rho
    # by iteratively optimizing rho, and solving a SDP feasibility problem
    # the outter loop will use nonlinear solver of scipy, by setting constraint
    # toward the minimum eigenvalue to ensure SDP
    # initialize rho
    rho = 0.1
    # SDP equation:
    # epsilon*I+lambda*(rho*I-S) <= 2*SBR^{-1}B^TS-SA-A^TS
    # -epsilon*I<=0
    tmp = 2*S@B@np.linalg.pinv(R)@B.T@S-S@A-A.T@S
    idty = np.identity(len(S))
    while True:
        # solving SDP to obtain lambda and epsilon
        c = matrix([1., 1.])
        G = [ matrix( [(rho*idty-S).flatten().tolist(), idty.flatten().tolist()] ) ]
        h = [ matrix(tmp) ]
        # adding constraint: epsilon >= 0, lambda >= 0
        G += [ matrix( [[-1.0],[0.0]] ) ]
        h += [ matrix( [[0.]] ) ]
        G += [ matrix( [[0.],[-1.]] ) ]
        h += [ matrix( [[0.]] ) ]
        sol = solvers.sdp(c, Gs=G, hs=h)
        print('after solving SDP:')
        print(sol['x'])
        x = sol['x']
        lam = x[0]
        eps = x[1]

        # seperate rho and other part
        const = tmp-eps*idty+lam*S
        def rosen(x):
            # we want to maximize the rho
            return -(x[0])
        def rosen_der(x):
            der = np.zeros_like(x)
            der[0] = -1.0
            return der
        def rosen_hess(x):
            return np.zeros((1,1))
        def cons_f(x):
            # x: [a0, a1]
            # the eigenvalue function
            evec, eval = np.linalg.eig( const-lam*x[0]*idty )
            return [eval.min()]
        def cons_J(x):
            # return 1*1 matrix
            J = np.zeros((1,1))
            # use eigenvalue derivative: d lambda/d A = vv^T
            dvdrho = -lam
            # the other two derivative
            drhodrho = 1.
            J = np.array([[dvdrho]])
            return J
        def cons_H(x, v):
            return np.zeros((1,1))
        lower = np.array([0.])
        upper = np.array([np.inf])
        nonlinear_constraint = NonlinearConstraint(cons_f, lower, upper, jac=cons_J, hess=cons_H)
        x0 = np.array([rho])
        bounds = Bounds([0.], [np.inf])
        res = minimize(rosen, x0, method='trust-constr', jac=rosen_der, hess=rosen_hess,
                constraints=[nonlinear_constraint],
                options={'verbose': 1}, bounds=bounds)
        x = res.x
        rho = x[0]
        print('rho: ')
        print(x)

def sample_tv_verify(t0, t1, rho_upper, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50):
    # sample points at t0 and t1, make sure that d(x^TSx)/dt <= rho_dot
    # here we parameterize rho in the time interval of [t0, t1] by a line [rho0, rho1]
    # we want to make sure rho1 <= rho_upper, while maximizing rho0
    # given the dynamics function, use sampling method to batch optimize rho
    # randomly sample points u such that norm(u) = 1
    U = np.random.normal(loc=0.0, scale=1.0, size=(numSample,len(S)))
    # individually normalize each sample
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    tmp = np.linalg.pinv(S0)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U0 = U@scipy.linalg.sqrtm(tmp)
    tmp = np.linalg.pinv(S1)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U1 = U@scipy.linalg.sqrtm(tmp)
    Sdot0 = -(Q-S0@B@np.linalg.pinv(R)@B.T@S0+S0@A+A.T@S0)
    Sdot1 = -(Q-S1@B@np.linalg.pinv(R)@B.T@S1+S1@A+A.T@S1)
    K0 = np.linalg.pinv(R)@B.T@S0
    K1 = np.linalg.pinv(R)@B.T@S1
    prev_rho0 = 1e-3
    rho0 = 1e-3
    rho_alpha = 0.1
    prev_rho1 = 1e-3
    while True:
        rho0_violate = False
        rho1 = 1e-3
        rho1_grid = np.linspace(rho1, rho_upper, num=101)
        # find a rho1 in (0, rho_upper] that can make sure the constraints are valid
        # varify the constraints are true
        valid_rho1_found = False  # if one valid rho1 is found
        valid_rho1 = 1e-3
        for i in range(len(rho1_grid)):
            rho1 = rho1_grid[i]
            X0 = rho0*U
            X1 = rho1*U
            rhodot = (rho1-rho0)/(t1-t0)
            violate = False
            # varify constraint at t0 first
            tmp0 = 2*X0@S0
            tmp1 = 2*X1@S1
            for i in range(len(numSample)):
                cons0 = X0[i].T@Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))-rhodot # should <= 0
                cons1 = X1[i].T@Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))-rhodot
                if cons0 > 0 or cons1 > 0:
                    # this rho0 is not working
                    violate = True
                    break
            if violate:
                # continue searching for the next rho1
                continue
            # all constraints are satisfied
            valid_rho1_found = True
            valid_rho1 = rho1_grid[i]
            break
        if valid_rho1_found:
            # advance to the next rho0
            prev_rho0 = rho0
            prev_rho1 = valid_rho1
            rho0 = rho0 * (1+rho_alpha)
        else:
            return prev_rho0, prev_rho1



def sample_ti_verify(xG, uG, S, K, func, numSample=50):
    # given the dynamics function, use sampling method to batch optimize rho
    # randomly sample points u such that norm(u) = 1
    U = np.random.normal(loc=0.0, scale=1.0, size=(numSample,len(S)))
    # individually normalize each sample
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    # randomly assign a length within 0 to 1
    for i in range(len(U)):
        alpha = np.random.uniform()
        U[i] = U[i] * alpha
    tmp = np.linalg.pinv(S)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U = U@scipy.linalg.sqrtm(tmp)
    # line search to find rho
    #rho = 0.1
    prev_rho = 1e-3
    rho = 1e-3
    rho_alpha = 0.1
    while True:
        X = rho*U
        cons_vec = []
        violate = False
        tmp = X@S  # num_sample x k
        for i in range(len(U)):
            #cons_vec.append(tmp[i]@func(xG+X[i],uG-K@X[i]))
            if tmp[i]@func(xG+X[i],uG-K@X[i]) >= 0.:
                # constraint is violated
                violate = True
                break
        print('rho: ')
        print(rho)
        if violate:
            return prev_rho
        prev_rho = rho
        rho = rho * (1+rho_alpha)

from python_tvlqr import *
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

def stable_u(x):
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    MIN_W, MAX_W = -7., 7

    MIN_TORQUE, MAX_TORQUE = -1., 1.

    LENGTH = 1.
    MASS = 1.
    DAMPING = .05
    gravity_coeff = MASS*9.81*LENGTH*0.5
    integration_coeff = 3. / (MASS*LENGTH*LENGTH)
    return np.array([gravity_coeff*np.cos(x[0])])

def jax_f(x, u):
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

def jaxfunc(x, u):
    return jax.numpy.asarray(jax_f(x, u))



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

    time_knot = np.cumsum(new_dt)
    # obtain a lqr for end position
    #A = jax.jacfwd(jaxfunc, argnums=0)(xtraj(time_knot[-1]), utraj(time_knot[-1]))
    #B = jax.jacfwd(jaxfunc, argnums=1)(xtraj(time_knot[-1]), utraj(time_knot[-1]))

    xG = np.array([0.70, 0.])
    uG = stable_u(xG)
    print(dynamics(xG, uG))
    A = jax.jacfwd(jaxfunc, argnums=0)(xG, uG)
    B = jax.jacfwd(jaxfunc, argnums=1)(xG, uG)

    A = np.asarray(A)
    B = np.asarray(B)
    Q = np.identity(2)
    R = np.identity(1)
    K, S, E = control.lqr(A, B, Q, R)

    #print(dynamics(xtraj(time_knot[-1]), utraj(time_knot[-1])))
    sample_ti_verify(xG, uG, S, K, dynamics, numSample=1000)



    sample_tv_verify(t0, t1, rho_upper, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50)
