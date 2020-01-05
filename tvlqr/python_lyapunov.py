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
def sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50):
    # sample points at t0 and t1, make sure that d(x^TSx)/dt <= rho_dot
    # here we parameterize rho in the time interval of [t0, t1] by a line [rho0, rho1]

    # we want to maximize rho0 while making sure the following constraint holds:
    # if xTSx<=rho1 => (x+x1-upper_x)T(upper_S)(x+x1-upper_x)<=upper_rho
    # here upper_x, upper_S, and upper_rho refer to the parameters of the next funnel/LQR cone
    # they may be different in the case of LQR cone (the end point is not exactly the goal)

    # given the dynamics function, use sampling method to batch optimize rho
    # randomly sample points u such that norm(u) = 1
    U = np.random.normal(loc=0.0, scale=1.0, size=(numSample,len(S0)))
    # individually normalize each sample
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    tmp = np.linalg.pinv(S0)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U0 = U@scipy.linalg.sqrtm(tmp)
    tmp = np.linalg.pinv(S1)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U1 = U@scipy.linalg.sqrtm(tmp)
    Sdot0 = -(Q-S0@B0@np.linalg.pinv(R)@B0.T@S0+S0@A0+A0.T@S0)
    Sdot1 = -(Q-S1@B1@np.linalg.pinv(R)@B1.T@S1+S1@A1+A1.T@S1)
    K0 = np.linalg.pinv(R)@B0.T@S0
    K1 = np.linalg.pinv(R)@B1.T@S1
    prev_rho0 = 1e-6
    rho0 = 1e-6
    rho_alpha = 0.2
    prev_rho1 = 1e-6
    rho_step = upper_rho / 20.
    rho0 = upper_rho
    initial = True
    goingdown = False
    while True:
        rho0_violate = False
        rho1 = 1e-6
        rho1_grid = np.linspace(rho1, upper_rho, num=101)
        # find a rho1 in (0, upper_rho] that can make sure the constraints are valid
        # varify the constraints are true
        valid_rho1_found = False  # if one valid rho1 is found
        valid_rho1 = 1e-6
        for k in range(len(rho1_grid)):
            rho1 = rho1_grid[k]
            X0 = np.sqrt(rho0)*U0
            X1 = np.sqrt(rho1)*U1
            rhodot0 = (rho1-rho0)/(t1-t0)  # should be bounded by this
            rhodot1 = (rho1-rho0)/(t1-t0)  # should be bounded by this

            violate = False
            # varify constraint at t0 first
            for i in range(numSample):
                cons0 = X0[i].T@Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))-rhodot0 # should <= 0
                cons1 = X1[i].T@Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))-rhodot1
                cons_upper = (X1[i]+x1-upper_x)@upper_S@(X1[i]+x1-upper_x)-upper_rho  # should <= 0
                if cons0 > 0 or cons1 > 0 or cons_upper > 0:
                    # this rho0 is not working
                    violate = True
                    break
            if violate:
                # continue searching for the next rho1
                continue
            # all constraints are satisfied
            valid_rho1_found = True
            valid_rho1 = rho1_grid[k]
            break
        if valid_rho1_found:
            # advance to the next rho0
            prev_rho0 = rho0
            prev_rho1 = valid_rho1
            #print('valid rho1 found. Advance to the next rho0...')
            #print("rho0 = %f, rho1 = %f" % (rho0, rho1))
            if initial:
                rho_step = rho_step / 5 # use small step for stepping up
                rho0 = rho0 + rho_step
                goingdown = False
                initial = False
                continue
            if not goingdown:
                # going up, find the first unsuccessful one
                rho0 = rho0 + rho_step
                continue
        else:
            if initial:
                # the first guess is not working, going down to find the first successful one
                goingdown = True
                initial = False
                rho0 = rho0 - rho_step
                continue
            if goingdown:
                # still trying to find the first successul one
                while rho0 <= rho_step:
                    rho_step = rho_step / 2  # smaller rho_step
                rho0 = rho0 - rho_step
                continue
        # if none of the above conditions, then it is time to return the value
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
    prev_rho = 1e-6
    rho = 1e-6
    rho_alpha = 0.1
    while True:
        X = np.sqrt(rho)*U
        cons_vec = []
        violate = False
        tmp = X@S  # num_sample x k
        for i in range(len(U)):
            #cons_vec.append(tmp[i]@func(xG+X[i],uG-K@X[i]))
            if tmp[i]@func(xG+X[i],uG-K@X[i]) >= 0.:
                # constraint is violated
                violate = True
                break
        if violate:
            return prev_rho
        prev_rho = rho
        rho = rho * (1+rho_alpha)
