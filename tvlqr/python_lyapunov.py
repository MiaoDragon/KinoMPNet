"""
this implements the lyapunov analysis for TVLQR using the following two methods:
1. linearization
2. sampling-based method
"""
from scipy.optimize import NonlinearConstraint
import numpy as np
from scipy.optimize import minimize, Bounds, linprog
import scipy
def sample_tv_verify_sqrtrho(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50):
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
        rho1_grid = np.linspace(upper_rho, rho1, num=101)
        # find a rho1 in (0, upper_rho] that can make sure the constraints are valid
        # varify the constraints are true
        valid_rho1_found = False  # if one valid rho1 is found
        valid_rho1 = 1e-6
        print('searching... current rho0: %f' % (rho0))
        for k in range(len(rho1_grid)):
            rho1 = rho1_grid[k]
            X0 = np.sqrt(rho0)*U0
            X1 = np.sqrt(rho1)*U1
            rhodot0 = (rho1-rho0)/(t1-t0)  # should be bounded by this
            rhodot1 = (rho1-rho0)/(t1-t0)  # should be bounded by this
            print('    searching... current rho1: %f' % (rho1))
            violate = False
            # varify constraint at t0 first
            for i in range(numSample):
                cons0 = X0[i].T@Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))-rhodot0 # should <= 0
                cons1 = X1[i].T@Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))-rhodot1
                cons_upper = (X1[i]+x1-upper_x)@upper_S@(X1[i]+x1-upper_x)-upper_rho  # should <= 0
                print('    constraint:')
                print('    cons0: %f, cons1: %f, cons_upper: %f' % (cons0, cons1, cons_upper))
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
            print('valid rho1 found. Advance to the next rho0...')
            print("rho0 = %f, rho1 = %f" % (rho0, rho1))
            print('upper_rho: %f' % (upper_rho))
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
                if rho0 > rho_step:
                    #while rho0 <= rho_step:
                    #    rho_step = rho_step / 2  # smaller rho_step
                    rho0 = rho0 - rho_step
                    continue
        # if none of the above conditions, then it is time to return the value
        print('rho0 = %f, rho1 = %f' % (prev_rho0, prev_rho1))
        return prev_rho0, prev_rho1


def sample_tv_verify(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, system=None, numSample=50):
    # assume we are instead set rho1 to be upper_rho, and then obtain rho_dot
    # here we assume upper_S is the same as S1
    # system is used to calculate circular state difference
    U = np.random.normal(loc=0.0, scale=1.0, size=(numSample,len(S0)))
    # individually normalize each sample
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    tmp = np.linalg.pinv(S1)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U1 = U@scipy.linalg.sqrtm(tmp)
    # U1 satisfies: u1[i].T S1 u1[i] = 1
    Sdot1 = -(Q-S1@B1@np.linalg.pinv(R)@B1.T@S1+S1@A1+A1.T@S1)
    K1 = np.linalg.pinv(R)@B1.T@S1

    # in case x1 is not the same as upper_x, we solve the optimization to obtain rho
    delta_x = upper_x-x1
    if system is not None:
        circular = system.is_circular_topology()
        # if it is an angle
        for i in range(len(delta_x)):
            if circular[i]:
                # if it is angle
                # should not change the "sign" of the delta_x
                # map to [-pi, pi]
                delta_x[i] = delta_x[i] - np.floor(delta_x[i] / (2*np.pi))*(2*np.pi)
                # should not change the "sign" of the delta_x
                if delta_x[i] > np.pi:
                    delta_x[i] = delta_x[i] - 2*np.pi                
    delta = np.sqrt(delta_x.T@upper_S@delta_x)
    rho1 = upper_rho - delta
    print('delta:')
    print(delta)
    print('upper_rho')
    print(upper_rho)
    # here we use a more relaxed version, to deal with cases when upper_x is not x1
    # in this case, we only care about (x_bar + x1 - x_upper).T S1 (x_bar + x1 - x_upper) = upper_rho^2
    # then given our previous samples, x_bar = upper_rho * U1[i] + x_upper - x1
    #X1 = upper_rho * U1 + upper_x.reshape(1,-1) - x1.reshape(1,-1)
    # below is using the inner ellipsoid
    X1 = rho1 * U1
    
    # we then obtain rhodot1 by setting it to be the following:
    #   max_{xTSx=rho1^2}(d/dt(xTSx))
    # the intersection point is defined by:
    # x = rho_upper/delta(x1-x_upper)+x_upper
    if delta > 1e-6:
        # set a lowerbound for delta
        #X1 = np.append(X1, np.array([upper_rho/delta*(x1-upper_x)+upper_x]), axis=0)
        # only consider the intersection of ellipsoids
        X1 = np.array([upper_rho/delta*(x1-upper_x)+upper_x])
        print(X1.shape)
        
    rhodot1 = -1e8
    for i in range(len(X1)):
        cons = X1[i].T@Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))
        if cons > rhodot1:
            rhodot1 = cons
    # since we actually find d/dt(rho^2), it is 2rho rhodot
    rhodot1 = rhodot1 / 2 / rho1
    print('rhodot1: %f' % (rhodot1))
    # backpropagate one time step to find rho0
    rho0 = rho1 - rhodot1*(t1-t0)    
    # obtain rhodot0
    tmp = np.linalg.pinv(S0)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U0 = U@scipy.linalg.sqrtm(tmp)
    # U1 satisfies: u1[i].T S1 u1[i] = 1
    Sdot0 = -(Q-S0@B0@np.linalg.pinv(R)@B0.T@S0+S0@A0+A0.T@S0)
    K0 = np.linalg.pinv(R)@B0.T@S0
    X0 = rho0 * U0
    rhodot0 = -1e8
    for i in range(len(X0)):
        cons = X0[i].T@Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))
        if cons > rhodot0:
            rhodot0 = cons
    if rhodot0 > rhodot1:
        rhodot1 = (rhodot0 + rhodot1) / 2
    rho0 = rho1 - rhodot1*(t1-t0)
    return rho0, rho1


def sample_tv_verify_prev(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50):
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
    upper_rho_threshold = 1e-10
    #rho_step = upper_rho / 200.
    #rho0 = upper_rho-upper_rho_threshold
    rho0 = (1.05)*upper_rho
    rho_step = upper_rho / 1000.
    initial = True
    goingdown = False

    while True:
        rho0_violate = False
        rho1 = 1e-3
        rho1_grid = np.linspace(upper_rho*1.1, upper_rho*0.8, num=101)
        rho1_grid = np.append(rho1_grid, np.linspace(upper_rho*0.8, 0., num=51), axis=0)

        #rho1_grid = np.linspace(upper_rho-upper_rho_threshold, rho1, num=101)
        # find a rho1 in (0, upper_rho] that can make sure the constraints are valid
        # varify the constraints are true
        valid_rho1_found = False  # if one valid rho1 is found
        valid_rho1 = 1e-3
        #print('searching... current rho0: %f' % (rho0))
        for k in range(len(rho1_grid)):
            rho1 = rho1_grid[k]
            X0 = rho0*U0  # xTSx = rho^2
            X1 = rho1*U1
            # dot(rho^2) = 2rho*rhodot
            rhodot0 = (rho1-rho0)/(t1-t0)
            rhodot1 = (rho1-rho0)/(t1-t0)
            rhodot0 = 2 * rhodot0 * rho0
            rhodot1 = 2 * rhodot1 * rho1
            #print('    searching... current rho1: %f' % (rho1))
            violate = False
            # varify constraint at t0 first
            for i in range(numSample):
                cons0 = X0[i].T@Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))-rhodot0 # should <= 0
                cons1 = X1[i].T@Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))-rhodot1
                #print('x1-upper_x:')
                #print(x1-upper_x)
                cons_upper = (X1[i]+x1-upper_x)@upper_S@(X1[i]+x1-upper_x)-upper_rho*upper_rho  # xTSx=rho^2  # should <= 0
                #print('    constraint:')
                #print('    cons0: %f, cons1: %f, cons_upper: %f' % (cons0, cons1, cons_upper))
                #print('    cons0: %d, cons1: %d, cons_upper: %d' % (cons0>0, cons1>0, cons_upper>0))
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
            #print('upper_rho: %f' % (upper_rho))
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
                if rho0 > rho_step:
                    #while rho0 <= rho_step:
                    #    rho_step = rho_step / 2  # smaller rho_step
                    rho0 = rho0 - rho_step
                    continue
        # if none of the above conditions, then it is time to return the value
        #print('rho0 = %f, rho1 = %f' % (prev_rho0, prev_rho1))
        return prev_rho0, prev_rho1

def sample_tv_verify_lam(t0, t1, upper_x, upper_S, upper_rho, S0, S1, A0, A1, B0, B1, R, Q, x0, x1, u0, u1, func, numSample=50):
    # here we assume the lyapunov function is V(x) = xT(S/||S||*)x, we normalize S by its
    # largest eigenvalue to make sure we use a small rho to represent a large area
    # the derivative of (S/S_norm) = S_dot/S_norm + S(-1/(S_norm)^2)*v_max^TS_dot v_max
    # obtain the max eigenvalue of S0 and S1
    w, v = np.linalg.eig(S0)
    idx_S0 = np.argmax(w)
    lam_S0 = w[idx_S0]
    v_S0 = v[:,idx_S0]
    w, v = np.linalg.eig(S1)
    idx_S1 = np.argmax(w)
    lam_S1 = w[idx_S1]
    v_S1 = v[:,idx_S1]
    w, v = np.linalg.eig(upper_S)
    idx_upper_S = np.argmax(w)
    lam_upper_S = w[idx_upper_S]
    v_upper_S = v[:,idx_upper_S]
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
    U0 = U@scipy.linalg.sqrtm(tmp) * np.sqrt(lam_S0)
    tmp = np.linalg.pinv(S1)
    tmp = scipy.linalg.sqrtm(tmp.T @ tmp)
    U1 = U@scipy.linalg.sqrtm(tmp) * np.sqrt(lam_S1)

    Sdot0 = -(Q-S0@B0@np.linalg.pinv(R)@B0.T@S0+S0@A0+A0.T@S0)
    Sdot1 = -(Q-S1@B1@np.linalg.pinv(R)@B1.T@S1+S1@A1+A1.T@S1)
    normalized_Sdot0 = Sdot0 / lam_S0 + ((-1/lam_S0/lam_S0)*v_S0.T@Sdot0@v_S0) * S0
    normalized_Sdot1 = Sdot1 / lam_S1 + ((-1/lam_S1/lam_S1)*v_S1.T@Sdot1@v_S1) * S1

    K0 = np.linalg.pinv(R)@B0.T@S0
    K1 = np.linalg.pinv(R)@B1.T@S1
    prev_rho0 = 1e-6
    rho0 = 1e-6
    rho_alpha = 0.2
    prev_rho1 = 1e-6
    upper_rho_threshold = 1e-10
    rho_step = upper_rho / 100.
    #rho0 = upper_rho-upper_rho_threshold
    rho0 = upper_rho*2
    initial = True
    goingdown = False

    while True:
        rho0_violate = False
        rho1 = 1e-6
        rho1_grid = np.linspace(upper_rho*2, rho1, num=201)
        #rho1_grid = np.linspace(upper_rho-upper_rho_threshold, rho1, num=101)
        # find a rho1 in (0, upper_rho] that can make sure the constraints are valid
        # varify the constraints are true
        valid_rho1_found = False  # if one valid rho1 is found
        valid_rho1 = 1e-6
        #print('searching... current rho0: %f' % (rho0))
        for k in range(len(rho1_grid)):
            rho1 = rho1_grid[k]
            X0 = rho0*U0  # xTSx = rho^2
            X1 = rho1*U1
            # dot(rho^2) = 2rho*rhodot
            rhodot0 = (rho1-rho0)/(t1-t0)
            rhodot1 = (rho1-rho0)/(t1-t0)
            rhodot0 = 2 * rhodot0 * rho0
            rhodot1 = 2 * rhodot1 * rho1
            #print('searching... upper rho: %f' % (upper_rho))
            #print('searching... current rho0: %f' % (rho0))
            #print('    searching... current rho1: %f' % (rho1))
            violate = False
            # varify constraint at t0 first
            for i in range(numSample):
                cons0 = X0[i].T@normalized_Sdot0@X0[i] + 2*X0[i].T@S0@(func(x0+X0[i],u0-K0@X0[i])-func(x0,u0))-rhodot0 # should <= 0
                cons1 = X1[i].T@normalized_Sdot1@X1[i] + 2*X1[i].T@S1@(func(x1+X1[i],u1-K1@X1[i])-func(x1,u1))-rhodot1
                #print('x1-upper_x:')
                #print(x1-upper_x)
                cons_upper = (X1[i]+x1-upper_x)@upper_S@(X1[i]+x1-upper_x)/lam_upper_S-upper_rho*upper_rho  # xTSx=rho^2  # should <= 0
                #print('    constraint:')
                #print('    cons0: %f, cons1: %f, cons_upper: %f' % (cons0, cons1, cons_upper))
                #print('    cons0: %d, cons1: %d, cons_upper: %d' % (cons0>0, cons1>0, cons_upper>0))
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
            #print('upper_rho: %f' % (upper_rho))
            if initial:
                #rho_step = rho_step / 5 # use small step for stepping up
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
                if rho0 > rho_step:
                    #while rho0 <= rho_step:
                    #    rho_step = rho_step / 2  # smaller rho_step
                    rho0 = rho0 - rho_step
                    continue
        # if none of the above conditions, then it is time to return the value
        #print('rho0 = %f, rho1 = %f' % (prev_rho0, prev_rho1))
        return prev_rho0, prev_rho1


def sample_ti_verify_sqrtrho(xG, uG, S, K, func, numSample=50):
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
    #xTSx<=rho^2
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
        if violate:
            return prev_rho
        prev_rho = rho
        rho = rho * (1+rho_alpha)
