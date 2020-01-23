from ctypes import *
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("../deps/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("../deps/sparse_rrt/deps/trajopt/build/lib/libutils.so")

import sys
import jax
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
from tools import dataloader
from sparse_rrt.planners import SST
#from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import pickle
import scipy
from plan_utility.informed_path import *
from plan_utility.plan_general import *
from plan_utility.data_structure import *
from tvlqr.python_lyapunov import *

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

def enforce_bounds(state):
    MIN_ANGLE, MAX_ANGLE = -np.pi, np.pi
    MIN_W, MAX_W = -7., 7

    if state[0] < -np.pi:
        state[0] += 2*np.pi
    elif state[0] > np.pi:
        state[0] -= 2 * np.pi
    state = np.clip(state, [MIN_ANGLE, MIN_W], [MAX_ANGLE, MAX_W])
    return state

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

def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151

    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

    eigVals, eigVecs = scipy.linalg.eig(A-B*K)

    return K, X, eigVals

jac_A = jax.jacfwd(jax_dynamics, argnums=0)
jac_B = jax.jacfwd(jax_dynamics, argnums=1)

data_folder = '../data/pendulum/'
test_data = data_loader.load_test_dataset(1, 5, '../data/pendulum/', obs_f=False)
# data_type: seen or unseen
obc, obs, paths, path_lengths = test_data

fes_env = []   # list of list
valid_env = []
time_env = []
time_total = []
for i in range(len(paths)):
    time_path = []
    fes_path = []   # 1 for feasible, 0 for not feasible
    valid_path = []      # if the feasibility is valid or not
    # save paths to different files, indicated by i
    #print(obs, flush=True)
    env_type = 'pendulum'
    # feasible paths for each env
    if env_type == 'pendulum':
        system = standard_cpp_systems.PSOPTPendulum()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 20, 0.002)
    elif env_type == 'cartpole_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 4.0, 'cartpole')
        system = _sst_module.CartPole()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type == 'acrobot_obs':
        #system = standard_cpp_systems.RectangleObs(obs[i], 6.0, 'acrobot')
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif args.env_type == 'acrobot_obs_2':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif args.env_type == 'acrobot_obs_3':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0

    for j in range(len(paths[0])):

        file = open(data_folder+'0/start_goal_%d.pkl' % (i), 'rb')
        p = pickle._Unpickler(file)
        p.encoding = 'latin1'
        start_goal = p.load()
        A = jax.jacfwd(jax_dynamics, argnums=0)(start_goal[1], stable_u(start_goal[1]))
        B = jax.jacfwd(jax_dynamics, argnums=1)(start_goal[1], stable_u(start_goal[1]))
        A = np.asarray(A)
        B = np.asarray(B)
        Q = np.identity(2)
        R = np.identity(1)
        K, lqr_S, E = lqr(A, B, Q, R)


        start = Node(start_goal[0])
        goal = Node(start_goal[1])

        goal.S0 = np.identity(2)
        goal.rho0 = 1.0
        lqr_rho = sample_ti_verify(xG, uG, lqr_S, K, dynamics, numSample=1000)
        goal.rho0 = lqr_rho
        goal.S0 = lqr_S

        state = paths[i][j]
        state = np.append(state, start_goal[1], axis=0)
        def informer(env, x0, xG, direction):
            # here we find the nearest point to x0 in the data, and depending on direction, find the adjacent node
            dis = np.abs(x0.x - state)
            circular = system.is_circular_topology()
            for i in range(len(x0.x)):
                if circular[i]:
                    # if it is angle
                    dis[:,i] = (dis[:,i] > np.pi) * (2*np.pi - dis[:,i]) + (dis[:,i] <= np.pi) * dis[:,i]
            S = np.identity(len(x0.x))
            #S = np.diag([1/30./30., 1/40./40., 1., 1.])
            #dif = np.sqrt(dis.T@S@dis)
            dif = []
            for i in range(len(dis)):
                dif.append(np.sqrt(dis[i].T@S@dis[i]))
            dif = np.array(dif)
            #dif = np.linalg.norm(dis, axis=1)
            max_d_i = np.argmin(dif)
            #print('current state: ')
            #print(x0.x)
            #print('chosen data:')
            #print(state[max_d_i])

            if direction == 0:
                # forward
                if max_d_i+1 == len(state):
                    next_idx = max_d_i
                else:
                    next_idx = max_d_i+1
                res = Node(state[next_idx])

            else:
                if max_d_i-1 == -1:
                    next_idx = max_d_i
                else:
                    next_idx = max_d_i-1
                res = Node(state[next_idx])
            return res

        time0 = time.time()
        time_norm = 0.
        fp = 0 # indicator for feasibility
        print ("step: i="+str(i)+" j="+str(j))
        p1_ind=0
        p2_ind=0
        p_ind=0
        if path_lengths[i][j]==0:
            # invalid, feasible = 0, and path count = 0
            fp = 0
            valid_path.append(0)
        if path_lengths[i][j]>0:
            fp = 0
            valid_path.append(1)
            #paths[i][j][0][1] = 0.
            #paths[i][j][path_lengths[i][j]-1][1] = 0.
            path = [paths[i][j][0], paths[i][j][path_lengths[i][j]-1]]
            # plot the entire path
            #plt.plot(paths[i][j][:,0], paths[i][j][:,1])

            start = Node(path[0])
            goal = Node(path[-1])
            goal.S0 = goal_S0
            goal.rho0 = goal_rho0    # change this later

            control = []
            time_step = []
            step_sz = DEFAULT_STEP
            MAX_NEURAL_REPLAN = 11
            if obs is None:
                obs_i = None
                obc_i = None
            else:
                obs_i = obs[i]
                obc_i = obc[i]
            for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                res, path_list = plan(obc[i], start, goal, paths[i][j], informer, system, dynamics, \
                           enforce_bounds, traj_opt, jac_A, jac_B, step_sz=step_sz, MAX_LENGTH=1000)
                #print('after neural replan:')
                #print(path)
                #path = lvc(path, obc[i], IsInCollision, step_sz=step_sz)
                #print('after lvc:')
                #print(path)
                if res:
                    fp = 1
                    print('feasible ok!')
                    break
                #if feasibility_check(bvp_solver, path, obc_i, IsInCollision, step_sz=0.01):
                #    fp = 1
                #    print('feasible, ok!')
                #    break
        if fp:
            # only for successful paths
            time1 = time.time() - time0
            time1 -= time_norm
            time_path.append(time1)
            print('test time: %f' % (time1))
            # write the path
            #print('planned path:')
            #print(path)
            #path = np.array(path)
            #np.savetxt('results/path_%d.txt' % (j), path)
            #np.savetxt('results/control_%d.txt' % (j), np.array(control))
            #np.savetxt('results/timestep_%d.txt' % (j), np.array(time_step))

        fes_path.append(fp)
