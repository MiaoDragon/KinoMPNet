import sys
import jax
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
from tools import data_loader
from sparse_rrt.planners import SST
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import pickle
import scipy
from plan_utility.informed_path import *
from plan_utility.plan_general import *
from plan_utility.data_structure import *
from plan_utility import pendulum, acrobot_obs

from tvlqr.python_lyapunov import *
env_type = 'acrobot_obs'
data_folder = '../data/acrobot_obs/'
# setup evaluation function and load function
if env_type == 'pendulum':
    IsInCollision =pendulum.IsInCollision
    normalize = pendulum.normalize
    unnormalize = pendulum.unnormalize
    obs_file = None
    obc_file = None
    dynamics = pendulum.dynamics
    jax_dynamics = pendulum.jax_dynamics
    enforce_bounds = pendulum.enforce_bounds

    obs_f = False
    #system = standard_cpp_systems.PSOPTPendulum()
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 2, 1, 0)
elif env_type == 'cartpole_obs':
    IsInCollision =cartpole.IsInCollision
    normalize = cartpole.normalize
    unnormalize = cartpole.unnormalize
    obs_file = None
    obc_file = None
    dynamics = cartpole.dynamics
    jax_dynamics = cartpole.jax_dynamics
    enforce_bounds = cartpole.enforce_bounds

    obs_f = True
    #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'cartpole')
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
elif env_type == 'acrobot_obs':
    IsInCollision =acrobot_obs.IsInCollision
    normalize = acrobot_obs.normalize
    unnormalize = acrobot_obs.unnormalize
    obs_file = None
    obc_file = None
    dynamics = acrobot_obs.dynamics
    jax_dynamics = acrobot_obs.jax_dynamics
    enforce_bounds = acrobot_obs.enforce_bounds
    obs_f = True
    #system = standard_cpp_systems.RectangleObs(obs_list, args.obs_width, 'acrobot')
    #bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)

jac_A = jax.jacfwd(jax_dynamics, argnums=0)
jac_B = jax.jacfwd(jax_dynamics, argnums=1)


test_data = data_loader.load_test_dataset(1, 5, data_folder, sp=1, obs_f=None)
# data_type: seen or unseen
obc, obs, paths, path_lengths, controls, costs = test_data

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
        step_sz = 0.02
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 4, 0.02)
        #goal_S0 = np.identity(4)
        goal_S0 = np.diag([1.,1.,0.,0.])
        goal_rho0 = 1.5
    elif env_type == 'acrobot_obs_2':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0
    elif env_type == 'acrobot_obs_3':
        system = _sst_module.PSOPTAcrobot()
        bvp_solver = _sst_module.PSOPTBVPWrapper(system, 4, 1, 0)
        traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 1, 50, 0.002)
        goal_S0 = np.identity(4)
        goal_rho0 = 1.0

    for j in range(len(paths[0])):
        state = paths[i][j]
        # obtain the sequence
        p_start = paths[i][j][0]
        detail_paths = [p_start]
        for k in range(len(controls[i][j])):
            for step in range(int(costs[i][j][k]/step_sz)):
                p_start = p_start + step_sz*dynamics(p_start, controls[i][j][k])
                p_start = enforce_bounds(p_start)
                detail_paths.append(p_start)
        #detail_paths.append(paths[i][j][-1])
        state = detail_paths[::30]
        
        def informer(env, x0, xG, direction):
            # here we find the nearest point to x0 in the data, and depending on direction, find the adjacent node
            dis = np.abs(x0.x - state)
            circular = system.is_circular_topology()
            for i in range(len(x0.x)):
                if circular[i]:
                    # if it is angle
                    dis[:,i] = (dis[:,i] > np.pi) * (2*np.pi - dis[:,i]) + (dis[:,i] <= np.pi) * dis[:,i]
            #S = np.identity(len(x0.x))
            S = np.diag([1.,1.,0.,0.])
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
            goal = Node(paths[i][j][-2])
            #goal = Node(path[-1])
            goal.S0 = goal_S0
            goal.rho0 = goal_rho0    # change this later

            control = []
            time_step = []
            
            MAX_NEURAL_REPLAN = 11
            if obs is None:
                obs_i = None
                obc_i = None
            else:
                obs_i = obs[i]
                obc_i = obc[i]
            for t in range(MAX_NEURAL_REPLAN):
                # adaptive step size on replanning attempts
                res, path_list = plan(None, start, goal, detail_paths, informer, system, dynamics, \
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
