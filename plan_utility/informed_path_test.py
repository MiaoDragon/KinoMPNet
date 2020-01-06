import sys
sys.path.append('../deps/sparse_rrt')
#from ctypes import *
#ctypes.cdll.LoadLibrary('')
#lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
#lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")

from sparse_rrt.planners import SST
#from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
import pickle
from plan_utility.informed_path import *
from plan_utility.plan_general import *
from plan_utility.data_structure import *

bvp_solver = _sst_module.PSOPTBVPWrapper(_system, 2, 1, 0)

low = []
high = []
state_bounds = _system.get_state_bounds()
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])


f = open('data/pendulum/0/path_%d.pkl' % (0), 'rb')
state = pickle.load(f)

f = open('data/pendulum/0/control_%d.pkl' % (0), 'rb')
control = pickle.load(f)

f = open('data/pendulum/0/cost_%d.pkl' % (0), 'rb')
times = pickle.load(f)


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

traj_opt = lambda x0, x1: bvp_solver.solve(x0, x1, 500, 20, 100, 0.002)

start = Node(state[0])
goal = Node(state[-1])
goal.S1 = np.identity(2)
goal.rho1 = 0.01
target_reached = plan(None, state[0], state[-1], informer, dynamics, traj_opt, jac_A, jac_B, MAX_LENGTH=1000)
