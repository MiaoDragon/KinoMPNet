from ctypes import *
#import faulthandler
#faulthandler.enable()
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")
from sparse_rrt.planners import SST
from env.cartpole_obs import CartPoleObs
#from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd

#obs_list = np.array(obs_list)
system = standard_cpp_systems.CartPole()

bvp_solver = _sst_module.BVPWrapper(system, 4, 1, 12, 0.002)
start = np.array([0., 0., 0., 0.])
goal = np.array([1., 0., np.pi/12, 0.])
solution = bvp_solver.solve(start, goal, 5000)
print(solution)
#solution = bvp_solver.solve(start, goal)
#print(solution)
