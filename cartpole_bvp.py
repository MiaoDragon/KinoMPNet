from ctypes import *
#ctypes.cdll.LoadLibrary('')
lib1 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libsco.so")
lib2 = CDLL("/home/yinglong/Documents/kinodynamic/sparse_rrt/deps/trajopt/build/lib/libutils.so")

from sparse_rrt.planners import SST
from env.cartpole_obs import CartPoleObs
from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd

#obs_list = np.array(obs_list)
system = standard_cpp_systems.CartPole()

#bvp_solver = _sst_module.BVPWrapper(system, 4, 1, 24, 0.002)



#obs_list = np.array(obs_list)
system = standard_cpp_systems.CartPole()
#system = CartPoleObs(obs_list)
# Create SST planner
min_time_steps = 10
max_time_steps = 200
integration_step = 0.002
max_iter = 200
goal_radius=1.5
random_seed=0
sst_delta_near=2.0
sst_delta_drain=1.2

low = []
high = []
state_bounds = system.get_state_bounds()
for i in range(len(state_bounds)):
    low.append(state_bounds[i][0])
    high.append(state_bounds[i][1])
    
start = np.random.uniform(low=low, high=high)
end = np.random.uniform(low=low, high=high)


start[1] = 0.
start[3] = 0.
end[1] = 0.
end[3] = 0.
planner = SST(
    state_bounds=system.get_state_bounds(),
    control_bounds=system.get_control_bounds(),
    distance=system.distance_computer(),
    start_state=start,
    goal_state=end,
    goal_radius=goal_radius,
    random_seed=0,
    sst_delta_near=sst_delta_near,
    sst_delta_drain=sst_delta_drain
)

    
# Run planning and print out solution is some statistics every few iterations.
time0 = time.time()
for iteration in range(max_iter):
    if iteration % 50 == 0:
        # from time to time use the goal
        sample = end
        #planner.step_with_sample(system, sample, 20, 200, 0.002)
    else:
        #planner.step(system, min_time_steps, max_time_steps, integration_step)
        sample = np.random.uniform(low=low, high=high)
        print('iteration: %d' % (iteration))
        # interation: 0.002
    new_sample = planner.step_with_sample(system, sample, 2, 20, 0.002)
    print('sample:')
    print(sample)
    print('new sample:')
    print(new_sample)
    #if iteration % 100 == 0:
    solution = planner.get_solution()
    if solution is not None:
        break
    
solution = planner.get_solution()
print("Solution: %s, Number of nodes: %s" % (planner.get_solution(), planner.get_number_of_nodes()))

print('time spent: %f' % (time.time() - time0))
assert solution is not None