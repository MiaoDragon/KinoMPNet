from ctypes import *
cdll.LoadLibrary('deps/sparse_rrt/deps/trajopt/build/lib/libsco.so')

from sparse_rrt.planners import SST
from env.cartpole_obs import CartPoleObs
from env.cartpole import CartPole
from sparse_rrt.systems import standard_cpp_systems
from sparse_rrt import _sst_module
import numpy as np
import time
from tools.pcd_generation import rectangle_pcd

system = standard_cpp_systems.CartPole()


#obs_list = np.array(obs_list)
#system = standard_cpp_systems.CartPoleObs(obs_list, 4.)
#system = CartPoleObs(obs_list)
# Create SST planner
min_time_steps = 10
max_time_steps = 200
integration_step = 0.002
max_iter = 100
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
        planner.step_with_sample(system, sample, min_time_steps, max_time_steps, integration_step)
    else:
        sample = np.random.uniform(low=low, high=high)
        new_sample = planner.step_with_sample(system, sample, min_time_steps, max_time_steps, integration_step)
        print('iteration: %d' % (iteration))
        print('sample:')
        print(sample)
        print('new_sample:')
        print(new_sample)
        # interation: 0.002
        #planner.step_with_sample(system, sample, 2, 20, 0.01)
    
    #if iteration % 100 == 0:
solution = planner.get_solution()
print("Solution: %s, Number of nodes: %s" % (planner.get_solution(), planner.get_number_of_nodes()))

print('time spent: %f' % (time.time() - time0))
assert solution is not None