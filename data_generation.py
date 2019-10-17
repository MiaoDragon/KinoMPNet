"""
using SST* to generate near-optimal paths in specified environment
"""
import argparse
from sparse_rrt import _sst_module
from sparse_rrt.systems import standard_cpp_systems
import numpy as np
import time
import pickle
from sparse_rrt.systems.acrobot import Acrobot, AcrobotDistance
from sparse_rrt.systems.point import Point

def main(args):
    # set up the environment
    if args.env_name == 'cartpole':
        env = standard_cpp_systems.CartPole()
    state_bounds = env.get_state_bounds()
    min_time_steps = 20
    max_time_steps = 200
    integration_step = 0.002
    low = []
    high = []
    for i in range(len(state_bounds)):
        low.append(state_bounds[i][0])
        high.append(state_bounds[i][1])
    ## TODO: add other env
    paths = []
    for i in range(args.N):
        # randomly sample collision-free start and goal
        while True:
            print('trial')
            start = np.random.uniform(low=low, high=high)
            end = np.random.uniform(low=low, high=high)
            planner = _sst_module.SSTWrapper(
                state_bounds=env.get_state_bounds(),
                control_bounds=env.get_control_bounds(),
                distance=env.distance_computer(),
                start_state=start,
                goal_state=end,
                goal_radius=0.5,
                random_seed=0,
                sst_delta_near=0.4,
                sst_delta_drain=0.2
            )
            # generate a path by using SST to plan for some maximal iterations
            time0 = time.time()
            for iter in range(args.max_iter):
                if iter % 100 == 0:
                    # from time to time use the goal
                    sample = end
                    planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                else:
                    sample = np.random.uniform(low=low, high=high)
                    planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                    #planner.step(env, min_time_steps, max_time_steps, integration_step)
                #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                solution = planner.get_solution()
                if solution is not None:
                    break
            print('spent time: %f' % (time.time() - time0))
            solution = planner.get_solution()
            if solution is None:
                continue
            else:
                print('path %d: succeeded.' % (i))
                path, controls, costs = solution
                print(path)
                path = np.array(path)
                paths.append(path)
                break
    # store the paths to file
    file = open(args.path_file, 'wb')
    pickle.dump(paths, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cartpole')
    parser.add_argument('--N', type=int, default=40000)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--path_file', type=str, default='./data/cartpole/train.pkl')
    args = parser.parse_args()
    main(args)
