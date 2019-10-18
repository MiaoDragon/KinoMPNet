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
import os

from tools.pcd_generation import rectangle_pcd

def main(args):
    # set up the environment
    if args.env_name == 'cartpole':
        env_constr = standard_cpp_systems.CartPole
        obs_list = None
        obc_list = None
        goal_radius=1.5
        random_seed=0
        sst_delta_near=2.
        sst_delta_drain=1.2
        min_time_steps = 10
        max_time_steps = 200
        integration_step = 0.002
    elif args.env_name == 'cartpole_obs':
        env_constr = standard_cpp_systems.CartPoleObs
        # randomly generate obstacle location
        obs_list = []
        width = 4.
        H = 0.5
        L = 2.5
        goal_radius=1.5
        random_seed=0
        sst_delta_near=2.
        sst_delta_drain=1.2
        min_time_steps = 10
        max_time_steps = 200
        integration_step = 0.002
        near = width * 1.2
        print('generating obs...')
        for i in range(args.N):
            obs_single = []
            for j in range(args.N_obs):
                low_h = - width/2 - L
                high_h = width/2 + L
                '''
                make sure the obstacle does not block the pole entirely
                by making sure the fixed point of the pole is not in the obs
                hence the valid range for y axis is:
                H + low_h ~ H - width/2, H + width/2 ~ H + high_h
                '''
                while True:
                    # first randomly see if it is left or right
                    side = np.random.randint(low=0, high=2)
                    # 0: left, 1: right
                    if side == 0:
                        obs = np.random.uniform(low=[-20, H+low_h], high=[20, H-width/2])
                    else:
                        obs = np.random.uniform(low=[-20, H+width/2], high=[20, H+high_h])
                    too_near = False
                    for k in range(len(obs_single)):
                        if np.linalg.norm(obs-obs_single[k]) < near:
                            too_near = True
                            break
                    if not too_near:
                        break

                obs_single.append(obs)
            obs_single = np.array(obs_single)
            obs_list.append(obs_single)
        obs_list = np.array(obs_list)
        # convert from obs to point cloud
        obc_list = rectangle_pcd(obs_list, width, 1400)

    ## TODO: add other env
    # store the obstacles and obc first
    file = open(args.obs_file, 'wb')
    pickle.dump(obs_list, file)
    file = open(args.obc_file, 'wb')
    pickle.dump(obc_list, file)

    for i in range(args.N):
        # load the obstacle by creating a new environment
        if args.env_name == 'cartpole':
            env = env_constr()
        elif args.env_name == 'cartpole_obs':
            env = env_constr(obs_list[i], width)

        state_bounds = env.get_state_bounds()
        low = []
        high = []
        for j in range(len(state_bounds)):
            low.append(state_bounds[j][0])
            high.append(state_bounds[j][1])

        paths = []
        actions = []
        costs = []
        for j in range(args.NP):
            # randomly sample collision-free start and goal
            while True:
                print('trial')
                start = np.random.uniform(low=low, high=high)
                end = np.random.uniform(low=low, high=high)
                # set the velocity terms to zero
                if args.env_name == 'cartpole':
                    start[1] = 0.
                    start[3] = 0.
                    end[1] = 0.
                    end[3] = 0.
                elif args.env_name == 'cartpole_obs':
                    start[1] = 0.
                    start[3] = 0.
                    end[1] = 0.
                    end[3] = 0.
                planner = _sst_module.SSTWrapper(
                    state_bounds=env.get_state_bounds(),
                    control_bounds=env.get_control_bounds(),
                    distance=env.distance_computer(),
                    start_state=start,
                    goal_state=end,
                    goal_radius=goal_radius,
                    random_seed=random_seed,
                    sst_delta_near=sst_delta_near,
                    sst_delta_drain=sst_delta_drain
                )
                # generate a path by using SST to plan for some maximal iterations
                time0 = time.time()
                print('obs: %d, path: %d' % (i, j))
                for iter in range(args.max_iter):
                    #if iter % 100 == 0:
                    #    # from time to time use the goal
                    #    sample = end
                    #    planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                    #else:
                        #sample = np.random.uniform(low=low, high=high)
                        #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                    planner.step(env, min_time_steps, max_time_steps, integration_step)
                    #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
                    #solution = planner.get_solution()
                    # don't break the searching to find better solutions
                    #if solution is not None:
                    #    break
                print('spent time: %f' % (time.time() - time0))
                solution = planner.get_solution()
                if solution is None:
                    continue
                else:
                    print('path %d: succeeded.' % (j))
                    path, controls, cost = solution
                    print(path)
                    path = np.array(path)
                    paths.append(path)
                    controls = np.array(controls)
                    actions.append(controls)
                    cost = np.array(cost)
                    costs.append(cost)
                    break
        # store the paths to file
        # check if obstacle exists, if not, then directly store at path_folder
        if obs_list is None:
            file = open(args.path_folder+args.path_file, 'wb')
            pickle.dump(paths, file)
            file = open(args.path_folder+args.action_file, 'wb')
            pickle.dump(actions, file)
            file = open(args.path_folder+args.cost_file, 'wb')
            pickle.dump(costs, file)

        else:
            # create a new directory under path_folder
            dir = args.path_folder+str(i)+'/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            file = open(dir+args.path_file, 'wb')
            pickle.dump(paths, file)
            file = open(dir+args.action_file, 'wb')
            pickle.dump(actions, file)
            file = open(dir+args.cost_file, 'wb')
            pickle.dump(costs, file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cartpole')
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--N_obs', type=int, default=6)
    parser.add_argument('--NP', type=int, default=5000)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--path_folder', type=str, default='./data/cartpole/')
    parser.add_argument('--path_file', type=str, default='path.pkl')
    parser.add_argument('--action_file', type=str, default='action.pkl')
    parser.add_argument('--cost_file', type=str, default='cost.pkl')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    args = parser.parse_args()
    main(args)
