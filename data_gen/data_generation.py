"""
using SST* to generate near-optimal paths in specified environment
"""
import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
import argparse
from sparse_rrt import _sst_module
from sparse_rrt.systems import standard_cpp_systems
import numpy as np
import time
import pickle
from sparse_rrt.systems.acrobot import Acrobot, AcrobotDistance
from sparse_rrt.systems.point import Point
import os
import gc
from multiprocessing import Process, Queue
from data_gen import cartpole_obs_gen, acrobot_obs_gen, cartpole_sg_gen, acrobot_sg_gen


def main(args):
    # set up the environment
    if args.env_name == 'pendulum':
        env_constr = standard_cpp_systems.PSOPTPendulum
        min_time_steps = 10
        max_time_steps = 200
        integration_step = 0.002
        max_iter = 1000
        goal_radius=0.1
        random_seed=0
        sst_delta_near=0.02
        sst_delta_drain=0.01
    elif args.env_name == 'cartpole':
        env_constr = standard_cpp_systems.PSOPTCartPole
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
        env_constr = standard_cpp_systems.RectangleObs
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
        obs_list, obc_list = cartpole_obs_gen.obs_gen(args.N, args.N_obs, N_pc=1400, width=width)
        ## TODO: add other env
        # store the obstacles and obc first
        for i in range(len(obs_list)):
            file = open(args.path_folder+'obs_%d.pkl' % (i+args.s), 'wb')
            pickle.dump(obs_list[i], file)
            file = open(args.path_folder+'obc_%d.pkl' % (i+args.s), 'wb')
            pickle.dump(obc_list[i], file)

    elif args.env_name == 'acrobot_obs':
        env_constr = standard_cpp_systems.RectangleObs
        # randomly generate obstacle location
        obs_list = []
        LENGTH = 20.
        width = 6.
        near = width * 1.2
        s_g_dis_threshold = LENGTH * 1.5
        goal_radius=2.0
        random_seed=0
        sst_delta_near=1.0
        sst_delta_drain=0.5
        #min_time_steps = 10
        #max_time_steps = 200
        #integration_step = 0.002
        min_time_steps = 5
        max_time_steps = 100
        integration_step = 0.02
        print('generating obs...')
        obs_list, obc_list = acrobot_obs_gen.obs_gen(args.N, args.N_obs, N_pc=1400, width=width)
        ## TODO: add other env
        # store the obstacles and obc first
        for i in range(len(obs_list)):
            file = open(args.path_folder+'obs_%d.pkl' % (i+args.s), 'wb')
            pickle.dump(obs_list[i], file)
            file = open(args.path_folder+'obc_%d.pkl' % (i+args.s), 'wb')
            pickle.dump(obc_list[i], file)

    ####################################################################################
    def plan_one_path_bvp(env, start, end, out_queue, path_file, control_file, cost_file, time_file):
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
        #print('obs: %d, path: %d' % (i, j))
        for iter in range(args.max_iter):
            if iter % 1000 == 0:
                print('still alive after %d iterations' % (iter))
            if iter % 100 == 0:
                # from time to time use the goal
                sample = end
                #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
            else:
                sample = np.random.uniform(low=low, high=high)
                #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
            #planner.step(env, min_time_steps, max_time_steps, integration_step)
            planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
            solution = planner.get_solution()
            # don't break the searching to find better solutions
            #if solution is not None:
            #    break
        plan_time = time.time() - time0
        if solution is None:
            out_queue.put(0)
        else:
            print('path succeeded.')
            path, controls, cost = solution
            print(path)
            path = np.array(path)
            controls = np.array(controls)
            cost = np.array(cost)

            file = open(path_file, 'wb')
            pickle.dump(path, file)
            file.close()
            file = open(control_file, 'wb')
            pickle.dump(controls, file)
            file.close()
            file = open(cost_file, 'wb')
            pickle.dump(cost, file)
            file.close()
            file = open(time_file, 'wb')
            pickle.dump(plan_time, file)
            file.close()
            out_queue.put(1)

    def plan_one_path_sst(env, start, end, out_queue, path_file, control_file, cost_file, time_file):
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
        #print('obs: %d, path: %d' % (i, j))
        for iter in range(args.max_iter):
            #print('iteration: %d' % (iter))
            planner.step(env, min_time_steps, max_time_steps, integration_step)
            #print('after step')
            #planner.step_with_sample(env, sample, min_time_steps, max_time_steps, integration_step)
            #solution = planner.get_solution()
            # don't break the searching to find better solutions
            #if solution is not None:
            #    break
        solution = planner.get_solution()
        plan_time = time.time() - time0
        if solution is None:
            out_queue.put(0)
        else:
            print('path succeeded.')
            path, controls, cost = solution
            print(path)
            path = np.array(path)
            controls = np.array(controls)
            cost = np.array(cost)

            file = open(path_file, 'wb')
            pickle.dump(path, file)
            file.close()
            file = open(control_file, 'wb')
            pickle.dump(controls, file)
            file.close()
            file = open(cost_file, 'wb')
            pickle.dump(cost, file)
            file.close()
            file = open(time_file, 'wb')
            pickle.dump(plan_time, file)
            file.close()
            out_queue.put(1)
    ####################################################################################
    queue = Queue(1)
    for i in range(args.N):
        # load the obstacle by creating a new environment
        if args.env_name == 'pendulum':
            env = env_constr()
        elif args.env_name == 'cartpole':
            env = env_constr()
        elif args.env_name == 'cartpole_obs':
            env = env_constr(obs_list[i], width, 'cartpole')
        elif args.env_name == 'acrobot_obs':
            env = env_constr(obs_list[i], width, 'acrobot')
        # generate rec representation of obs
        obs_recs = []
        for k in range(len(obs_list[i])):
            # for each obs setting
            obs_recs.append([[obs_list[i][k][0]-width/2,obs_list[i][k][1]-width/2],
                             [obs_list[i][k][0]-width/2,obs_list[i][k][1]+width/2],
                             [obs_list[i][k][0]+width/2,obs_list[i][k][1]+width/2],
                             [obs_list[i][k][0]+width/2,obs_list[i][k][1]-width/2]])

        state_bounds = env.get_state_bounds()
        low = []
        high = []
        for j in range(len(state_bounds)):
            low.append(state_bounds[j][0])
            high.append(state_bounds[j][1])

        paths = []
        actions = []
        costs = []
        times = []
        suc_n = 0

        for j in range(args.NP):
            plan_start = time.time()
            while True:
                # randomly sample collision-free start and goal
                #start = np.random.uniform(low=low, high=high)
                #end = np.random.uniform(low=low, high=high)


                # set the velocity terms to zero
                if args.env_name == 'pendulum':
                    #start[1] = 0.
                    start = np.random.uniform(low=low, high=high)
                    end = np.random.uniform(low=low, high=high)
                    end[1] = 0.
                elif args.env_name == 'cartpole':
                    start, end = cartpole_sg_gen.start_goal_gen(low, high, width, obs_list[i], obs_recs)
                elif args.env_name == 'cartpole_obs':
                    start, end = cartpole_sg_gen.start_goal_gen(low, high, width, obs_list[i], obs_recs)
                elif args.env_name == 'acrobot_obs':
                    start, end = acrobot_sg_gen.start_goal_gen(low, high, width, obs_list[i], obs_recs)
                dir = args.path_folder+str(i+args.s)+'/'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path_file = dir+args.path_file+'_%d'%(j+args.sp) + ".pkl"
                control_file = dir+args.control_file+'_%d'%(j+args.sp) + ".pkl"
                cost_file = dir+args.cost_file+'_%d'%(j+args.sp) + ".pkl"
                time_file = dir+args.time_file+'_%d'%(j+args.sp) + ".pkl"
                sg_file = dir+args.sg_file+'_%d'%(j+args.sp)+".pkl"
                p = Process(target=plan_one_path_sst, args=(env, start, end, queue, path_file, control_file, cost_file, time_file))
                p.start()
                p.join()
                res = queue.get()
                print('obtained result:')
                print(res)
                if res:
                    # plan successful
                    file = open(sg_file, 'wb')
                    sg = [start, end]
                    pickle.dump(sg, file)
                    file.close()
                    break
            print('path planning time: %f' % (time.time() - plan_start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cartpole')
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--N_obs', type=int, default=6)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--NP', type=int, default=5000)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--path_folder', type=str, default='./data/cartpole/')
    parser.add_argument('--path_file', type=str, default='path')
    parser.add_argument('--control_file', type=str, default='control')
    parser.add_argument('--cost_file', type=str, default='cost')
    parser.add_argument('--time_file', type=str, default='time')
    parser.add_argument('--sg_file', type=str, default='start_goal')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    args = parser.parse_args()
    main(args)
