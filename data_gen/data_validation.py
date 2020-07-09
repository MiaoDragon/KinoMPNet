# this validate if the data generated is diverse enough by comparing start and goal


"""
using SST* to generate near-optimal paths in specified environment
"""
import sys
sys.path.append('../deps/sparse_rrt')
sys.path.append('..')
import argparse
import tools.data_loader as data_loader
import numpy as np
def main(args):
    obc, obs, path, sg, path_length, control, cost = data_loader.load_test_dataset(args.N, args.NP, args.path_folder, obs_f=True, s=0, sp=0)
    id_list = []
    for i in range(len(sg)):
        for j in range(len(sg[0])):
            #id_new = str(round(np.sum(sg[i][j][0] + sg[i][j][1]), 1))
            id_new = str(round(np.sum(sg[i][j][0]), 1)) + ", " + str(round(np.sum(sg[i][j][1]), 4))

            print("id_new: %s" % (id_new))
            if id_new in id_list:
                print('same start goal!')
                return
            id_list.append(id_new)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='cartpole_obs')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--N_obs', type=int, default=6)
    parser.add_argument('--s', type=int, default=0)
    parser.add_argument('--sp', type=int, default=0)
    parser.add_argument('--NP', type=int, default=1000)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--path_folder', type=str, default='../data/cartpole_obs/')
    parser.add_argument('--path_file', type=str, default='path')
    parser.add_argument('--control_file', type=str, default='control')
    parser.add_argument('--cost_file', type=str, default='cost')
    parser.add_argument('--time_file', type=str, default='time')
    parser.add_argument('--sg_file', type=str, default='start_goal')
    parser.add_argument('--obs_file', type=str, default='./data/cartpole/obs.pkl')
    parser.add_argument('--obc_file', type=str, default='./data/cartpole/obc.pkl')
    args = parser.parse_args()
    main(args)
