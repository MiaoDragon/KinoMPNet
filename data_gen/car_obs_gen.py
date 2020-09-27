import sys
sys.path.append('..')
from tools.pcd_generation import rectangle_pcd
import numpy as np
def obs_gen(N, N_obs, N_pc=1400, width=4):
    MIN_X = -25
    MAX_X = 25
    MIN_Y = -35
    MAX_Y = 35
    obs_list = []
    near = width * 2
    print('generating obs...')
    for i in range(N):
        obs_single = []
        for j in range(N_obs):
            low_x = -25 + width/2
            high_x = 25 - width/2
            low_y = -35 + width/2
            high_y = 35 - width/2
            while True:
                # randomly sample in the entire space
                obs = np.random.uniform(low=[low_x, low_y], high=[high_x, high_y])
                # check if it is near enough to previous obstacles
                too_near = False
                for k in range(len(obs_single)):
                    if np.linalg.norm(obs - obs_single[k]) < near:
                        too_near = True
                        break
                if not too_near:
                    break

            obs_single.append(obs)
        obs_single = np.array(obs_single)
        obs_list.append(obs_single)
    obs_list = np.array(obs_list)
    # convert from obs to point cloud
    obc_list = rectangle_pcd(obs_list, width, N_pc)
    return obs_list, obc_list
