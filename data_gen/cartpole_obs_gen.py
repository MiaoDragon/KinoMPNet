import sys
sys.path.append('..')
from tools.pcd_generation import rectangle_pcd
import numpy as np
def obs_gen(N, N_obs, N_pc=1400, width=4):
    H = 0.5
    L = 2.5
    near = width * 1.2
    for i in range(N):
        obs_single = []
        for j in range(N_obs):
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
    obc_list = rectangle_pcd(obs_list, width, N_pc)
    return obs_list, obc_list
