import sys
sys.path.append('..')
from tools.pcd_generation import rectangle_pcd
import numpy as np
def obs_gen(N, N_obs, N_pc=1400, width=4):
    # generate one obs in each phase
    LENGTH = 20.
    near = LENGTH * 1.2
    obs_list = []
    bottom_threshold = width / 10
    for i in range(N):
        obs_single = []
        for j in range(N_obs):
            '''
            The obstacle should be inside the maximal circle, but also not blocking (0,0)
            '''
            while True:
                obs = np.random.normal(size=2)
                obs = obs / np.linalg.norm(obs) * (LENGTH * 2 - LENGTH / 2)
                if j % 4 == 0:
                    obs = np.abs(obs)  # +,+
                elif j % 4 == 1:
                    obs = np.abs(obs)
                    obs[0] = -obs[0]  # -,+
                elif j % 4 == 2:
                    obs = np.abs(obs)
                    obs[1] = -obs[1]  # +,-
                elif j % 4 == 3:
                    obs = -np.abs(obs)
                while True:
                    # make sure it does not block (0,0)
                    alpha = np.random.uniform(low=0.8, high=1.)
                    obs_ = alpha * obs
                    # make sure that when the obstacle is below x-axis, it does not intersect with the pole
                    if j % 4 == 1:
                        obs_[0] = min(obs_[0], -width/2-bottom_threshold)
                    if j % 4 == 2:
                        obs_[0] = max(obs_[0], width/2+bottom_threshold)

                    if np.abs(obs_).max() > width/2:
                        obs = obs_
                        break
                # see if it is cluttered enough by making sure obstacles can't be
                # too close
                too_near = False
                for k in range(len(obs_single)):
                    if np.linalg.norm(obs-obs_single[k]) < near:
                        too_near = True
                        #print("too near")
                        break
                if not too_near:
                    #print('not too near')
                    break

            obs_single.append(obs)
        obs_single = np.array(obs_single)
        obs_list.append(obs_single)


    obs_list = np.array(obs_list)
    # convert from obs to point cloud
    obc_list = rectangle_pcd(obs_list, width, N_pc)
    return obs_list, obc_list
