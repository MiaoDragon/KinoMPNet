import sys
sys.path.append('..')
from plan_utility.line_line_cc import line_line_cc
import numpy as np
def start_goal_gen(low, high, width, obs, obs_recs):
    # using obs information and bound, to generate good start and goal
    width = 4.
    H = 0.5
    L = 2.5


    while True:
        start = np.random.uniform(low=low, high=high)
        # set the position to be within smaller bound
        ratio = .8
        start[0] = start[0] * ratio
        start_x0 = start[0]
        start_y0 = H
        start_x1 = start[0] + L * np.sin(start[2])
        start_y1 = H + L * np.cos(start[2])
        cf = True
        for i in range(len(obs_recs)):
            for j in range(4):
                if line_line_cc(start_x0, start_y0, start_x1, start_y1, obs_recs[i][j][0], obs_recs[i][j][1],
                                obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                    cf = False
                    break
        while True:
            end = np.random.uniform(low=low, high=high)
            end[0] = end[0] * ratio
            end_x0 = end[0]
            end_y0 = H
            end_x1 = end[0] + L * np.sin(end[2])
            end_y1 = H + L * np.cos(end[2])
            cf = True
            for i in range(len(obs_recs)):
                for j in range(4):
                    if line_line_cc(end_x0, end_y0, end_x1, end_y1, obs_recs[i][j][0], obs_recs[i][j][1],
                                    obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                        cf = False
                        break
            if cf:
                break
        if np.abs(start[0] - end[0]) >= width*6:
            break
    start[1] = 0.
    start[3] = 0.
    return start, end
