import sys
sys.path.append('..')
from plan_utility.line_line_cc import line_line_cc
import numpy as np
def start_goal_gen(low, high, width, obs, obs_recs):
    # using obs information and bound, to generate good start and goal
    LENGTH = 20.
    near = width * 1.2
    s_g_dis_threshold = LENGTH * 1.6
    start = np.zeros(4)  # here we fix the start
    end = np.zeros(4)
    """
    while True:
        start[0] = np.random.uniform(low=low[0], high=high[0])
        # make sure start midpoint
        x0_start = 0.
        y0_start = 0.
        x1_start = LENGTH * np.cos(start[0] - np.pi/2)
        y1_start = LENGTH * np.sin(start[0] - np.pi/2)
        #print('x1 = %f, y1 = %f' % (x1_start, y1_start))
        # make sure (x0, y0) to (x1, y1) is collision free
        cf = True
        for i in range(len(obs_recs)):
            for j in range(4):
                if line_line_cc(x0_start, y0_start, x1_start, y1_start, obs_recs[i][j][0], obs_recs[i][j][1],
                                obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                    cf = False
                    break
        if cf:
            break
    #print('generated start point')

    while True:
        start[1] = np.random.uniform(low=low[1], high=high[1])
        # make sure start and end not in collision
        x2_start = LENGTH * np.cos(start[0] + start[1] - np.pi/2) + x1_start
        y2_start = LENGTH * np.sin(start[0] + start[1] - np.pi/2) + y1_start
        # make sure (x0, y0) to (x1, y1) is collision free
        cf = True
        for i in range(len(obs_recs)):
            for j in range(4):
                if line_line_cc(x1_start, y1_start, x2_start, y2_start, obs_recs[i][j][0], obs_recs[i][j][1],
                                obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                    cf = False
                    break
        if cf:
            break
    """
    x0_start = 0.
    y0_start = 0.
    x1_start = LENGTH * np.cos(start[0] - np.pi/2)
    y1_start = LENGTH * np.sin(start[0] - np.pi/2)
    x2_start = LENGTH * np.cos(start[0] + start[1] - np.pi/2) + x1_start
    y2_start = LENGTH * np.sin(start[0] + start[1] - np.pi/2) + y1_start

    # start endpoint
    while True:
        #print('generated endpoint')

        end[0] = np.random.uniform(low=low[0], high=high[0])

        # make sure start midpoint
        x0_goal = 0.
        y0_goal = 0.
        x1_goal = LENGTH * np.cos(end[0] - np.pi/2)
        y1_goal = LENGTH * np.sin(end[0] - np.pi/2)
        # start endpoint
        end[1] = np.random.uniform(low=low[1], high=high[1])
        # make sure start and end not in collision
        x2_goal = LENGTH * np.cos(end[0] + end[1] - np.pi/2) + x1_goal
        y2_goal = LENGTH * np.sin(end[0] + end[1] - np.pi/2) + y1_goal
        # make sure y2 is always positive, if not, then the mirrow image will be positive
        if y2_goal < 0.:
            end[0] = np.pi - end[0]
            end[1] = -end[1]
            x1_goal = LENGTH * np.cos(end[0] - np.pi/2)
            y1_goal = LENGTH * np.sin(end[0] - np.pi/2)
            x2_goal = LENGTH * np.cos(end[0] + end[1] - np.pi/2) + x1_goal
            y2_goal = LENGTH * np.sin(end[0] + end[1] - np.pi/2) + y1_goal
        # check if y2_goal is high enough
        if y2_goal <= LENGTH:
            continue
        # make sure the acrobot is collision free
        cf = True
        for i in range(len(obs_recs)):
            for j in range(4):
                if line_line_cc(x0_goal, y0_goal, x1_goal, y1_goal, obs_recs[i][j][0], obs_recs[i][j][1],
                                obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                    cf = False
                    break
                if line_line_cc(x1_goal, y1_goal, x2_goal, y2_goal, obs_recs[i][j][0], obs_recs[i][j][1],
                                obs_recs[i][(j+1)%4][0], obs_recs[i][(j+1)%4][1]):
                    cf = False
                    break
            if not cf:
                break
        # need to be in different phase, i.e. at least one sign need to be different
        if np.linalg.norm(np.array([x2_goal,y2_goal])-np.array([x2_start,y2_start])) < s_g_dis_threshold:
            continue
        # add endpoint length constraint
        if cf:
            break

        #start_sign = np.array([x2_start, y2_start]) >= 0
        #end_sign = np.array([x2_goal, y2_goal]) >= 0
        #if start_sign[0] != end_sign[0] or start_sign[1] != end_sign[1]:
        #    #print('sg generated.')
        #    break

    print('feasible sg generated.')
    start[2] = 0.
    start[3] = 0.
    end[2] = 0.
    end[3] = 0.
    return start, end
