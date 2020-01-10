import sys
sys.path.append('..')
import numpy as np
def start_goal_gen(low, high, width, obs, obs_recs):
    # using obs information and bound, to generate good start and goal
    start = np.random.uniform(low=low, high=high)
    end = np.random.uniform(low=low, high=high)
    # set the position to be within smaller bound
    ratio = .8
    start[0] = start[0] * ratio
    end[0] = end[0] * ratio
    return start, end
