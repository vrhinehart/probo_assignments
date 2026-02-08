import numpy as np
from day_5.bayes_tools import *

# Part A:
# We can approach this simply because the door in this system never acts spontaneously, and opening always has a 4/5 chance of working.
# For the door to remain closed, you need the 0.5 that it starts closed, then the 1/5 ^ 3 that it fails to open three times in a row.
# This probability is 0.004, so the door has an 0.996 probability of being open at the end.

if __name__ == "__main__":
    # All matrices are true state in rows and output in columns
    sensor = np.asarray([[4/5, 1/5],
                        [2/5, 3/5]]) 
    actuator_0 = np.asarray([[1, 0], 
                            [0, 1]])
    actuator_1 = np.asarray([[1/5, 4/5], 
                            [0, 1]])
    
    observations =  [0, 0, 1, 0, 1]
    actions =       [1, 0, 0, 1, 1]
    initial_state = [0.5, 0.5]
    fwd_belief, fwd_raw = bayes_filter(initial_state, observations, sensor, [actuator_0, actuator_1], actions)

    print("Filter Beliefs:")
    print(np.round(fwd_belief[:,1:], 3))

    smooth_belief = bayes_smooth(initial_state, observations, sensor, [actuator_0, actuator_1], actions)

    print("Smooth Beliefs:")
    print(np.round(smooth_belief, 3))

    # Filter Beliefs:
    # [[0.667 0.4   0.148 0.01 ]
    # [0.333 0.6   0.852 0.99 ]]
    # Smooth Beliefs:
    # [[0.433 0.433 0.433 0.131 0.01 ]
    # [0.567 0.567 0.567 0.869 0.99 ]]



