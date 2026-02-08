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
    timesteps = len(observations)
    fwd_belief = np.empty((2, timesteps+1)) # rows per state and cols per timestep. Extra timestep at the start for the prior
    fwd_belief[:,0] = (0.5, 0.5) # 50/50 starting prior
    for step in range(timesteps):
        actuator = actuator_1 if actions[step] else actuator_0
        posterior = bayes_filter_step(fwd_belief[:,step], observations[step], sensor, actuator)
        fwd_belief[:,step+1] = posterior / np.sum(posterior)

    print("Filter Beliefs:")
    print(np.round(fwd_belief[:,1:], 3))

    rev_belief = np.empty((2, timesteps))
    rev_belief[:,-1] = (1, 1) # starting point
    for step in range(timesteps-1, 0, -1): # for every timestep except zero
        actuator = actuator_1 if actions[step] else actuator_0
        rev_belief[:,step-1] = bayes_reverse_step(rev_belief[:,step], observations[step], sensor, actuator)
    smooth_belief = np.empty_like(rev_belief)
    for step in range(timesteps):
        alpha = fwd_belief[:, step + 1] # get around the extra alpha prior at the start
        beta = rev_belief[:, step]
        smooth_belief[:,step] = alpha * beta / np.sum(alpha * beta)

    print("Smooth Beliefs:")
    print(np.round(smooth_belief, 3))


