from day_5.bayes_tools import *
import numpy as np

# Shutdown version:
# On the first timestep, it's [0, 0.5, 0.5]
# On the second timestep, it's
step_2 = 0.5 * np.asarray([0, 0.8, 0.2]) + 0.5 * np.asarray([0, 0.3, 0.7])
print(step_2) # [0. 0.55 0.45]
# Step 3 is done the same way
step_3 = step_2[1] * np.asarray([0, 0.8, 0.2]) + step_2[2] * np.asarray([0, 0.3, 0.7])
print(step_3) # [0. 0.575 0.425]
# and step 4! (man I wish this were generalized...)
step_4 = step_3[1] * np.asarray([0, 0.8, 0.2]) + step_3[2] * np.asarray([0, 0.3, 0.7])
print(step_4) # [0. 0.5875 0.4125]
# looks like it's asymptotically approaching a final distribution!

# Bayes filter and smooth: 

if __name__ == "__main__":
    # All matrices are true state in rows and output in columns
    # sensor  = np.asarray([[0.6, 0.2, 0.2],[0.2, 0.6, 0.2],[0.2, 0.2, 0.6]])  # measurement model
    # transition = np.asarray([[0.1, 0.4, 0.5],[0.4, 0, 0.6],[0, 0.6, 0.4]])  # transition model
    sensor  = np.asarray([[0, 0.5, 0.5],[0, 0.9, 0.1],[0, 0.1, 0.9]])  # measurement model
    transition = np.asarray([[0, 0.5, 0.5],[0, 0.8, 0.2],[0, 0.3, 0.7]])  # transition model
    
    # observations = [0, 2, 2]
    observations =  [1, 2, 2, 1, 2]
    initial_state = np.asarray((1, 0, 0))

    fwd_belief, fwd_raw = bayes_filter(initial_state, observations, sensor, transition)

    print("Forward Steps:")
    print(np.round(fwd_raw.T, 3))

    # Forward Steps:
    # [[0.5   0.    0.   ]
    # [0.    0.05  0.45 ]
    # [0.    0.035 0.585]
    # [0.    0.295 0.067]
    # [0.    0.071 0.263]]

    print("Filter Beliefs:")
    print(np.round(fwd_belief.T, 3))

    # Filter Beliefs:
    # [[1.    0.    0.   ]
    # [0.    0.1   0.9  ]
    # [0.    0.056 0.944]
    # [0.    0.815 0.185]
    # [0.    0.212 0.788]]

    smooth_belief = bayes_smooth(initial_state, observations, sensor, transition)

    print("Smooth Beliefs:")
    print(np.round(smooth_belief.T, 3))

    # Smooth Beliefs:
    # [[1.    0.    0.   ]
    # [0.    0.049 0.951]
    # [0.    0.093 0.907]
    # [0.    0.634 0.366]
    # [0.    0.212 0.788]]