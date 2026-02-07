from day_5.door_bot import bayes_filter_step, bayes_reverse_step
import numpy as np

"""
Questions:
Why is there a col of zeroes in the sensor table?
Why do we apply the sensor probabilities to the initial conditions? 
Why do we base the next step on the non-normalized alpha instead of the normalized? 
Why does it seem to work fine to calculate the smoothed values using the already normalized filter values?
"""

if __name__ == "__main__":
    # All matrices are true state in rows and output in columns
    # sensor  = np.asarray([[0.6, 0.2, 0.2],[0.2, 0.6, 0.2],[0.2, 0.2, 0.6]])  # measurement model
    # transition = np.asarray([[0.1, 0.4, 0.5],[0.4, 0, 0.6],[0, 0.6, 0.4]])  # transition model
    sensor  = np.asarray([[0, 0.5, 0.5],[0, 0.9, 0.1],[0, 0.1, 0.9]])  # measurement model
    transition = np.asarray([[0, 0.5, 0.5],[0, 0.8, 0.2],[0, 0.3, 0.7]])  # transition model
    
    # observations = [0, 2, 2]
    observations =  [1, 2, 2, 1, 2]
    initial_state = np.asarray((1, 0, 0))

    timesteps = len(observations)
    fwd_belief = np.empty((3, timesteps)) # rows per state and cols per timestep.
    fwd_belief[:,0] = initial_state
    fwd_raw = np.empty_like(fwd_belief)
    fwd_raw[:,0] = initial_state @ sensor * initial_state
    print(fwd_raw[:,0])
    for step in range(timesteps-1):
        posterior = bayes_filter_step(fwd_raw[:,step], observations[step+1], sensor, transition)
        fwd_raw[:,step+1] = posterior
        fwd_belief[:,step+1] = posterior / np.sum(posterior)

    print("Forward Steps:")
    print(np.round(fwd_raw.T, 3))

    print("Filter Beliefs:")
    print(np.round(fwd_belief.T, 3))

    rev_raw = np.empty((3, timesteps))
    rev_raw[:,-1] = (1, 1, 1) # starting point
    for step in range(timesteps-1, 0, -1): # for every timestep except zero
        rev_raw[:,step-1] = bayes_reverse_step(rev_raw[:,step], observations[step], sensor, transition)
    smooth_belief = np.empty_like(rev_raw)
    for step in range(timesteps):
        # alpha = fwd_belief[:, step]
        alpha = fwd_raw[:,step]
        beta = rev_raw[:, step]
        smooth_belief[:,step] = alpha * beta / np.sum(alpha * beta)

    print("Smooth Beliefs:")
    print(np.round(smooth_belief.T, 3))