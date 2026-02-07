from day_5.door_bot import bayes_filter_step, bayes_reverse_step
import numpy as np

"""
Prior - 0.015 faulty
States - faulting (1) or not faulting (0)
Reading under 1 counts as faulting, over counts as not faulting
"""

if __name__ == "__main__":
    sensor = np.asarray([[2/3, 1/3],
                        [0, 1]])
    transition = np.asarray([[1, 0],
                             [0, 1]]) # assuming the sensor cannot change true state; it's either faulting or it isn't.
    initial_state = np.asarray((1-0.015, 0.015))
    observations = [1] * 10

    timesteps = len(observations)
    fwd_belief = np.empty((2, timesteps)) # rows per state and cols per timestep.
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