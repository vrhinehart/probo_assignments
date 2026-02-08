from day_5.bayes_tools import *
import numpy as np


# Prior - 0.015 faulty
# States - faulting (1) or not faulting (0)
# Reading under 1 counts as faulting, over counts as not faulting

if __name__ == "__main__":
    sensor = np.asarray([[2/3, 1/3],
                        [0, 1]])
    transition = np.asarray([[1, 0],
                             [0, 1]]) # assuming the sensor cannot change true state; it's either faulting or it isn't.
    initial_state = np.asarray((1-0.015, 0.015))
    observations = [1] * 10

    fwd_belief, fwd_raw = bayes_filter(initial_state, observations, sensor, transition)

    print("Forward Steps:")
    print(np.round(fwd_raw.T, 3))

    print("Filter Beliefs:")
    print(np.round(fwd_belief.T, 3))

# Forward Steps:
# [[0.328 0.015]
#  [0.328 0.015]
#  [0.319 0.044]
#  [0.293 0.121]
#  [0.236 0.291]
#  [0.149 0.552]
#  [0.071 0.787]
#  [0.028 0.917]
#  [0.01  0.971]
#  [0.003 0.99 ]]
# Filter Beliefs:
# [[0.985 0.015]
#  [0.956 0.044]
#  [0.879 0.121]
#  [0.709 0.291]
#  [0.448 0.552]
#  [0.213 0.787]
#  [0.083 0.917]
#  [0.029 0.971]
#  [0.01  0.99 ]
#  [0.003 0.997]]