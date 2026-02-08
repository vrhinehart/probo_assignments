import numpy as np

def bayes_forward_step(last_belief, observation, sensor_model, transition_model):
    """
    Arguments
    last_belief: Vector of probability distribution from the previous step
    observation: State number that the sensor has observed
    sensor_model: Matrix with rows for truth and columns for observation
    transition_model: Matrix with rows for last states and columns for next states.
        In case of an actuator, pass the transition_model that corresponds to the action being taken this step.

    Returns
    next_belief: Vector of probability distribution for each possible state.
        Not normalized. To normalize, divide by its sum.
    """
    # n_states = len(last_belief)
    # states = range(n_states)
    # prior = np.zeros(n_states)
    # for next_state in states:
    #     for last_state in states:
    #         prior[next_state] += last_belief[last_state] * transition_model[last_state, next_state]
    prior = last_belief @ transition_model
    likelihood = sensor_model[:,observation] # likelihood of each true state given the observation
    next_belief = likelihood * prior
    return next_belief.copy()

def bayes_reverse_step(next_belief, next_observation, sensor_model, transition_model):
    """
    Arguments
    next_belief: Vector of probability distribution from the next step
    next_observation: State number that the sensor observes in the next step
    sensor_model: Matrix with rows for truth and columns for observation
    transition_model: Matrix with rows for last states and columns for next states.
        In case of an actuator, pass the transition_model that corresponds to the action taken in the NEXT step.

    Returns
    belief: Vector of probability distribution for each possible state.
        Not normalized.
    """
    belief = transition_model @ (next_belief * sensor_model[:, next_observation]) # sensor_model slice is a vector, so it gets elementwise-multiplied with the other vector before matrix multiplication
    return belief.copy()

def bayes_filter(initial_state, observations, sensor, transition_list, action_list = None):
    if not action_list:
        action_list = np.zeros(len(observations))
        if len(transition_list) != 1:
            # If I didn't put my single transition matrix in an iterable, put it in one.
            transition_list = [np.asarray(transition_list)]
    timesteps = len(observations)
    fwd_belief = np.empty((len(initial_state), timesteps)) # rows per state and cols per timestep.
    fwd_belief[:,0] = initial_state
    fwd_raw = np.empty_like(fwd_belief)
    fwd_raw[:,0] =  sensor[:,observations[0]] * initial_state
    for step in range(timesteps-1):
        action = int(action_list[step+1])
        transition = transition_list[action]
        posterior = bayes_forward_step(fwd_belief[:,step], observations[step+1], sensor, transition)
        fwd_raw[:,step+1] = posterior
        fwd_belief[:,step+1] = posterior / np.sum(posterior)
    return fwd_belief.copy(), fwd_raw.copy()

def bayes_reverse(observations, sensor, transition_list, action_list = None):
    if not action_list:
        action_list = np.zeros(len(observations))
        if len(transition_list) != 1:
            # If I didn't put my single transition matrix in an iterable, put it in one.
            transition_list = [np.asarray(transition_list)]
    timesteps = len(observations)
    rev_raw = np.empty((3, timesteps))
    rev_raw[:,-1] = (1, 1, 1) # starting point
    for step in range(timesteps-1, 0, -1): # for every timestep except zero
        action = int(action_list[step])
        transition = transition_list[action]
        rev_raw[:,step-1] = bayes_reverse_step(rev_raw[:,step], observations[step], sensor, transition)
    return rev_raw.copy()

def bayes_smooth(initial_state, observations, sensor, transition_list, action_list = None):
    fwd_belief, fwd_raw = bayes_filter(initial_state, observations, sensor, transition_list, action_list)
    rev_raw = bayes_reverse(observations, sensor, transition_list, action_list)
    timesteps = len(observations)
    smooth_belief = np.empty_like(rev_raw)
    for step in range(timesteps):
        alpha = fwd_belief[:, step]
        # alpha = fwd_raw[:,step] (we don't need this, it works fine with normalized versions)
        beta = rev_raw[:, step]
        smooth_belief[:,step] = alpha * beta / np.sum(alpha * beta)
    return smooth_belief
