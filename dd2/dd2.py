import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

moves = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1))) #x, y
world_size = 10
reward = 2 # for visiting a new space
cost = -1 # for revisiting a space
penalty = -1.2 # for going out of bounds
rewards = np.ones((world_size,world_size)) * reward
rewards_earned = np.zeros_like(rewards)

#rng = np.random.default_rng()

n_steps = 100
pos = np.empty((n_steps+1, 2))
pos[0] = np.asarray((0,0))

attempts_per_move = 10000
horizon = 100

def add_to_average(old_average, n, new_value):
    return (old_average * n + new_value) / (n + 1)

@njit(parallel=True)
def parallel_search(moves, attempts_per_move, current_pos, horizon, world_size, rewards, penalty):
    n_moves = len(moves)
    all_results = np.zeros(n_moves * attempts_per_move)

    for attempt in prange(len(all_results)):
        first_move = attempt // attempts_per_move
        plan_pos = current_pos + moves[first_move]
        for plan_step in range(horizon):
            next_move = np.random.randint(0,n_moves)
            plan_pos = plan_pos + moves[next_move]
        if (plan_pos[0] >= 0 and plan_pos[0] < world_size and
            plan_pos[1] >= 0 and plan_pos[1] < world_size):
            all_results[attempt] = rewards[int(plan_pos[0]), int(plan_pos[1])]
        else:
            all_results[attempt] = penalty

    return all_results

for robot_step in range(n_steps):
    # move_averages = np.zeros(len(moves))
    # for move in range(len(moves)):
    #     for attempt in range(attempts_per_move):
    #         plan_pos = pos[robot_step] + moves[move] # make first move
    #         for plan_step in range(horizon): # random rollout
    #             next_move = rng.choice(moves)
    #             plan_pos = plan_pos + next_move
    #         if np.all((0 <= plan_pos) & (plan_pos < world_size)):
    #             reward_seen = rewards[tuple(plan_pos.astype(int))]    # get reward
    #         else:
    #             reward_seen = penalty # out of bounds
    #         move_averages[move] = add_to_average(move_averages[move],attempt,reward_seen)
    move_results = parallel_search(moves, attempts_per_move, pos[robot_step], horizon, world_size, rewards, penalty)
    move_averages = move_results.reshape(len(moves), attempts_per_move).mean(axis=1)
    best_move = np.argmax(move_averages)
    
    pos[robot_step + 1] = pos[robot_step] + moves[best_move] # make best move
    new_pos_tuple = tuple(pos[robot_step + 1].astype(int))
    reward_earned = rewards[new_pos_tuple]
    rewards_earned[new_pos_tuple] = reward_earned
    rewards[new_pos_tuple] = cost

    print(robot_step)

plt.figure(figsize=(8, 8))

# Create color map for grid cells
color_grid = np.ones((world_size, world_size, 3))  # RGB, default white
for i in range(world_size):
    for j in range(world_size):
        if rewards_earned[i, j] == reward:
            color_grid[j, i] = [.67, 1, .67]  # green
        elif rewards_earned[i, j] == cost:
            color_grid[j, i] = [1, 0.65, 0.65]  # red

plt.imshow(color_grid, origin='lower', extent=(0, world_size, 0, world_size))

# Plot trajectory - offset by 0.5 to center in grid squares
pos_centered = pos + 0.5
colors = plt.cm.rainbow(np.linspace(0, 1, len(pos_centered)))
plt.scatter(pos_centered[:, 0], pos_centered[:, 1], c=range(len(pos_centered)), cmap='rainbow', s=20)
for i in range(len(pos_centered) - 1):
    plt.plot(pos_centered[i:i+2, 0], pos_centered[i:i+2, 1], color=colors[i], alpha=0.3, linewidth=1)

plt.xlim(0, world_size)
plt.ylim(0, world_size)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Robot Trajectory')
plt.grid(True)
plt.show()