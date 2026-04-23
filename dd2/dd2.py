import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange

moves = np.asarray(((1, 0), (-1, 0), (0, 1), (0, -1))) #x, y
world_size = 20
reward = 10 # for visiting a new space
adjacency_reward = 0 # additional reward for adjacent new spaces
cost = -1 # for revisiting a space
penalty = 0 # for hitting the boundary
discount = 0.9
rewards = np.ones((world_size,world_size)) * reward
rewards_earned = np.zeros_like(rewards)

#rng = np.random.default_rng()

n_steps = 400
pos = np.empty((n_steps+1, 2))
pos[0] = np.asarray((0,0))

attempts_per_move = 1000
horizon = 1000

def add_to_average(old_average, n, new_value):
    return (old_average * n + new_value) / (n + 1)

@njit(parallel=True)
def parallel_search(moves, attempts_per_move, current_pos, horizon, world_size, rewards, penalty):
    n_moves = len(moves)
    all_results = np.zeros(n_moves * attempts_per_move)

    for attempt in prange(len(all_results)):
        first_move = attempt // attempts_per_move
        plan_pos = current_pos + moves[first_move]
        if (plan_pos[0] >= 0 and plan_pos[0] < world_size and
            plan_pos[1] >= 0 and plan_pos[1] < world_size):
            all_results[attempt] += rewards[int(plan_pos[0]), int(plan_pos[1])]
        else:
            all_results[attempt] = float('-inf')
            continue
        
        for plan_step in range(horizon):
            next_move = np.random.randint(0,n_moves)
            next_pos = plan_pos + moves[next_move]
            if (next_pos[0] >= 0 and next_pos[0] < world_size and
                next_pos[1] >= 0 and next_pos[1] < world_size):
                plan_pos = next_pos
                all_results[attempt] += rewards[int(plan_pos[0]), int(plan_pos[1])] * (discount ** plan_step)
            else:
                all_results[attempt] += penalty * (discount ** plan_step)

    return all_results

for robot_step in range(n_steps):
    move_results = parallel_search(moves, attempts_per_move, pos[robot_step], horizon, world_size, rewards, penalty)
    move_averages = move_results.reshape(len(moves), attempts_per_move).mean(axis=1)
    best_move = np.argmax(move_averages)
    
    pos[robot_step + 1] = pos[robot_step] + moves[best_move] # make best move
    new_pos_tuple = tuple(pos[robot_step + 1].astype(int))
    reward_earned = rewards[new_pos_tuple]
    rewards_earned[new_pos_tuple] = reward_earned
    rewards[new_pos_tuple] = cost
    
    for move in range(len(moves)):
        next_pos = pos[robot_step + 1] + moves[move]
        next_pos_tuple = tuple(next_pos.astype(int))
        try:
            if rewards[next_pos_tuple] >= reward:
                rewards[next_pos_tuple] += adjacency_reward
        except IndexError:
            pass

    print(robot_step)

np.savetxt('pos.csv', pos.astype(int), delimiter=',', fmt='%d', header='X,Y', comments='')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: trajectory
color_grid = np.ones((world_size, world_size, 3))  # RGB, default white
for i in range(world_size):
    for j in range(world_size):
        if rewards_earned[i, j] >= reward:
            color_grid[j, i] = [.67, 1, .67]  # green
        elif rewards_earned[i, j] == cost:
            color_grid[j, i] = [1, 0.65, 0.65]  # red

ax1.imshow(color_grid, origin='lower', extent=(0, world_size, 0, world_size))

pos_centered = pos + 0.5
colors = plt.cm.rainbow(np.linspace(0, 1, len(pos_centered)))

# Black outline for line
for i in range(len(pos_centered) - 1):
    ax1.plot(pos_centered[i:i+2, 0], pos_centered[i:i+2, 1], color='black', alpha=0.8, linewidth=1.5)

# Black outline for markers
ax1.scatter(pos_centered[:, 0], pos_centered[:, 1], c='black', s=30, zorder=2)

# Colored markers and line on top
ax1.scatter(pos_centered[:, 0], pos_centered[:, 1], c=range(len(pos_centered)), cmap='rainbow', s=20, zorder=3)
for i in range(len(pos_centered) - 1):
    ax1.plot(pos_centered[i:i+2, 0], pos_centered[i:i+2, 1], color=colors[i], alpha=0.3, linewidth=1)

ax1.set_xlim(0, world_size)
ax1.set_ylim(0, world_size)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Robot Trajectory')
ax1.grid(True)

# Right plot: rewards array
im = ax2.imshow(rewards.T, origin='lower', cmap='RdYlGn', extent=(0, world_size, 0, world_size), vmin=-5, vmax=5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Rewards Grid')
ax2.grid(True)
plt.colorbar(im, ax=ax2)

plt.tight_layout()
plt.show()