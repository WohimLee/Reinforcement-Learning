
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import trange

def data_gen(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
            best_action_rates = best_action_counts.mean(axis=1)
            mean_rewards = rewards.mean(axis=1)
            yield time, best_action_rates, mean_rewards


# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2,1)

# intialize two line objects (one in each axes)
line1_1, = ax1.plot([], [], lw=2)
line1_2, = ax1.plot([], [], lw=2)
line1_3, = ax1.plot([], [], lw=2)

line2_1, = ax2.plot([], [], lw=2, color='r')
line2_2, = ax2.plot([], [], lw=2, color='r')
line2_3, = ax2.plot([], [], lw=2, color='r')

line = [line1_1, line1_2, line1_3,
        line2_1, line2_2, line2_3]

# # the same axes initalizations as before (just now we do it for both of them)
# for ax in [ax1, ax2]:
#     ax.set_ylim(-1.1, 1.1)
#     ax.set_xlim(0, 5)
#     ax.grid()

# initialize the data arrays 
xdata, y1data, y2data = [], [], []
def run(data):
    # update the data
    t, best_action_rates, mean_rewards = data

    y1_1 = mean_rewards[0]
    y1_2 = mean_rewards[1]
    y1_3 = mean_rewards[2]
    
    y2_1 = best_action_rates[0]
    y2_2 = best_action_rates[0]
    y2_3 = best_action_rates[0]
    
    
    xdata.append(t)
    y1data.append(y1)
    y2data.append(y2)

    # axis limits checking. Same as before, just for both axes
    for ax in [ax1, ax2]:
        xmin, xmax = ax.get_xlim()
        if t >= xmax:
            ax.set_xlim(xmin, 2*xmax)
            ax.figure.canvas.draw()

    # update the data of both line objects
    line[0].set_data(xdata, y1data)
    line[1].set_data(xdata, y2data)

    return line

ani = animation.FuncAnimation(fig, run, data_gen, blit=True, interval=10,
    repeat=False)
plt.show()



if __name__ == "__main__":
    pass