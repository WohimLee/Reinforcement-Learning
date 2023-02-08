
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1):
        self.k = k_arm
        self.epsilon = epsilon
        self.initial = initial
        self.step_size = step_size
        self.time = 0
        self.indices = np.arange(self.k)
        

    def reset(self):
        self.q_true = np.random.randn(self.k)
        self.best_action = np.argmax(self.q_true)
        # 这里初始值改成了 initial
        self.q_estimation = np.zeros(self.k) + self.initial
        self.action_count = np.zeros(self.k)
        self.time = 0

    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        else:
            q_best = np.max(self.q_estimation)
            return np.random.choice(np.where(self.q_estimation == q_best)[0])

    def step(self, action):
        # 每次 Action 实际的 Reward
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1

        # 这里改成了乘以 step_size
        self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits):
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
    return best_action_rates, mean_rewards


def algorithm(runs=2000, time=1000):
    bandits = []
    bandits.append(Bandit(epsilon=0, initial=5, step_size=0.1))
    bandits.append(Bandit(epsilon=0.1, initial=0, step_size=0.1))
    best_action_rates, _ = simulate(runs, time, bandits)

    plt.plot(best_action_rates[0]*100, label='$\epsilon = 0, q = 5$')
    plt.plot(best_action_rates[1]*100, label='$\epsilon = 0.1, q = 0$')
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action (%)')
    plt.legend()

    plt.savefig('imgs/bandit-2.0.png')
    plt.close()

if __name__ == '__main__':
    algorithm()
