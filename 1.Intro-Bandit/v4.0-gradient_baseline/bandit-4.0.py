
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    def __init__(self, k_arm=10, step_size=0.1,
                 gradient=False, gradient_baseline=False, true_reward=0.):
        self.k = k_arm
        self.step_size = step_size
        self.indices = np.arange(self.k)
        self.time = 0
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward


    def reset(self):
        # 改成了均值为 N(self.true_reward, 1) 的正态分布
        self.q_true = np.random.randn(self.k) + self.true_reward
        self.best_action = np.argmax(self.q_true)
        self.q_estimation = np.zeros(self.k) 
        self.action_count = np.zeros(self.k)
        self.time = 0

    def act(self):
        # 改成了 softmax
        exp_est = np.exp(self.q_estimation)
        self.action_prob = exp_est / np.sum(exp_est)
        return np.random.choice(self.indices, p=self.action_prob)

    def step(self, action):
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        one_hot = np.zeros(self.k)
        one_hot[action] = 1
        if self.gradient_baseline:
            baseline = self.average_reward
        else:
            baseline = 0
        self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)

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
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Bandit(gradient=True, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_rates, _ = simulate(runs, time, bandits)
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_rates[i]*100, label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action (%)')
    plt.legend()

    plt.savefig('./imgs/bandit-4.0.png')
    plt.close()



if __name__ == '__main__':
    algorithm()
