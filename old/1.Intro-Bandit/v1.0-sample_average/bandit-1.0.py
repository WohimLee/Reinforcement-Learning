
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    def __init__(self, k_arm=10, epsilon=0., sample_average=False):
        self.k = k_arm
        self.epsilon = epsilon
        self.sample_average = sample_average
        # 每个 Action 的索引
        self.indices = np.arange(self.k)
 

    def reset(self):
        '''
        功能：用于每个 Run 重置数据
        '''
        # 每一个 Run 重置 q_expectation
        self.q_expectation = np.random.randn(self.k)
        # Reward 期望值最高的 Action 的索引
        self.best_action = np.argmax(self.q_expectation)
        # 每个 Action 的估计值存储空间，shape=(10,)
        self.q_estimation = np.zeros(self.k)
        # 每个 Action 被选择的次数
        self.action_count = np.zeros(self.k)

    def act(self):
        '''
        功能：获取一个 Action
        '''
        # epsilon*100% 的概率尝试新的 Action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        # 否则就选估计 Reward 最大的 Action
        q_best = np.max(self.q_estimation)
        # 如果最大的 Reward 值有好几个，随机选一个
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        '''
        功能：执行一个 Action，并且更新这个 Action 的 Reward
        '''
        # 因为 reward 不是确定的，而是有一定概率分布
        # 所以这里模拟老虎机实际出来的 reward
        reward = np.random.randn() + self.q_expectation[action]
        self.action_count[action] += 1

        # sample average 的更新公式
        if self.sample_average:
            self.q_estimation[action] += (reward - self.q_estimation[action]) \
                                        / self.action_count[action]
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
    # 每个不同 epsilon 的 Bandit 中，每个 time
    # 在 2000 个 run 中采取 best Action 的概率
    best_action_rates = best_action_counts.mean(axis=1)
    # 每个不同 epsilon 的 Bandit 中，每个 time
    # 2000 个 run 的平均 Reward
    mean_rewards = rewards.mean(axis=1)
    return best_action_rates, mean_rewards

def algorithm(runs=2000, time=1000):
    # 贪心算法、0.1-贪心算法、0.01-贪心算法
    epsilons = [0, 0.1, 0.01]
    bandits = [Bandit(epsilon=eps) for eps in epsilons]
    best_action_rates, mean_rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, mean_rewards):
        plt.plot(rewards, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('time')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, rate in zip(epsilons, best_action_rates):
        plt.plot(rate*100, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('time')
    plt.ylabel('Optimal Action (%)')
    plt.legend()

    plt.savefig('../imgs/bandit-1.0.png')
    plt.close()
    
    
if __name__ == '__main__':
    algorithm()
