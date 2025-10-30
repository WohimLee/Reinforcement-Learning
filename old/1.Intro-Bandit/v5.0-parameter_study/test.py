
import numpy as np



class Bandit:
    def __init__(self, k=10, epsilon=0., initial=0., expect_reward=0.):
        self.k = k
        self.epsilon = epsilon
        self.initial = initial
        self.expect_reward = expect_reward
        self.indices = np.arange(self.k)
    
    def reset(self):
        self.q_expect = np.random.randn(self.k) + self.expect_reward
        self.best_action = np.argmax(self.q_expect)
        self.Q_estimation = np.zeros(self.k) + self.initial
        
        self.action_count = np.zeros(self.k)
        self.time = 0
    
    def act(self):
        
        Q_best = np.argmax(self.Q_estimation)
        return np.random.choice(np.where(self.Q_estimation == Q_best))
    
    def step(self, action):
        pass
    
def simulate(runs, time):
    pass



def algorithm(runs=2000, time=1000):
    algos = ['$\epsilon$-greedy']

    

if __name__ == '__main__':
    algorithm()