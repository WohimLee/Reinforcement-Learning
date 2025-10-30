
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(88)

def distribution_expectation():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10), showmeans=True)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('../imgs/action-reward_distribution.png')
    plt.close()
    
distribution_expectation()





