import time
import random
import numpy as np

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
DISCOUNT = 0.9

# left, up, right, down
ACTIONS = [
    np.array([0, -1]), 
    np.array([-1, 0]), 
    np.array([0, 1]), 
    np.array([1, 0])
]

NUM_ACTIONS = len(ACTIONS)

ACTIONS_FIGS=[ '←', '↑', '→', '↓']

ACTION_PROB = 0.25


def print_before_step(state, action_idx):
    if state == A_POS:
        print(f"Current State: A, Action: {ACTIONS_FIGS[action_idx]}")
    elif state == B_POS:
        print(f"Current State: B, Action: {ACTIONS_FIGS[action_idx]}")
    else:
        print(f"Current State: {state}, Action: {ACTIONS_FIGS[action_idx]}")
            

def print_after_step(state, reward):
    if state == A_PRIME_POS:
        print(f"Jump to State: A', Reward: {reward}")
    elif state == B_PRIME_POS:
        print(f"Jump to State: B', Reward: {reward}")
    else:
        print(f"Jump to State: {state}, Reward: {reward}")


def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5
    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or y < 0 or x >WORLD_SIZE or y > WORLD_SIZE:
        reward = -1.
        return state, reward
    else:
        reward = 0
        return next_state, reward
    
def algorithm():
    state = [0,0]
    cumulative_reward = 0.
    i=0
    while True:
        i+=1
        action_idx = np.random.randint(NUM_ACTIONS)
        action = ACTIONS[action_idx]
        print_before_step(state, action_idx)
        
        state, reward = step(state, action)
        print_after_step(state, reward)
        
        # cumulative_reward += reward
        discount_rate = pow(DISCOUNT, i)
        cumulative_reward += reward*discount_rate
        print(f"Cumulative Reward = {reward} * {discount_rate:.2f}",
                               f" = {cumulative_reward:.2f}\n")
        
        time.sleep(2)

        
if __name__ == '__main__':
    algorithm()
