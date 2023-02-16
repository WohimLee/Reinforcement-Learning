
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

matplotlib.use('Agg')

WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]
gamma = 0.9

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]

ACTION_PROB = 0.25


def step(state, action):
    # A 的特殊情况
    if state == A_POS:
        return A_PRIME_POS, 10
    # B 的特殊情况
    if state == B_POS:
        return B_PRIME_POS, 5

    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows

    # Add cells
    for (i, j), val in np.ndenumerate(image):

        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"
        
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
        

    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def algorithm():
    I = np.eye(WORLD_SIZE*WORLD_SIZE)
    P = np.zeros(I.shape)
    r = np.zeros(WORLD_SIZE*WORLD_SIZE)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            # state_curr: 当前时刻 State 在网格上的坐标
            state_curr = [i, j]
            # 算出 state_curr 行排列的索引
            state_curr_idx = np.ravel_multi_index(state_curr, (WORLD_SIZE, WORLD_SIZE))
            for action in ACTIONS:
                # state_next: 下一时刻 State 在网格上的坐标
                state_next, reward = step(state_curr, action)
                # 算出 state_next 行排列的索引
                state_next_idx = np.ravel_multi_index(state_next, (WORLD_SIZE, WORLD_SIZE))
                # 每个 Action 换到 "下一个State"，P 加 0.25
                P[state_curr_idx, state_next_idx] += ACTION_PROB
                # 算出 r(s)，把每个 Action 加和
                r[state_curr_idx] += ACTION_PROB*reward
    value = np.linalg.solve(I-gamma*P, r)
    draw_image(np.round(value.reshape(WORLD_SIZE, WORLD_SIZE), decimals=2))
    plt.savefig('./imgs/test.png')
    plt.close()


if __name__ == '__main__':
    algorithm()

    
