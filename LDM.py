import numpy as np
import matplotlib.pyplot as plt
from gridworld import *

def define_G(state, density, gamma):
    nrow, ncol, _ = state.shape
    num_s = nrow*ncol
    num_a = 4
    G = density.copy()
    while True:
        G_p = G.copy()
        for s_t in range(num_s):
            for a_t in range(num_a):
                state = set_state(state, s_t)
                next_state = makeMove(state, a_t)
                reward = getReward(next_state)
                next_state_val = state_to_val(next_state)
                if reward == -30 or reward == 30:
                    G_p[s_t, a_t] = density[s_t, a_t]
                else:
                    G_p[s_t, a_t] = max(density[s_t, a_t], gamma * min(G[next_state_val, a_tt] for a_tt in range(num_a)))
        if np.all(G == G_p):
            plt.imshow(G)
            plt.title('G')
            plt.colorbar()
            plt.savefig('G')
            plt.clf()
            return G
        else:
            G = G_p
    

def compute_density(dataset, nrow, ncol, action_size):
    state_size = nrow*ncol
    n = len(dataset.memory)
    density = np.zeros((state_size, action_size)) + 0.00000001
    for transition in dataset.memory:
        state = transition.state.reshape((nrow, ncol, 4))
        i, j = getLoc(state.reshape((nrow, ncol, 4)), 3)
        state_val = i*4+j
        action = transition.action
        density[state_val, action] += 1/n
    density = -np.log(density)
    return density

def get_c_from_threshold(density, dataset, k):
    nrow, ncol = density.shape
    densities = []
    for i in range(nrow):
        for j in range(ncol):
            densities.append(density[i, j])
    densities = np.array(densities)
    sorted_idx = np.argsort(densities)
    idx = max(0, np.rint(len(densities)*k).astype(int) - 1)
    neg_log_c = densities[sorted_idx[idx]]
    c = np.exp(-neg_log_c)
    return c
    