import numpy as np
import sys
import multiprocessing as mp

import numpy.random as npr

def state_transition_func(xt, time_t):
    time_t += 1
    return 0.5 * xt + 25 * xt / (1 + xt ** 2) + 8 * np.cos(1.2 * time_t)


def transfer_func(x):
    return x ** 2 / 20


def generate_data(f=state_transition_func, g=transfer_func, q=1, r=1, x0=0, seq_len=100):
    x = np.zeros(seq_len)
    y = np.zeros(seq_len)
    q_sqrt = np.sqrt(q)
    r_sqrt = np.sqrt(r)
    x[0] = x0  # deterministic initial state
    y[0] = g(x[0]) + r_sqrt * np.random.normal(size=1)

    for t in range(1, seq_len):
        x[t] = f(x[t - 1], t - 1) + q_sqrt * np.random.normal(size=1)
        y[t] = g(x[t]) + r_sqrt * np.random.normal(size=1)

    return x, y


def multinomial_resampling(w, size=0):
    if size == 0:
        n = len(w)
    else:
        n = size

    ls = np.random.choice(len(w), size=n, replace=True, p=w)
    return ls
