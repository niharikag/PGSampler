######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

# Implements blocked SMC/ blocked PG

from rust_library import run_blocked_smc
import numpy as np
import sys
import multiprocessing as mp
import numpy.random as npr
from experiments.utils import transfer_func, state_transition_func, generate_data


def blocked_pg_sample(q, r, y, x0, x_ref, block_size=4, overlap=2, n_particles=100, max_iter=1000):
    if overlap >= block_size / 2:
        sys.exit("Error: Overlap cannot exceed block size")
    seq_len = len(y)
    startID = np.arange(0, seq_len, block_size - overlap)

    if startID[-1] == seq_len - 1:
        numBlocks = len(startID) - 1
    else:
        numBlocks = len(startID)

    print(numBlocks)

    x_new_ref = np.zeros(seq_len)
    pool = mp.Pool(processes=numBlocks)

    for n_iter in range(1, max_iter):
        inputs = []
        for i in range(numBlocks):
            s = startID[i]
            u = min(s + block_size - 1, seq_len - 1)
            if i + 1 == numBlocks:  # last block
                pos_block = -1
                x_init = x_ref[s - 1]
                x_last = 0
            elif i > 0:  # Intermediate block
                pos_block = 1
                x_init = x_ref[s - 1]
                x_last = x_ref[u + 1]
            else:  # First block
                x_init = x0
                pos_block = 0
                x_last = x_ref[u + 1]

            inputs.append((q, r, y[s:u + 1].copy(), x_init, x_ref[s:u + 1].copy(), pos_block, x_last, s, n_particles))

        results = pool.starmap(run_blocked_smc, inputs)

        for n in range(numBlocks):
            x_new_ref[startID[n]:(block_size + startID[n])] = results[n]

        x_ref = x_new_ref.copy()

    return x_new_ref


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    np.random.seed(123)
    # Set up some parameters
    n_particles = 100  # Number of particles
    seq_len = 100  # Length of data record
    f1 = state_transition_func
    g1 = transfer_func
    r = 1.
    q = 0.1
    x_ref = np.zeros(seq_len)
    x0 = 0.0
    # Generate data`
    x, y = generate_data(r=1.0, q=0.1, seq_len=seq_len)

    block_size = 30
    overlap = 1

    # Initialize states by running a CSMC
    x_ref = run_blocked_smc(q, r, y, x0, x_ref, 0, 0, 0, n_particles)

    # call blocked PG: runs blocked SMC in parallel
    s_time = time.time()
    x_pg = blocked_pg_sample(q, r, y, x0=0, x_ref=x_ref, block_size=block_size, overlap=overlap, 
                             n_particles=n_particles, max_iter=10000)
    print("time taken by iterate_blocked_pg: ", time.time() - s_time)
    plt.figure(figsize=(10, 4))
    plt.plot(x, marker='o', label='Real States', markersize=3)
    plt.plot(x_pg, label='Blocked PG States')
    plt.legend()
    plt.savefig("../plots/blocked_PG.png")
    plt.show()
