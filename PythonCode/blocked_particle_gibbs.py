######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

# Implements blocked SMC/ blocked PG

import numpy as np
import sys
import scipy.stats as stats
import multiprocessing as mp
from conditional_smc import CSMC
from utils import multinomial_resampling
from scipy.stats import norm


class BlockedSMC(object):
    def __init__(self, f, g, q, r):
        if (f is None) or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self.f = f
        self.g = g
        self.q = q
        self.r = r
        self.particles = None
        self.normalised_weights = None
        self.ancestors = None
        self.seq_len = 0
        self.n_particles = 100
        self.log_likelihood = 0.0
        self.resampling_method = multinomial_resampling

    def resample_step(self, t):
        new_ancestors = self.resampling_method(self.normalised_weights[:, t])
        self.ancestors[:, t] = new_ancestors.astype(int)

    def propagation_step(self, t):
        xpred = self.f(self.particles[:, t - 1], t - 1)
        new_ancestors = self.ancestors[:, t - 1]
        self.particles[:, t] = xpred[new_ancestors] + np.sqrt(self.q) * np.random.normal(size=self.n_particles)

    def weighting_step(self, y_t, t):
        ypred = self.g(self.particles[:, t])
        logweights = norm.logpdf(ypred, y_t, np.sqrt(self.r))
        max_weight = max(logweights)  # Subtract the maximum value for numerical stability
        w_p = np.exp(logweights - max_weight)
        self.normalised_weights[:, t] = w_p / sum(w_p)  # Save the normalized weights

        # accumulate the log-likelihood
        self.log_likelihood = self.log_likelihood + max_weight + np.log(sum(w_p)) - np.log(self.n_particles)

    def get_state_trajectory(self):
        if self.particles is None:
            print("call generateWeightedParticles first")
            exit(0)

        x_star = np.zeros(self.seq_len)
        indx = multinomial_resampling(self.normalised_weights[:, self.seq_len - 1], 1)[0]

        for t in reversed(range(self.seq_len)):
            indx = self.ancestors[indx, t]
            x_star[t] = self.particles[indx, t]
        return x_star

    def generate_particles(self, y, x0, x_ref, pos_block=0, x_last=None, start_time=0):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # Number of states
        self.seq_len = len(y)

        # initialize
        self.seq_len = len(y)
        self.log_likelihood = 0
        self.particles = np.zeros((self.n_particles, self.seq_len))
        self.normalised_weights = np.zeros((self.n_particles, self.seq_len))
        self.ancestors = np.zeros((self.n_particles, self.seq_len), dtype=int)

        if pos_block == 0:
            # Init state, at t=0
            self.particles[:, 0] = x0  # Deterministic initial condition
        else:
            self.particles[:, 0] = x0 + np.sqrt(self.q) * np.random.normal(size=self.n_particles)

        self.particles[self.n_particles - 1, 0] = x_ref[0]

        # weighting step
        self.weighting_step(y[0], 0)
        self.ancestors[:, 0] = np.arange(self.n_particles)

        for t in range(1, self.seq_len):
            # resampling step
            self.resample_step(t - 1)
            self.ancestors[self.n_particles - 1, t] = self.n_particles - 1

            # propagation step
            self.propagation_step(t)
            self.particles[self.n_particles - 1, t] = x_ref[t]

            # weighting step
            self.weighting_step(y[t], t)

        self.ancestors[:, self.seq_len - 1] = np.arange(self.n_particles)

        # Reweighting step
        if pos_block >= 0:
            # compute unnormalized weights
            un_weights = stats.norm.logpdf(self.g(self.particles[:, self.seq_len-1]), y[self.seq_len-1],
                                           np.sqrt(self.r))
            weights_1 = stats.norm.logpdf(self.f(self.particles[:, self.seq_len - 1], start_time+self.seq_len-1),
                                          x_last, np.sqrt(self.q)) + un_weights
            weights_1 = np.exp(weights_1 - np.max(weights_1))
            self.normalised_weights[:, self.seq_len-1] = weights_1 / sum(weights_1)  # Save the normalized weights

    def sample_states(self, y, x0, x_ref, pos_block=0, x_last=None, start_time=0, n_particles=100):
        self.n_particles = n_particles
        self.generate_particles(y, x0, x_ref, pos_block, x_last, start_time)
        x_states = self.get_state_trajectory()

        return x_states


def run_blocked_pg(f, g, q, r, y, x0, x_ref, pos_block, x_last, start_time, n_particles):
    b_smc = BlockedSMC(f, g, q, r)
    b_smc._n_particles = n_particles
    x_states = b_smc.sample_states(y, x0, x_ref, pos_block, x_last, start_time)
    return x_states


# Given theta estimate states using PG
def blocked_pg_sampler(f, g, q, r, y, x0, x_ref, block_size=10, overlap=2, n_particles=100, max_iter=100):
    if overlap >= block_size / 2:
        sys.exit("Error: Overlap cannot exceed block size")

    seq_len = len(y)
    start_id = np.arange(0, seq_len, block_size - overlap)
    num_blocks = len(start_id)

    '''
    if start_id[-1] == self.seq_len-1:
        num_blocks = len(start_id)-1
    else:
        num_blocks = len(start_id)
    '''

    x_new_ref = np.zeros(seq_len)

    pool = mp.Pool(processes=num_blocks)

    for n_iter in range(1, max_iter):
        inputs = []
        for i in range(num_blocks):
            s = start_id[i]
            u = min(s + block_size-1, seq_len-1)
            if i+1 == num_blocks:  # last block
                pos_block = -1
                x_init = x_ref[s-1]
                x_last = None
            elif i > 0:  # Intermediate block
                pos_block = 1
                x_init = x_ref[s-1]
                x_last = x_ref[u+1]
            else:  # First block
                pos_block = 0
                x_init = x0
                x_last = x_ref[u+1]

            inputs.append((f, g, q, r, y[s:u + 1].copy(), x_init, x_ref[s:u + 1].copy(),
                           pos_block, x_last, s, n_particles))

        results = pool.starmap(run_blocked_pg, inputs)

        for n in range(num_blocks):
            x_new_ref[start_id[n]:(block_size+start_id[n])] = results[n]

        x_ref = x_new_ref.copy()
    pool.close()
    return x_new_ref


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import state_transition_func, transfer_func, generate_data
    import time

    np.random.seed(123)
    # Set up some parameters
    nparticles = 100  # Number of particles
    seqlen = 100  # Length of data record
    f1 = state_transition_func
    g1 = transfer_func
    r1 = 1.
    q1 = 0.1
    xref = np.zeros(seqlen)
    x_0 = np.zeros(nparticles)

    # Generate data`
    data_x, data_y = generate_data(r=r1, q=q1, seq_len=seqlen)

    pf = CSMC(f1, g1, q=q1, r=r1)

    # with ancestor sampling
    l1 = 30
    p1 = 1
    xref = pf.sample_states(data_y, x_0, xref, nparticles)
    s_time = time.time()
    x_pg = blocked_pg_sampler(f1, g1, q1, r1, data_y, x_0, x_ref=xref, block_size=l1, overlap=p1,
                              n_particles=nparticles)
    print("time taken by iterate_blocked_pg: ", time.time() - s_time)

    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x_pg, label='Blocked PG States')
    plt.legend()

    plt.show()
