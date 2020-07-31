######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

# Implements standard SMC: boot strap particle filter

import numpy as np
from scipy.stats import norm
import sys
from utils import multinomial_resampling, systematic_resampling, stratified_resampling


class BPF(object):
    def __init__(self, f, g, q, r):
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

    # sample a state trajectory from _particles
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

    # returns log likelihood
    def get_log_likelihood(self):
        return self.log_likelihood

    def generate_particles(self, y, x0=0):
        """
        :param y:
        :param x0:
        :return:
        """

        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # initialize
        self.seq_len = len(y)
        self.log_likelihood = 0
        self.particles = np.zeros((self.n_particles, self.seq_len))
        self.normalised_weights = np.zeros((self.n_particles, self.seq_len))
        self.ancestors = np.zeros((self.n_particles, self.seq_len), dtype=int)

        # Init state, at t=0
        self.particles[:, 0] = x0  # Deterministic initial condition

        # weighting step at t=0
        self.weighting_step(y[0], 0)

        for t in range(1, self.seq_len):
            # resampling step
            self.resample_step(t-1)
            # propagation step
            self.propagation_step(t)
            # weighting step
            self.weighting_step(y[t], t)

        self.ancestors[:, self.seq_len-1] = np.arange(self.n_particles)

    def sample_states(self, y, x0=0, n_particles=100, resampling_method="multi"):
        """
        :param y:
        :param x0:
        :param n_particles:
        :param resampling_method:
        :return:
        """
        self.n_particles = n_particles

        # set resampling method
        if resampling_method == 'systematic':
            self.resampling_method = systematic_resampling
        elif resampling_method == 'stratified':
            self.resampling_method = stratified_resampling
        else:
            self.resampling_method = multinomial_resampling

        self.generate_particles(y, x0)
        x_states = self.get_state_trajectory()

        return x_states


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import transfer_func, state_transition_func, generate_data
    import time

    # Set up some parameters
    n = 500  # Number of particles
    seq_len = 500
    q1 = 0.1
    r1 = 1.0
    np.random.seed(123)
    # Generate data
    data_x, data_y = generate_data(r=r1, q=q1, seq_len=seq_len)

    bpf = BPF(state_transition_func, transfer_func, q=q1, r=r1)
    s_time = time.time()
    x_bpf = bpf.sample_states(data_y, 0, n, "multi")
    print("time", time.time() - s_time)

    plt.figure(figsize=(10, 4))

    plt.plot(x_bpf - data_x,  color='indigo')
    plt.xlabel("time")
    plt.ylim((-20, 20))
    plt.ylabel("error in state estimate")
    plt.savefig("plots/err_x_BPF")
    plt.show()
    '''
    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x_mult, label='Estimated States')
    plt.legend()
    plt.show()
    '''