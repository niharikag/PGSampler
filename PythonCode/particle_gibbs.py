######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################
# particle Gibbs sampler

# Input:
#   param - state parameters
#   y - measurements
#   x0 - initial state
#   max_iter - number of MCMC runs
#   N - number of particles
#   resamplingMethod - resampling methods:
#     multinomical and systematics resampling methods are supported
# Output:
#       The function returns the sample paths of (q, r, x_{1:T})

import numpy as np
import utils
import sys
import scipy.stats as stats
from conditional_smc import CSMC


class PG(object):
    def __init__(self, f, g, q=0, r=0):
        """ This method is called when you create an instance of the class."""
        # Stop, if input parameters are NULL
        if (f is None) or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self.q_est = None  # process noise
        self.r_est = None  # measurement noise
        self.x_est = None  # states
        self.seq_len = 0
        self.n_particles = 100
        self.csmc = CSMC(f, g, q, r)

    def generate_samples(self, y, x0, q_init, r_init, prior_a, prior_b, n_particles=100, max_iter=100,
                         resampling_method='multi', ancestor_sampling=False):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameters are NULL")

        self.seq_len = len(y)  # Number of states
        self.n_particles = n_particles

        # Initialize the state parameters
        q = np.zeros(max_iter)  # process noise
        r = np.zeros(max_iter)  # measurement noise
        x = np.zeros((max_iter, self.seq_len))  # states

        q[0] = q_init
        r[0] = r_init
        x_ref = np.zeros(self.seq_len)

        # Initialize the noise parameters
        # self.csmc.set_noise_param(q[0], r[0])
        self.csmc.q = q[0]
        self.csmc.r = r[0]

        # Initialize the state by running a CPF
        x[0, :] = self.csmc.sample_states(y, x0, x_ref, n_particles, ancestor_sampling,
                                          resampling_method, 1)

        # Run MCMC loop
        for k in range(1, max_iter):
            # Sample the parameters (inverse gamma posteriors)
            seq_t_1 = np.array(range(0, self.seq_len-1))
            err_q = x[k-1, 1:self.seq_len] - self.csmc.f(x[k-1, seq_t_1], seq_t_1)
            err_q = sum(err_q**2)
            q[k] = stats.invgamma.rvs(a=prior_a + (self.seq_len-1)/2, scale=(prior_b + err_q/2), size=1)
            err_r = y - self.csmc.g(x[k-1, :])
            err_r = sum(err_r**2)
            r[k] = stats.invgamma.rvs(a=prior_a + self.seq_len/2, scale=(prior_b + err_r/2), size=1)

            # Run CPF
            # self.set_noise_param(q[k], r[k])
            self.csmc.q = q[k]
            self.csmc.r = r[k]

            x[k, :] = self.csmc.sample_states(y, x0, x[k-1, :], max_iter=1)

        self.q_est = q
        self.r_est = r
        self.x_est = x

    def get_states(self):
        if self.x_est is None:
            sys.exit("call simulate method first")
        return self.x_est[-1]

    def get_q(self):
        if self.q_est is None:
            sys.exit("call simulate method first")
        return self.q_est

    def get_r(self):
        if self.r_est is None:
            sys.exit("call simulate method first")
        return self.r_est


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import time as t
    np.random.seed(123)
    # Set up some parameters
    N = 200  # Number of particles
    seq_len = 100  # Length of data record
    f1 = utils.state_transition_func
    g1 = utils.transfer_func
    r1 = 1.
    q1 = 0.1
    xref = np.zeros(seq_len)
    # Generate data
    data_x, data_y = utils.generate_data(r=r1, q=q1, seq_len=seq_len)

    q_initialize = 1.0
    r_initialize = 0.1
    priora = .01
    priorb = .01
    iters = 100
    burnIn = int(iters*.3)
    pg = PG(f1, g1)
    s_time = t.time()
    pg.generate_samples(data_y, 0.0, q_initialize, r_initialize, priora, priorb, n_particles=N, max_iter=iters,
                        ancestor_sampling=False)
    print("time taken by PG", t.time() - s_time)

    q_est = pg.get_q()
    r_est = pg.get_r()
    x_sample = pg.get_states()

    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x_sample[-1], label='CPF+Multinomial Filtered States')
    plt.legend()
    # plt.show()

    plt.savefig("plots/PG")
    plt.clf()

    n_bins = int(np.floor(np.sqrt(iters - burnIn)))
    grid = np.arange(burnIn, iters, 1)

    # Plot q=q^2 (black line shows the posterior mean)
    q_trace = q_est[burnIn:iters]
    print(q_trace)
    plt.subplot(3, 1, 1)
    plt.hist(q_trace, n_bins, density=True, facecolor='#9670B3')
    plt.xlabel("q")
    plt.ylabel("Estimated posterior density")
    plt.axvline(np.mean(q_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, q_trace, color='#9670B3')
    plt.xlabel("iteration")
    plt.ylabel("q")
    plt.axhline(np.mean(q_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
    idx = int(acf.size / 2)
    acf = acf[idx:]
    acf = acf[0:100]
    acf /= acf[0]
    grid = range(len(acf))
    plt.plot(grid, acf, color='#9670B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of q")

    # plt.show()
    plt.savefig("plots/systemNoise")
    plt.clf()

    # Plot r=r^2 (black line shows the posterior mean)
    r_trace = r_est[burnIn:iters]
    plt.subplot(3, 1, 1)
    plt.hist(r_trace, n_bins, density=True, facecolor='#708CB3')
    plt.xlabel("r")
    plt.ylabel("Estimated posterior density")
    plt.axvline(np.mean(r_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    grid = range(len(r_trace))
    plt.plot(grid, r_trace, color='#708CB3')
    plt.xlabel("iteration")
    plt.ylabel("r")
    plt.axhline(np.mean(r_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    acf = np.correlate(r_trace - np.mean(r_trace), r_trace - np.mean(r_trace), mode='full')
    idx = int(acf.size / 2)
    acf = acf[idx:]
    acf = acf[0:100]
    acf /= acf[0]
    grid = range(len(acf))
    plt.plot(grid, acf, color='#708CB3')
    plt.xlabel("lag")
    plt.ylabel("ACF of r")

    # plt.show()
    plt.savefig("plots/measurementNoise")
