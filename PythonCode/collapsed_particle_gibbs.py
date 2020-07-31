######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################
# Implements Collapsed particle Gibbs

import numpy as np
import utils
import sys
import scipy.stats as stats
from marg_csmc import MargCSMC


class CollapsedPG(MargCSMC):
    def __init__(self, f, g, prior):
        """ This method is called when you create an instance of the class."""
        # Stop, if input parameters are NULL
        if (f is None) or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self._q_est = None
        self._r_est = None
        self._x_est = None
        MargCSMC.__init__(self, f, g, prior)

    def generate_samples(self, y, x0, q_init, r_init, prior_a, prior_b, n_particles=100, max_iter=100):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameters are NULL")

        self._seq_len = len(y)
        q = np.zeros(max_iter)
        r = np.zeros(max_iter)
        x = np.zeros((max_iter, self._seq_len))

        q[0] = q_init
        r[0] = r_init
        x_ref = np.zeros(self._seq_len)

        # Initialize the state by running a CPF
        x[0, :] = self.sample_states(y, x0, x_ref, n_particles)

        # Run MCMC loop
        for k in range(1, max_iter):
            # Sample the parameters (inverse gamma posteriors)
            seq_t_1 = np.array(range(0, self._seq_len-1))
            err_q = x[k-1, 1:self._seq_len] - self._f(x[k-1, seq_t_1], seq_t_1)
            err_q = sum(err_q**2)
            q[k] = stats.invgamma.rvs(a=prior_a + (self._seq_len-1)/2, scale=(prior_b + err_q/2), size=1)

            err_r = y - self._g(x[k-1, :])
            err_r = sum(err_r**2)
            r[k] = stats.invgamma.rvs(a=prior_a + self._seq_len/2, scale=(prior_b + err_r/2), size=1)

            x[k, :] = self.sample_states(y, x0, x[k-1, :], n_particles, max_iter=1)

        self._q_est = q
        self._r_est = r
        self._x_est = x

    def get_states(self):
        if self._x_est is None:
            sys.exit("call simulate method first")
        return self._x_est[-1]

    def get_q(self):
        if self._q_est is None:
            sys.exit("call simulate method first")
        return self._q_est

    def get_r(self):
        if self._r_est is None:
            sys.exit("call simulate method first")
        return self._r_est


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # Set up some parameters
    n_particle = 100  # Number of particles
    seq_len = 100  # Length of data record
    f1 = utils.state_transition_func
    g1 = utils.transfer_func
    # R = 1.
    # Q = 0.1
    xref = np.zeros(seq_len)
    # Generate data
    data_x, data_y = utils.generate_data(r=1.0, q=0.1, seq_len=seq_len)

    rinit = 0.1
    qinit = 1.
    priora = .01
    priorb = .01
    n_iter = 10
    burnIn = int(n_iter*.3)
    pg = CollapsedPG(f1, g1, np.ones(4))

    pg.generate_samples(data_y, 0, qinit, rinit, priora, priorb, n_particles=n_particle, max_iter=n_iter)
    q_est = pg.get_q()
    r_est = pg.get_r()
    x_mult = pg.get_states()

    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x_mult, label='CPF+Multinomial Filtered States')
    plt.legend()
    plt.show()

    # plt.show()
    plt.savefig("plots/PG")
    # plt.clf()

    n_bins = int(np.floor(np.sqrt(n_iter - burnIn)))
    grid = np.arange(burnIn, n_iter, 1)

    # Plot Q=q^2 (black line shows the posterior mean)
    q_trace = q_est[burnIn:n_iter]
    plt.subplot(3, 1, 1)
    plt.hist(q_trace, n_bins, density=True, facecolor='#9670B3')
    plt.xlabel("Q")
    plt.ylabel("Estimated posterior density")
    plt.axvline(np.mean(q_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, q_trace, color='#9670B3')
    plt.xlabel("iteration")
    plt.ylabel("Q")
    plt.axhline(np.mean(q_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#9670B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of Q")

    plt.show()

    # Plot R=r^2 (black line shows the posterior mean)
    r_trace = q_est[burnIn:n_iter]
    plt.subplot(3, 1, 1)
    plt.hist(r_trace, n_bins, density=True, facecolor='#708CB3')
    plt.xlabel("R")
    plt.ylabel("Estimated posterior density")
    plt.axvline(np.mean(r_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, r_trace, color='#708CB3')
    plt.xlabel("iteration")
    plt.ylabel("R")
    plt.axhline(np.mean(r_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    macf = np.correlate(r_trace - np.mean(r_trace), r_trace - np.mean(r_trace), mode='full')
    idx = int(macf.size / 2)
    macf = macf[idx:]
    macf = macf[0:100]
    macf /= macf[0]
    grid = range(len(macf))
    plt.plot(grid, macf, color='#708CB3')
    plt.xlabel("lag")
    plt.ylabel("ACF of R")

    plt.show()
