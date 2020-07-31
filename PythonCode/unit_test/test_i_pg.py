######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# UT for iPG
# Published under GNU General Public License
#####################################################################################
import numpy as np
import unittest
import matplotlib.pyplot as plt
import sys
plt.switch_backend('agg')
sys.path.append(".")
from utils import state_transition_func, transfer_func, generate_data
from interacting_particle_gibbs import interacting_pg, i_pg_states
from unit_test.parameters import q, r, q_init, r_init, prior_a, prior_b


# Set up some parameters
n_particles = 500  # Number of particles

f = state_transition_func
g = transfer_func

# Generate data
true_x, y = generate_data(r=r, q=q, seq_len=100)
# true_x, y = load_data()
n_iter = 100
burnIn = int(n_iter * .3)


class IPGTestCase(unittest.TestCase):

    def test_ipg_only_states(self):

        x = i_pg_states(f, g, q, r, y=y, x0=0, n_nodes=4, n_particles=n_particles, max_iter=n_iter)

        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x-x,  color='indigo')
        plt.xlabel("time")
        plt.ylim((-20, 20))
        plt.ylabel("error in state estimate")
        plt.savefig("plots/err_x_iPG")

    def test_iPG(self):

        x, q_est, r_est = interacting_pg(f, g, y, 0, q_init, r_init, prior_a, prior_b, n_nodes=4,
                                         n_particles=n_particles, max_iter=n_iter)

        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x-x,  color='indigo')
        plt.xlabel("time")
        plt.ylim((-20, 20))
        plt.ylabel("error in state estimate")
        plt.savefig("plots/iPG_err_x")

        n_bins = int(np.floor(np.sqrt(n_iter - burnIn)))
        grid = np.arange(burnIn, n_iter, 1)

        # Plot q=q^2 (black line shows the posterior mean)
        q_trace = q_est[burnIn:n_iter]
        plt.subplot(3, 1, 1)
        plt.hist(q_trace, n_bins, density=True, facecolor='#9670B3')
        plt.xlabel("q")
        plt.ylabel("Estimated posterior density")
        plt.axvline(np.mean(q_trace), color='k')

        # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
        plt.subplot(3, 1, 2)
        plt.plot(grid, q_trace, color='#9670B3')
        plt.xlabel("n_iteration")
        plt.ylabel("q")
        plt.axhline(np.mean(q_trace), color='k')

        # Plot the autocorrelation function
        plt.subplot(3, 1, 3)
        macf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
        idx = int(macf.size / 2)
        macf = macf[idx:]
        macf = macf[0:100]
        macf /= macf[0]
        # grid = range(len(macf))
        plt.plot(grid[0:100], macf, color='#9670B3')
        plt.xlabel("lag")
        plt.ylabel("ACF of q")
        plt.savefig("plots/iPG_Q")
        plt.clf()
        # plt.show()

        # Plot r=r^2 (black line shows the posterior mean)
        r_trace = r_est[burnIn:n_iter]
        plt.subplot(3, 1, 1)
        plt.hist(r_trace, n_bins, density=True, facecolor='#708CB3')
        plt.xlabel("r")
        plt.ylabel("Estimated posterior density")
        plt.axvline(np.mean(r_trace), color='k')

        # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
        plt.subplot(3, 1, 2)
        plt.plot(grid, r_trace, color='#708CB3')
        plt.xlabel("iteration")
        plt.ylabel("r")
        plt.axhline(np.mean(r_trace), color='k')

        # Plot the autocorrelation function
        plt.subplot(3, 1, 3)
        macf = np.correlate(r_trace - np.mean(r_trace), r_trace - np.mean(r_trace), mode='full')
        idx = int(macf.size / 2)
        macf = macf[idx:]
        macf = macf[0:100]
        macf /= macf[0]
        # grid = range(len(macf))
        plt.plot(grid[0:100], macf, color='#708CB3')
        plt.xlabel("lag")
        plt.ylabel("ACF of r")
        plt.savefig("plots/iPG_R")
        plt.clf()


if __name__ == '__main__':
    unittest.main()
