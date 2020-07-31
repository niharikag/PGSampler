######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# UT for collapsed PG
# Published under GNU General Public License
#####################################################################################

import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from utils import state_transition_func, transfer_func, generate_data
from collapsed_particle_gibbs import CollapsedPG
from unit_test.parameters import q, r, q_init, r_init, prior_a, prior_b
plt.switch_backend('agg')

# Set up some parameters
N = 500  # Number of particles

f = state_transition_func
g = transfer_func
seq_len = 100
# Load or Generate data
true_x, y = generate_data(r=r, q=q, seq_len=seq_len)
# true_x, y = load_data('data/test.json')
x0 = 0.0
max_iter = 100
burnIn = int(max_iter * .3)


class PGTestCase(unittest.TestCase):
    def test_PG(self):
        pg = CollapsedPG(f, g, np.ones(4))
        pg.generate_samples(y, x0, q_init, r_init, prior_a, prior_b, n_particles=N, max_iter=max_iter)
        x = pg.get_states()
        q_est = pg.get_q()
        r_est = pg.get_r()

        self.assertEqual(len(x), len(true_x))

        plt.plot(true_x - x, color='indigo')
        # plt.plot(x, color='indigo', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("error in state estimate")
        plt.ylim((-20, 20))
        plt.savefig("plots/collapsed_PG_err_x")
        plt.clf()
        # plt.show()

        n_bins = int(np.floor(np.sqrt(max_iter - burnIn)))
        grid = np.arange(burnIn, max_iter, 1)

        # Plot Q=q^2 (black line shows the posterior mean)
        q_trace = q_est[burnIn:max_iter]
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
        # grid = range(len(macf))
        plt.plot(grid[0:100], macf, color='#9670B3')
        plt.xlabel("lag")
        plt.ylabel("ACF of Q")
        plt.savefig("plots/collapsed_PG_Q")
        plt.clf()
        # plt.show()

        # Plot R=r^2 (black line shows the posterior mean)
        r_trace = r_est[burnIn:max_iter]
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
        # grid = range(len(macf))
        plt.plot(grid[0:100], macf, color='#708CB3')
        plt.xlabel("lag")
        plt.ylabel("ACF of R")
        plt.savefig("plots/collapsed_PG_R")
        plt.clf()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
