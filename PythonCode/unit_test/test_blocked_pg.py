######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# UT for CPF
# Published under GNU General Public License
#####################################################################################
import unittest
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import sys
sys.path.append(".")
from utils import state_transition_func, transfer_func, generate_data
from blocked_particle_gibbs import iterate_blocked_pg
import numpy as np
from unit_test.parameters import q, r
from conditional_smc import CSMC


# Set up some parameters
n_particles = 100  # Number of particles
f = state_transition_func
g = transfer_func


# Generate data
true_x, y = generate_data(r=r, q=q, seq_len=100)
# true_x, y = load_data()
seq_len = len(y)

x0 = np.zeros(n_particles)


class BlockedSMCTestCase(unittest.TestCase):
    def test_particle_Gibbs(self):
        block_size = 30
        overlap = 2
        x_ref = np.zeros(seq_len)
        pf = CSMC(f, g, q=q, r=r)
        x_ref = pf.sample_states(y, x0, x_ref, n_particles)
        x = iterate_blocked_pg(f, g, q, r, y, 0, x_ref=x_ref, block_size=block_size, overlap=overlap,
                               n_particles=n_particles)
        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x - x, color='indigo')
        # plt.plot(x, color='indigo', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("error in state estimate")
        plt.ylim((-20, 20))
        plt.savefig("plots/err_x_bPG")
        plt.show()


if __name__ == '__main__':
    unittest.main()
