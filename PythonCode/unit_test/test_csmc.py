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
from utils import state_transition_func, transfer_func, generate_data
from conditional_smc import CSMC
import numpy as np
from unit_test.parameters import q, r

# Set up some parameters
n_particles = 100  # Number of particles
f = state_transition_func
g = transfer_func
# Generate data
true_x, y = generate_data(r=r, q=q, seq_len=100)
# true_x, y = load_data()
seq_len = len(y)
x_ref = np.zeros(seq_len)
x0 = np.zeros(n_particles)


class CPFTestCase(unittest.TestCase):
    def test_particle_Gibbs(self):
        cpf = CSMC(f, g, q=q, r=r)
        x = cpf.sample_states(y, x0=0, x_ref=x_ref, n_particles=n_particles, resampling_method='multi', max_iter=100)
        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x - x, color='indigo')
        # plt.plot(x, color='indigo', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("error in state estimate")
        plt.ylim((-20, 20))
        plt.savefig("plots/err_x_PG")
        plt.show()

    def test_PGAS(self):
        cpf = CSMC(f, g, q=q, r=r)
        x = cpf.sample_states(y, x0=0, x_ref=x_ref, n_particles=n_particles, ancestor_sampling=True,
                              resampling_method='multi', max_iter=100)
        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x - x, color='indigo')
        # plt.plot(x, color='indigo', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("error in state estimate")
        plt.ylim((-20, 20))
        plt.savefig("plots/err_x_PGAS")
        plt.show()


if __name__ == '__main__':
    unittest.main()
