######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# UT for BPF
# Published under GNU General Public License
#####################################################################################
import unittest
import matplotlib.pyplot as plt
from utils import state_transition_func, transfer_func, generate_data
from bootstrap_particle_filter import BPF
from unit_test.parameters import q, r
import time


# Set up some parameters
n_particles = 500  # Number of particles

f1 = state_transition_func
g1 = transfer_func
seq_len = 100

# Generate data
true_x, y = generate_data(r=r, q=q, seq_len=seq_len)
# true_x, y = load_data()


class BPFTestCase(unittest.TestCase):

    def test_multinomial_BPF(self):
        bpf = BPF(f1, g1, q=q, r=r)
        s_time = time.time()
        x = bpf.sample_states(y, 0, n_particles, "multi")
        print("time", time.time() - s_time)

        self.assertEqual(len(x), len(true_x))
        # plt.plot(true_x, marker='o', label='Real States', markersize=3)
        # plt.plot(x, label='BPF+Multinomial Filtered States')
        plt.plot(true_x-x,  color='indigo')
        # plt.plot(x, color='indigo', linewidth=1.5)
        plt.xlabel("time")
        plt.ylim((-20, 20))
        plt.ylabel("error in state estimate")
        plt.savefig("plots/err_x_BPF")
        plt.show()

    def test_systematic_BPF(self):
        bpf = BPF(f1, g1, q=q, r=r)
        x = bpf.sample_states(y, 0, n_particles, "systematic")
        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x, marker='o', label='Real States', markersize=3)
        plt.plot(x, label='BPF+Multinomial Filtered States')
        # plt.show()

    def test_stratified_BPF(self):
        bpf = BPF(f1, g1, q=q, r=r)
        x = bpf.sample_states(y, 0, n_particles, "stratified")
        self.assertEqual(len(x), len(true_x))
        plt.plot(true_x, marker='o', label='Real States', markersize=3)
        plt.plot(x, label='BPF+Multinomial Filtered States')
        # plt.show()


if __name__ == '__main__':
    unittest.main()
