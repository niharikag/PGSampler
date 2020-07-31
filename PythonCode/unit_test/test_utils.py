######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# UT for utility functions
# Published under GNU General Public License
#####################################################################################

import unittest
from utils import generate_data, load_data
import matplotlib.pyplot as plt
import numpy as np
from unit_test.parameters import q, r
import sys


plt.switch_backend('agg')
sys.path.append(".")

np.random.seed(123)
seq_len = 500


class UtilsTestCase(unittest.TestCase):
    def test_generateData(self):
        x, y = generate_data(q=q, r=r, seq_len=seq_len)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), seq_len)

    def test_load_data(self):
        x, y = generate_data(q=q, r=r, seq_len=seq_len)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(x), seq_len)

        # Plot data
        plt.figure(figsize=(20, 4))
        plt.subplot(1, 3, 1)
        plt.plot(x[0:100], color='sandybrown', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("latent state")

        plt.subplot(1, 3, 2)
        plt.plot(y[0:100], color='lightskyblue', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("observations")

        plt.subplot(1, 3, 3)
        mu = np.mean(y)
        acf_y = np.correlate(y - mu, y - mu, mode='full')
        idx = int(acf_y.size / 2)
        acf_y = acf_y[idx:]
        acf_y = acf_y[0:100]
        acf_y /= acf_y[0]
        grid = range(len(acf_y))

        # plt.acorr((x - mu), maxlags=5)
        plt.plot(grid, acf_y, color='crimson', linewidth=1.5)
        plt.xlabel("time")
        plt.ylabel("ACF of y")

        plt.savefig("plots/data.png")
        # plt.show()


if __name__ == '__main__':
    unittest.main()
