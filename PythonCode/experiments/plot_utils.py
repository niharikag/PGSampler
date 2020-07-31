######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np


# plot geneology of the survived particles
def particle_geneology(particles, ancestors, len_geneology=10):
    n_particles, seq_len = particles.shape
    x_matrix = np.zeros((n_particles, seq_len))
    start_index = seq_len - len_geneology

    for t in range(seq_len):
        x_matrix[:, t] = t

    # plot all the particles first
    plt.scatter(x_matrix[:, start_index:seq_len], particles[:, start_index:seq_len], s=10)

    # plot geneology
    x_star = np.zeros(len_geneology)
    for j in range(n_particles):
        index = 0
        for t in range(start_index, seq_len):
            x_star[index] = particles[ancestors[j, t], t]
            index = index+1

        x_dim = list(range(start_index, seq_len))
        plt.plot(x_dim, x_star, lw=1, color='grey')
    # plt.show()


# plot geneology of all the particles generated (survived and died)
def particle_geneology_all(particles, ancestors, len_geneology=10):
    n_particles, seq_len = particles.shape
    x_matrix = np.zeros((n_particles, seq_len))
    start_index = seq_len - len_geneology

    for t in range(start_index, seq_len):
        x_matrix[:, t] = t

    # plot all the particles first
    plt.scatter(x_matrix[:, start_index:seq_len], particles[:, start_index:seq_len], color="black", s=10)

    # plot geneology
    x_star = np.zeros(seq_len)
    for i in range(seq_len-1, start_index-1, -1):
        for j in range(n_particles):
            x_star[i] = particles[j, i]

            for t in range(start_index, i):
                x_star[t] = particles[ancestors[j, t], t]
            x_dim = list(range(start_index, i+1))
            plt.plot(x_dim, x_star[start_index:i+1], color="grey")

    # plot geneology of survived
    x_star = np.zeros(len_geneology)
    for j in range(n_particles):
        index = 0
        for t in range(start_index, seq_len):
            x_star[index] = particles[ancestors[j, t], t]
            index = index + 1

        x_dim = list(range(start_index, seq_len))
        plt.plot(x_dim, x_star, color='red')

    # plt.show()


def acf(x, maxlag=50):
    """
    Plots the autocorrelation function of a sequence x
    :param x: Sequence
    :param maxlag: largest lag
    """
    mu = np.mean(x)
    plt.acorr((x - mu), maxlags=maxlag)
    # plt.show


if __name__ == '__main__':
    data_x = np.random.randn(100)
    acf(data_x)
    plt.show()
