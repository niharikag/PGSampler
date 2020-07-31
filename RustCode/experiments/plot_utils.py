import csv
import numpy as np
import matplotlib.pyplot as plt
from rust_library import run_particle_gibbs
from experiments.utils import transfer_func, state_transition_func, generate_data


np.random.seed(123)
# Set up some parameters
n_particles = 100
seq_len = 100  # Length of data record
f = state_transition_func
g = transfer_func
r = 1.
q = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = generate_data(r=r, q=q, seq_len=seq_len)


def load_file(file_name):
    data = []

    file = open(file_name)
    reader = csv.reader(file,  delimiter=',')

    for row in reader:
        data.append(row)
    file.close()

    return np.asarray(data[0]).astype(float)


def plot_pg_results():
    max_iters = 10000
    q_initialize = 1.0
    r_initialize = 0.1
    priora = .01
    priorb = .01

    burn_in = int(max_iters * 0.3)

    (x, pg_q, pg_r) = run_particle_gibbs(data_y, 0.0, xref, q_initialize, r_initialize, priora, priorb,
                                         n_particles, False, max_iters)

    q_trace = pg_q[burn_in:]
    n_bins = int(np.floor(np.sqrt(max_iters - burn_in)))
    grid = np.arange(burn_in, max_iters, 1)

    plt.figure(figsize=(10, 4))
    print(max(np.abs(x - data_x)))
    plt.plot(x - data_x, color='indigo')
    plt.xlabel("time")
    plt.ylim((-20, 20))
    plt.ylabel("error in state estimate")
    plt.savefig("../plots/err_x_PG")
    plt.show()

    # Plot Q=q^2
    plt.subplot(3, 1, 1)
    plt.hist(q_trace, n_bins, density=True, facecolor='#9670B3')
    plt.xlabel("Q")
    plt.ylabel("Est. posterior density")
    plt.axvline(np.mean(q_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, q_trace, color='#9670B3') # for Q
    plt.xlabel("iteration")
    plt.ylabel("Q")
    plt.axhline(np.mean(q_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
    idx = int(acf.size / 2)
    acf = acf[idx:]
    acf = acf[0:500]
    acf /= acf[0]
    grid = range(len(acf))
    plt.plot(grid, acf, color='#9670B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of Q")
    plt.savefig("../plots/PG_Q")
    #plt.clf()
    plt.show()

    grid = np.arange(burn_in, max_iters, 1)

    # Plot R=r^2 (black line shows the posterior mean)
    r_trace = pg_r[burn_in:]
    plt.subplot(3, 1, 1)
    plt.hist(r_trace, n_bins, density=True, facecolor='#708CB3')
    plt.xlabel("R")
    plt.ylabel("Est. posterior density")
    plt.axvline(np.mean(r_trace), color='k')

    # Plot the trace of the Markov chain after burn-in (solid black line = posterior mean)
    plt.subplot(3, 1, 2)
    plt.plot(grid, r_trace, color='#708CB3') # for Q
    plt.xlabel("iteration")
    plt.ylabel("R")
    plt.axhline(np.mean(r_trace), color='k')

    # Plot the autocorrelation function
    plt.subplot(3, 1, 3)
    acf = np.correlate(r_trace - np.mean(r_trace), r_trace - np.mean(r_trace), mode='full')
    idx = int(acf.size / 2)
    acf = acf[idx:]
    acf = acf[0:500]
    acf /= acf[0]
    grid = range(len(acf))
    plt.plot(grid, acf, color='#708CB3')
    plt.xlabel("lag")
    plt.ylabel("ACF of R")
    plt.savefig("../plots/PG_R")
    #plt.clf()
    plt.show()


def plot_acf(data, max_iters, burn_in):
    d_trace = data[burn_in:max_iters]

    # Plot the autocorrelation function
    acf = np.correlate(d_trace - np.mean(d_trace), d_trace - np.mean(d_trace), mode='full')
    idx = int(acf.size / 2)
    acf = acf[idx:]
    acf = acf[0:100]
    acf /= acf[0]
    grid = range(len(acf))
    plt.plot(grid, acf, color='#9670B3')
    plt.xlabel("lag")
    plt.ylabel("ACF of Q")
    plt.savefig("../plots/PG_Q")
    #plt.clf()
    plt.show()


if __name__ == "__main__":
    plot_pg_results()

    '''
    file_name = 'output/pg_q.txt'
    data = load_file(file_name)
    max_iters = len(data)
    burn_in = int(max_iters * 0.5)

    plot_acf(data, max_iters, burn_in)
    '''

    if 0:
        file_name = '../output/pg_q.txt'
        data = load_file(file_name)
        max_iters = len(data)
        burn_in = int(max_iters*0.3)

        plot_Q_hist(data, max_iters, burn_in)

        file_name = '../output/pg_r.txt'
        data = load_file(file_name)
        max_iters = len(data)
        #burn_in = int(max_iters*0.3)

        plot_R_hist(data, max_iters, burn_in)