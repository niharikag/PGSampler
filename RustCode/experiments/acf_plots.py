import numpy as np
from rust_library import run_particle_gibbs, run_collapsed_pg
import matplotlib.pyplot as plt

from experiments.utils import transfer_func, state_transition_func, generate_data


np.random.seed(123)
# Set up some parameters
n_particles = [50]
seq_len = 500  # Length of data record
f1 = state_transition_func
g1 = transfer_func
r1 = 1.
q1 = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = generate_data(r=r1, q=q1, seq_len=seq_len)

max_iters = 10000
q_initialize = 1.0
r_initialize = 0.1
priora = .01
priorb = .01

burn_in = int(max_iters*0.3)


def pgas_acf():
    for n in range(len(n_particles)):
        (x, pg_q, pg_r) = run_particle_gibbs(data_y, 0.0, xref, q_initialize, r_initialize, priora, priorb,
                                             n_particles[n], True, max_iters)

        q_trace = pg_q[burn_in:]

        acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
        idx = int(len(acf) / 2)
        acf = acf[idx:]
        acf = acf[0:500]
        acf /= acf[0]

        grid = range(len(acf))

        plt.plot(grid, acf, label=n_particles[n])

    plt.xlabel("lag")
    plt.ylabel("ACF of Q")
    plt.legend()
    # plt.savefig("../plots/PGAS_ACF_Q.png")
    # plt.show()
    # plt.clf()


def pg_acf():
    for n in range(len(n_particles)):
        (x, pg_q, pg_r) = run_particle_gibbs(data_y, 0.0, xref, q_initialize, r_initialize, priora, priorb,
                                             n_particles[n], False, max_iters)

        q_trace = pg_q[burn_in:]

        acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
        idx = int(len(acf) / 2)
        acf = acf[idx:]
        acf = acf[0:15]
        acf /= acf[0]

        grid = range(len(acf))
        plt.figure(figsize=(8, 4))
        # plt.plot(grid, acf, label=n_particles[n])
        plt.plot(grid, acf, label="PG")

    # print(pg_q)
    plt.xlabel("lag")
    plt.ylabel("ACF of Q")
    plt.legend()
    # plt.savefig("../plots/PG_ACF_Q.png")
    # plt.show()
    # plt.clf()


def collapsed_pg_acf():
    for n in range(len(n_particles)):
        (x, pg_q, pg_r) = run_collapsed_pg(data_y, 0.0, xref, q_initialize, r_initialize, priora, priorb,
                                           n_particles[n], max_iters)

        q_trace = pg_q[burn_in:]

        acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
        idx = int(len(acf) / 2)
        acf = acf[idx:]
        acf = acf[0:15]
        acf /= acf[0]

        grid = range(len(acf))

        plt.plot(grid, acf, label="Collapsed G")

    plt.xlabel("lag")
    plt.ylabel("ACF of Q")
    plt.legend()
    plt.savefig("../plots/CollapsedPG_ACF.png")
    # plt.show()
    # plt.clf()


pg_acf()

# pgas_acf()

collapsed_pg_acf()
