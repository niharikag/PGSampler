import numpy as np
from experiments import utils
import matplotlib.pyplot as plt
from rust_library import run_particle_gibbs
from experiments.interacting_particle_gibbs import interacting_pg_sampler

np.random.seed(123)
# Set up some parameters
seq_len = 50  # Length of data record
f1 = utils.state_transition_func
g1 = utils.transfer_func
r = 1.
q = 0.1
xref = np.zeros(seq_len)
# Generate data


n_particles = 100
max_iters = [10, 100, 1000, 10000] # 50000
#max_iters = 1000
q_initialize = 1.0
r_initialize = 0.1
priora = .01
priorb = .01
iters = 5

n_nodes = 4

#x_csmc = np.zeros((max_iters, seq_len))
mse_pg = np.zeros((len(max_iters), iters))
mse_ipg = np.zeros((len(max_iters), iters))
for i in range(iters):
    data_x, data_y = utils.generate_data(r=r, q=q, seq_len=seq_len)
    for n in range(len(max_iters)) :
        (x, q_pg, r_pg) = run_particle_gibbs(data_y, 0.0, xref, 1.0, 0.1, .01, .01, n_particles, False, max_iters[n])
        m = np.sum((data_x - x) ** 2) / seq_len
        mse_pg[n, i] = m

        x = interacting_pg_sampler(q, r, data_y, n_nodes, n_particles, max_iters[n])
        m = np.sum((data_x - x) ** 2) / seq_len
        mse_ipg[n, i] = m


plt.plot(max_iters, np.sum(mse_pg, axis=1), label="PG")
plt.plot(max_iters, np.sum(mse_ipg, axis=1), label="Interacting PG")
plt.xlabel("iterations")
plt.ylabel("MSE")
plt.legend()
plt.savefig("plots/mse.png")
plt.show()

