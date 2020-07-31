import numpy as np
import utils
from particle_gibbs import PG
import matplotlib.pyplot as plt
from conditional_smc import CSMC
from interacting_particle_gibbs import i_pg_states


np.random.seed(123)
# Set up some parameters
seq_len = 100  # Length of data record
f1 = utils.state_transition_func
g1 = utils.transfer_func
r1 = 1.
q1 = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = utils.generate_data(r=r1, q=q1, seq_len=seq_len)

n_particles = 100
max_iters = [100, 200, 300] # 50000
q_initialize = 1.0
r_initialize = 0.1
priora = .01
priorb = .01
#iters = 1000
pg = PG(f1, g1)

#x_csmc = np.zeros((max_iters, seq_len))
mse = []
mse_ipg = []
for n in range(len(max_iters)):
    cpf = CSMC(f1, g1, q=q1, r=r1)
    x = cpf.sample_states(data_y, 0.0, xref, n_particles, max_iter = max_iters[n])
    m = np.sum((data_x-x)**2)/seq_len
    mse.append(m)

    x = i_pg_states(f1, g1, q1, r1, data_y, 0.0, max_iter=max_iters[n])
    m = np.sum((data_x - x) ** 2) / seq_len
    mse_ipg.append(m)

plt.plot(max_iters, mse)
plt.plot(max_iters, mse_ipg)
plt.xlabel("iterations")
plt.ylabel("MSE")
plt.savefig("../plots/mse.png")
plt.show()

