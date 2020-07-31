import numpy as np
import utils
from particle_gibbs import PG
import matplotlib.pyplot as plt

np.random.seed(123)
# Set up some parameters
n_particles = [5, 10] #, 50, 100]  # Number of particles
seq_len = 100  # Length of data record
f1 = utils.state_transition_func
g1 = utils.transfer_func
r1 = 1.
q1 = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = utils.generate_data(r=r1, q=q1, seq_len=seq_len)

max_iters = 1000 # 50000
q_initialize = 1.0
r_initialize = 0.1
priora = .01
priorb = .01
pg = PG(f1, g1)
burn_in = int(max_iters*0.5)
pg_q = []
pgas_q = []*len(n_particles)
for n in range(len(n_particles)):
    pg.generate_samples(data_y, 0.0, q_initialize, r_initialize, priora, priorb, n_particles=n_particles[n],
                        max_iter=max_iters, ancestor_sampling=False)
    pg_q.append(pg.get_q())

    '''
    pg.generate_samples(data_y, 0.0, q_initialize, r_initialize, priora, priorb, n_particles=N, max_iter=iters,
                        ancestor_sampling=True)
    pgas_q[n] = pg.get_q()
    '''
    q_trace = pg_q[n][burn_in:]

    acf = np.correlate(q_trace - np.mean(q_trace), q_trace - np.mean(q_trace), mode='full')
    idx = int(len(acf) / 2)
    acf = acf[idx:]
    acf = acf[0:100]
    acf /= acf[0]
    #grid = np.arange(burn_in, max_iters, 1)
    grid = range(len(acf))
    plt.plot(grid, acf, label=n_particles[n])

plt.xlabel("lag")
plt.ylabel("ACF of Q")
plt.legend()
plt.savefig("../plots/PGAS_ACF_Q")
plt.show()
#plt.clf()



