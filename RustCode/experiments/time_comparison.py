import time as t
import numpy as np
from experiments import utils
from experiments.interacting_particle_gibbs import interacting_pg_sampler
from experiments.blocked_particle_gibbs import blocked_pg_sample
from rust_library import run_blocked_smc


np.random.seed(123)
# Set up some parameters
n_particles = 500  # Number of particles
seq_len = 500  # Length of data record
f = utils.state_transition_func
g = utils.transfer_func
r = 1.
q = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = utils.generate_data(r=r, q=q, seq_len=seq_len)

max_iters = [1000, 5000, 10000, 20000]
#max_iters = [10, 50, 100, 200, 500]

def time_ipg():
    pg_time = []
    for iters in max_iters:
        s_time = t.time()
        _res_ipg = interacting_pg_sampler(q, r, y=data_y, x0=0, n_nodes=16, n_particles=n_particles, max_iter=iters)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by iPG:", total_time)



def time_blocked_pg():
    pg_time = []
    x_ref = np.zeros(seq_len)
    x_ref = run_blocked_smc(q, r, data_y, 0.0, x_ref, 0, 0, 0, 100)
    for iters in max_iters:
        s_time = t.time()
        _res_ipg = blocked_pg_sample(q, r, data_y, 0.0, x_ref, 30, 1, n_particles, max_iter=iters)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by blocked PG:", total_time)

#time_ipg()
time_blocked_pg()