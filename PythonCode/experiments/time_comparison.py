import time as t
import numpy as np
import utils
from particle_gibbs import PG
from interacting_particle_gibbs import interacting_pg_sampler
from blocked_particle_gibbs import blocked_pg_sampler
from conditional_smc import CSMC
from collapsed_particle_gibbs import CollapsedPG
import sys


np.random.seed(123)
# Set up some parameters
n_particles = 100  # Number of particles
seq_len = 100  # Length of data record
f = utils.state_transition_func
g = utils.transfer_func
r = 1.
q = 0.1
xref = np.zeros(seq_len)
# Generate data
data_x, data_y = utils.generate_data(r=r, q=q, seq_len=seq_len)

#max_iters = [1000, 5000, 10000, 20000]
max_iters = [10, 50, 100]

def time_pg():
    q_initialize = 0.1
    r_initialize = 1.
    priora = .01
    priorb = .01    
    pg = PG(f, g)
    pg_time = []
    for iters in max_iters:
        s_time = t.time()
        pg.generate_samples(data_y, 0.0, q_initialize, r_initialize, priora, priorb, n_particles=n_particles, max_iter=iters,
                        ancestor_sampling=False)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by PG", total_time)
        sys.stdout.flush()


def time_i_pg():
    n_nodes = 16
    pg_time = []
    for iters in max_iters:
        s_time = t.time()
        interacting_pg_sampler(f, g, q, r, data_y, 0.0, n_nodes, n_particles, iters)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by i PG", total_time)
        
        
def time_blocked_pg():
    block_size = 30
    overlap = 1
    pg_time = []
    pf = CSMC(f, g, q, r)
    x0 = 0.0

    xref = np.zeros(seq_len)
    xref = pf.sample_states(data_y, x0, xref, n_particles)

    for iters in max_iters:
        s_time = t.time()
        blocked_pg_sampler(f, g, q, r, data_y, 0, xref, block_size, overlap, n_particles, iters)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by Blocked PG", total_time)
        sys.stdout.flush()

from collapsed_particle_gibbs import CollapsedPG
def time_collapsed_pg():
    block_size = 30
    overlap = 1
    pg_time = []
    pg = CollapsedPG(f, g, np.ones(4))
    x0 = 0.0
    rinit = 0.1
    qinit = 1.
    priora = .01
    priorb = .01
    for iters in max_iters:
        s_time = t.time()
        pg.generate_samples(data_y, x0, qinit, rinit, priora, priorb, n_particles, iters)
        total_time = t.time() - s_time
        pg_time.append(total_time)
        print("time taken by Collapsed PG", total_time)
        sys.stdout.flush()

time_collapsed_pg()

#time_pg()
#time_blocked_pg()
#time_i_pg()
