import numpy as np
import utils
from particle_gibbs import PG
import matplotlib.pyplot as plt
from conditional_smc import CSMC
from bootstrap_particle_filter import BPF
from sklearn.neighbors import KernelDensity


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

n_particles = 1000
max_iters = 1000 # 50000
q_initialize = 1.0
r_initialize = 0.1
priora = .01
priorb = .01
#iters = 1000
pg = PG(f1, g1)
burn_in = int(max_iters*0.5)
#x_csmc = np.zeros((max_iters, seq_len))


#for i in range(max_iters):
    #cpf = CSMC(f1, g1, q=q1, r=r1)
    #x = cpf.sample_states(data_y, 0.0, xref, n_particles, max_iter=1)
    #x_csmc[i, :] = x
    #xref = x.copy()


    # xref = x.copy()

bpf = BPF(f1, g1, q=q1, r=r1)
x = bpf.sample_states(data_y, 0.0, n_particles)
x_csmc = bpf.particles

kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x_csmc[:,1].reshape(-1,1))
log_dens = kde.score_samples(x_csmc[:,1].reshape(-1,1))
den = np.exp(log_dens)
#plt.plot(log_dens, 20, density=True, facecolor='#9670B3')
plt.scatter(x_csmc[:,1], den)
plt.xlabel("q")
plt.ylabel("Estimated posterior density")
plt.savefig("../plots/iPG_state_1.png")
plt.show()

x_1 = utils.state_transition_func(0, 0) + 0.1 * np.random.normal(size=10000)
kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(x_1.reshape(-1,1))
log_dens = kde.score_samples(x_1.reshape(-1,1))
den = np.exp(log_dens)
#plt.plot(log_dens, 20, density=True, facecolor='#9670B3')
plt.scatter(x_1, den)
plt.xlabel("q")
plt.ylabel("Estimated posterior density")
plt.savefig("plots/iPG_state_1.png")
plt.show()

'''
plt.hist(x_csmc[:,49], 20, density=True, facecolor='#9670B3')
plt.xlabel("q")
plt.ylabel("Estimated posterior density")
plt.savefig("plots/iPG_state_1.png")
plt.show()

plt.hist(x_csmc[:,99], 20, density=True, facecolor='#9670B3')
plt.xlabel("q")
plt.ylabel("Estimated posterior density")
plt.savefig("plots/iPG_state_1.png")
plt.show()

'''