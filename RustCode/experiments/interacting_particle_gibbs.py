######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################
# interacting particle Gibbs sampler (iPG)


from matplotlib import pyplot as plt
import numpy as np
from experiments import utils
import multiprocessing as mp
from rust_library import run_csmc
from rust_library import run_bpf


def interacting_pg_sampler(q, r, y, x0=0, n_nodes=4, n_particles=10, max_iter=100):
    # Number of states
    seq_len = len(y)
    # Initialize
    n_node_smc = n_nodes // 2  # no of nodes running SMC
    n_node_csmc = n_nodes - n_node_smc  # no of nodes running CSMC
    x_smc = np.zeros((n_node_smc, seq_len))
    x_csmc = np.zeros((n_node_csmc, seq_len))
    ll_smc = np.zeros(n_node_smc)
    ll_csmc = np.zeros(n_node_csmc)
    x_refs = np.zeros((n_node_csmc, seq_len))

    pool = mp.Pool(processes=n_node_csmc)
    inputs = [(q, r, y, x0, n_particles) for _i in range(n_node_csmc)]
    results = pool.starmap(run_bpf, inputs)

    for i in range(n_node_csmc):
        x_refs[i, :] = np.array(results[i][0])

    # Run MCMC loop
    for m in range(1, max_iter):
        #pool = mp.Pool(processes=n_node_smc)
        inputs = [(q, r, y, x0, n_particles) for _i in range(n_node_csmc)]
        results = pool.starmap(run_bpf, inputs)

        for i in range(n_node_smc):
            x_smc[i, :] = np.array(results[i][0])
            ll_smc[i] = np.exp(results[i][1])

        #pool = mp.Pool(processes=n_node_csmc)
        inputs = [(q, r, y, x0, x_refs[i], n_particles, False, 1) for i in range(n_node_csmc)]
        results = pool.starmap(run_csmc, inputs)

        for i in range(n_node_csmc):
            x_csmc[i, :] = np.array(results[i][0])
            ll_csmc[i] = np.exp(results[i][1])

        weights = np.zeros(n_node_smc + 1)
        weights[0:n_node_smc] = ll_smc

        for i in range(n_node_csmc):
            weights[n_node_smc] = ll_csmc[i]
            norm_weights = weights / sum(weights)

            index = utils.multinomial_resampling(norm_weights, size=1)[0]
            if index == n_node_smc:
                x_refs[i, :] = x_csmc[i, :]
            else:
                x_refs[i, :] = x_smc[i, :]
                ll_csmc[i] = ll_smc[i]

    pool.close()
    norm_weights = ll_csmc/sum(ll_csmc)

    index = utils.multinomial_resampling(norm_weights, size=1)[0]
    return x_refs[index, :]



if __name__ == '__main__':
    import time


    def test_ipg():
        # Set up some parameters
        n_particles = 100  # Number of particles
        seq_len = 100  # Length of data record
        r = 1.
        q = 0.1

        # Generate data
        x, y = utils.generate_data(r=1.0, q=0.1, seq_len=seq_len)

        s_time = time.time()
        res_ipg = interacting_pg_sampler(q, r, y=y, x0=0, n_nodes=16, n_particles=n_particles, max_iter=100)
        print("time taken by iPG:", time.time() - s_time)

        x_axis = list(range(1, seq_len + 1))
        plt.figure(figsize=(10, 4))
        plt.scatter(x_axis, x, s=10)
        plt.plot(x_axis, x, label='Real States')
        plt.plot(x_axis, res_ipg, label='iPMCMC Filtered States')
        plt.legend()
        plt.show()

        plt.savefig("../plots/iPG")


    test_ipg()
