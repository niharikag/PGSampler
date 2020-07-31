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
import utils
import sys
from conditional_smc import CSMC
from bootstrap_particle_filter import BPF
import multiprocessing as mp
import scipy.stats as stats


def run_bpf(f, g, q, r, y, x0, n_particles):
    pf = BPF(f, g, q, r)
    x_states = pf.sample_states(y, x0, n_particles)
    x_ll = pf.get_log_likelihood()
    return x_states, x_ll


def run_csmc(f, g, q, r, y, x0, x_ref, n_particles):
    pf = CSMC(f, g, q, r)
    x_states = pf.sample_states(y, x0, x_ref, n_particles, max_iter=1)
    x_ll = pf.get_log_likelihood()
    return x_states, x_ll


def interacting_pg_sampler(f, g, q, r, y, x0=0, n_nodes=4, n_particles=10, max_iter=100):
    # Stop, if input parameters are NULL
    if (f is None) or (g is None) or (y is None):
        sys.exit("Error: the input parameters are NULL")

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
    inputs = [(f, g, q, r, y, x0, n_particles) for _i in range(n_node_csmc)]
    results = pool.starmap(run_bpf, inputs)

    for i in range(n_node_csmc):
        x_refs[i, :] = np.array(results[i][0])

    # Run MCMC loop
    for m in range(1, max_iter):
        pool = mp.Pool(processes=n_node_smc)
        inputs = [(f, g, q, r, y, x0, n_particles) for _i in range(n_node_csmc)]
        results = pool.starmap(run_bpf, inputs)

        for i in range(n_node_smc):
            x_smc[i, :] = np.array(results[i][0])
            ll_smc[i] = np.exp(results[i][1])

        pool = mp.Pool(processes=n_node_csmc)
        inputs = [(f, g, q, r, y, x0, x_refs[i], n_particles) for i in range(n_node_csmc)]
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

    norm_weights = ll_csmc/sum(ll_csmc)

    index = utils.multinomial_resampling(norm_weights, size=1)[0]
    return x_refs[index, :]


def interacting_pg_sampler_full(f, g, y, x0, q_init, r_init, prior_a, prior_b, n_nodes=4, n_particles=10, max_iter=100):
    # Stop, if input parameters are NULL
    if (f is None) or (g is None) or (y is None):
        sys.exit("Error: the input parameters are NULL")

    # Number of states
    seq_len = len(y)

    q = np.zeros(max_iter)
    r = np.zeros(max_iter)
    x_est = np.zeros((max_iter, seq_len))

    q[0] = q_init
    r[0] = r_init

    # Initialize
    n_node_smc = n_nodes // 2  # no of nodes running SMC
    n_node_csmc = n_nodes - n_node_smc  # no of nodes running CSMC
    x_smc = np.zeros((n_node_smc, seq_len))
    x_csmc = np.zeros((n_node_csmc, seq_len))
    ll_smc = np.zeros(n_node_smc)
    ll_csmc = np.zeros(n_node_csmc)
    x_refs = np.zeros((n_node_csmc, seq_len))

    pool = mp.Pool(processes=n_node_csmc)
    inputs = [(f, g, q[0], r[0], y, x0, n_particles) for _i in range(n_node_csmc)]
    results = pool.starmap(run_bpf, inputs)

    for i in range(n_node_csmc):
        x_refs[i, :] = np.array(results[i][0])

    # Run MCMC loop
    for m in range(1, max_iter):
        # Sample the parameters (inverse gamma posteriors)
        seq_t_1 = np.array(range(0, seq_len - 1))
        # seq_t_1 = np.float(seq_t_1)
        err_q = x_est[m - 1, 1:seq_len] - f(x_est[m - 1, seq_t_1], seq_t_1)
        err_q = sum(err_q ** 2)
        q[m] = stats.invgamma.rvs(a=prior_a + (seq_len - 1) / 2, scale=(prior_b + err_q / 2), size=1)
        err_r = y - g(x_est[m - 1, :])
        err_r = sum(err_r ** 2)
        r[m] = stats.invgamma.rvs(a=prior_a + seq_len / 2, scale=(prior_b + err_r / 2), size=1)

        # pool = mp.Pool(processes=n_node_smc)
        inputs = [(f, g, q[m], r[m], y, x0, n_particles) for _i in range(n_node_csmc)]

        # with mp.Pool(processes=n_node_smc) as pool:
        # pool = mp.Pool(processes=n_node_smc)
        results = pool.starmap(run_bpf, inputs)

        for i in range(n_node_smc):
            x_smc[i, :] = np.array(results[i][0])
            ll_smc[i] = np.exp(results[i][1])

        # pool = mp.Pool(processes=n_node_csmc)
        inputs = [(f, g, q[m], r[m], y, x0, x_refs[i], n_particles) for i in range(n_node_csmc)]
        results = pool.starmap(run_csmc, inputs)

        for i in range(n_node_csmc):
            x_csmc[i, :] = np.array(results[i][0])
            ll_csmc[i] = np.exp(results[i][1])

        weights = np.zeros(n_node_smc + 1)
        weights[0:n_node_smc] = ll_smc

        for i in range(n_node_csmc):
            weights[n_node_smc] = ll_csmc[i]
            norm_weights = weights/sum(weights)
            index = utils.multinomial_resampling(norm_weights, size=1)[0]
            if index == n_node_smc:
                x_refs[i, :] = x_csmc[i, :]
            else:
                x_refs[i, :] = x_smc[index, :]
                ll_csmc[i] = ll_smc[index]

        norm_weights = ll_csmc / sum(ll_csmc)
        index = utils.multinomial_resampling(norm_weights, size=1)[0]
        x_est[m, :] = x_refs[index, :].copy()
    pool.close()
    norm_weights = ll_csmc/sum(ll_csmc)
    index = utils.multinomial_resampling(norm_weights, size=1)[0]
    return x_refs[index, :], q, r


if __name__ == '__main__':
    import time


    def test_ipg_only_states():
        # Set up some parameters
        n_particles = 100  # Number of particles
        seq_len = 100  # Length of data record
        f = utils.state_transition_func
        g = utils.transfer_func
        r = 1.
        q = 0.1

        # Generate data
        x, y = utils.generate_data(r=1.0, q=0.1, seq_len=seq_len)

        s_time = time.time()
        res_ipg = i_pg_states(f, g, q, r, y=y, x0=0, n_nodes=4, n_particles=n_particles, max_iter=100)
        print("time taken by iPG_states_only:", time.time() - s_time)

        x_axis = list(range(1, seq_len + 1))

        plt.scatter(x_axis, x, s=10)
        plt.plot(x_axis, x, label='Real States')
        plt.plot(x_axis, res_ipg, label='iPMCMC Filtered States')
        plt.legend()
        plt.show()

        # plt.savefig("plots/iPG")


    def test_ipg():
        # Set up some parameters
        n_particles = 100  # Number of particles
        seq_len = 100  # Length of data record
        f = utils.state_transition_func
        g = utils.transfer_func
        # Init and priors
        r_ini = 0.1
        q_ini = 1.
        prior_a = .01
        prior_b = .01

        # Generate data
        x, y = utils.generate_data(r=1.0, q=0.1, seq_len=seq_len)

        s_time = time.time()
        interacting_pg_sampler(f, g, y, 0, q_ini, r_ini, prior_a, prior_b, n_nodes=4, n_particles=n_particles, max_iter=100)
        print("time taken by iPG:", time.time() - s_time)


    test_ipg_only_states()
