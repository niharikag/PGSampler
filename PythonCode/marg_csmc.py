######################################################################################
#
# Author : Niharika Gauraha
#          KTH
#          Email : niharika@kth.se
#
# Published under GNU General Public License
#####################################################################################

# Implements Marginalized Conditional SMC (MCSMC)

import numpy as np
from utils import multinomial_resampling
import sys
import scipy.stats as stats


class MargCSMC(object):
    def __init__(self, f, g, prior):
        if (f is None) or (g is None):
            sys.exit("Error: the input parameters are NULL")
        self._f = f
        self._g = g  # current global model
        self._resampling_method = multinomial_resampling
        self._particles = None
        self._ancestors = None
        self._normalised_weights = None
        self._seq_len = 0
        self._n_particles = 100
        self._a0 = prior[0]
        self._b0 = prior[1]
        self._c0 = prior[2]
        self._d0 = prior[3]
        self._alpha_x = 0.0
        self._alpha_y = 0.0
        self._beta_x = None
        self._beta_y = None
        self._sx = None
        self._sy = None

    def _resample_step(self, t):
        new_ancestors = self._resampling_method(self._normalised_weights[:, t])
        self._ancestors[:, t] = new_ancestors.astype(int)

    def _propagation_step(self, t):
        xpred = self._f(self._particles[:, t - 1], t - 1)
        nu = 2 * self._alpha_x
        sig2 = self._beta_x / self._alpha_x
        ancestors = self._ancestors[:, t - 1]
        self._particles[:, t] = xpred[ancestors] + np.sqrt(sig2) * stats.t.rvs(df=nu, size=self._n_particles)

    def _weighting_step(self, y_t, t):
        if t == 0:
            nu = 2 * self._alpha_y
            sig2 = self._beta_y / self._alpha_y
            ypred = self._g(self._particles[:, t])
            weights = stats.t.logpdf(y_t, df=nu, loc=ypred, scale=np.sqrt(sig2))

            # compute normalized weights
            max_weight = max(weights)  # Subtract the maximum value for numerical stability
            w = np.exp(weights - max_weight)
            self._normalised_weights[:, t] = w / sum(w)

            # sx = sx[newAncestors] + -0.5 * (particles[:, t] - xpred[newAncestors]) ** 2
            self._sy = - 0.5 * (y_t - ypred) ** 2
        else:
            nu = 2 * self._alpha_y
            sig2 = self._beta_y / self._alpha_y
            xpred = self._f(self._particles[:, t-1], t-1)
            ypred = self._g(self._particles[:, t])
            weights =  stats.t.logpdf(y_t, df=nu, loc=ypred, scale=np.sqrt(sig2))
            max_weight = max(weights)  # Subtract the maximum value for numerical stability
            w = np.exp(weights - max_weight)
            self._normalised_weights[:, t] = w / sum(w)
            ancestors = self._ancestors[:, t-1]
            self._sx = self._sx[ancestors] - 0.5 * (self._particles[:, t] - xpred[ancestors]) ** 2
            self._sy = self._sy[ancestors] - 0.5 * (y_t - ypred) ** 2

        self._alpha_x = self._alpha_x + 1 / 2
        self._beta_x = self._b0 - self._sx
        self._alpha_y = self._alpha_y + 1 / 2
        self._beta_y = self._d0 - self._sy

    # sample a state trajectory from _particles
    def _get_state_trajectory(self):
        if self._particles is None:
            print("call generateWeightedParticles first")
            exit(0)

        x_star = np.zeros(self._seq_len)
        indx = multinomial_resampling(self._normalised_weights[:, self._seq_len - 1], 1)[0]

        for t in reversed(range(self._seq_len)):
            indx = self._ancestors[indx, t]
            x_star[t] = self._particles[indx, t]
        return x_star

    def _generate_particles(self, y, x0, x_ref):
        # Stop, if input parameters are NULL
        if y is None:
            sys.exit("Error: the input parameter is NULL")

        # Number of states
        self._seq_len = len(y)

        # initialize
        self._seq_len = len(y)
        self._particles = np.zeros((self._n_particles, self._seq_len))
        self._normalised_weights = np.zeros((self._n_particles, self._seq_len))
        self._ancestors = np.zeros((self._n_particles, self._seq_len), dtype=int)
        self._sx = np.zeros(self._n_particles)
        self._sy = np.zeros(self._n_particles)
        self._beta_x = np.zeros(self._n_particles)
        self._beta_y = np.zeros(self._n_particles)

        # Init state, at t=0
        self._particles[:, 0] = x0  # Deterministic initial condition

        # Hyperarameters
        self._alpha_x = self._a0
        self._beta_x = self._b0
        self._alpha_y = self._c0
        self._beta_y = self._d0

        # weighting step
        self._weighting_step(y[0], 0)

        for t in range(1, self._seq_len):
            # resampling step
            self._resample_step(t - 1)
            self._ancestors[self._n_particles - 1, t-1] = self._n_particles - 1

            # propagation step
            self._propagation_step(t)
            self._particles[self._n_particles - 1, t] = x_ref[t]

            # weighting step
            self._weighting_step(y[t], t)

        self._ancestors[:, self._seq_len-1] = np.arange(self._n_particles)

    def sample_states(self, y, x0, x_ref, n_particles=100, max_iter=100):

        self._n_particles = n_particles

        x_new_ref = x_ref.copy()

        for i in range(max_iter):
            self._generate_particles(y, x0, x_new_ref)
            x_new_ref = self._get_state_trajectory()

        return x_new_ref


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import state_transition_func, transfer_func, generate_data
    # Set up some parameters
    n_particle = 100  # Number of particles
    seq_len = 100  # Length of data record
    f1 = state_transition_func
    g1 = transfer_func
    # R = 1.
    # Q = 0.1
    xref = np.zeros(seq_len)
    x_0 = np.zeros(n_particle)
    # Generate data
    data_x, data_y = generate_data(r=1.0, q=0.1, seq_len=seq_len)

    pf = MargCSMC(f1, g1, np.ones(4))

    x_mult = pf.sample_states(data_y, x0=0, x_ref=xref, n_particles=n_particle, max_iter=100)

    plt.plot(data_x, marker='o', label='Real States', markersize=3)
    plt.plot(x_mult, label='Marg CSMC States')

    plt.legend()
    plt.show()
