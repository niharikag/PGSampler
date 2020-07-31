use rand::Rng;
use crate::utils;
use std::process::exit;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;

// base SMC (Boot strap particle filter) class
#[allow(dead_code)]
pub struct SMCBase {
    _f: FDef,
    _g: GDef,
    _q:f64,
    _r:f64,
    _seq_len: usize,
    _n_particles: usize,
    _log_likelihood: f64,
    _particles: Vec<Vec<f64>>,
    _normalised_weights: Vec<f64>,
    _ancestors: Vec<Vec<usize>>,
    _q_sqrt: f64,
    _rng: StdRng
}

#[allow(dead_code)]
impl SMCBase {
    pub fn new(f: FDef, g: GDef, q: f64, r: f64) -> Self {
        SMCBase {
            _f: f,
            _g: g,
            _q: q,
            _r: r,
            _seq_len: 0,
            _n_particles: 100,
            _log_likelihood: 0.0,
            _particles: Default::default(),
            _normalised_weights: Default::default(),
            _ancestors: Default::default(),
            _q_sqrt: q.sqrt(),
            _rng: StdRng::from_entropy()
        }
    }

    // returns log-likelihood
    pub fn get_log_likelihood(&self) -> f64 {
        self._log_likelihood
    }

    // sample state trajectory from a set of weighted particles
    fn _get_state_trajectory(&mut self) -> Vec<f64> {
        if self._seq_len == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }

        let mut x_star: Vec<f64> = vec![0.0; self._seq_len as usize]; //Default::default();
        let mut indx: usize = utils::multinomial_resampling(& mut self._rng,
                                                             &self._normalised_weights,
                                                             &1_usize)[0];
        for t in (0..self._seq_len).rev() {
            indx = self._ancestors[t][indx];
            x_star[t] = self._particles[t][indx];
        }
        x_star
    }

    // computes weights of particles set at a given time t, and observation y
    fn _weighting_step(&mut self, y_t:f64, t: usize) {
        let mut weights: Vec<f64> = vec![0.0; self._n_particles];
        let mut max_weight: f64;
        let particles = & self._particles[t];
        let ypred:Vec<f64> = (self._g)(particles);

        for i in 0..self._n_particles {
            //ypred = (self._g)(particles[i]);
            weights[i] = -(1.0 / (2.0 * self._r)) * (ypred[i] - y_t).powi(2)
        }

        max_weight = weights[0];
        for i in 0..self._n_particles
        {
            if max_weight < weights[i] {
                max_weight = weights[i];
            }
        }
        // log sum trick to avoid numerical underflow
        weights.iter_mut().for_each(|x| *x = (*x-max_weight).exp());
        //let weights:Vec<f64> = weights.iter().map(|x| (x-max_weight).exp()).collect();
        let w_sum:f64 = weights.iter().sum();
        // compute normalized weights
        self._normalised_weights = weights.iter().map(|x| x /w_sum).collect();

        // accumulate log-likelihood
        self._log_likelihood += max_weight + w_sum.ln() - (self._n_particles as f64).ln();
    }

    fn _resample_step(&mut self, t:usize){
        self._ancestors[t-1] = utils::multinomial_resampling(&mut self._rng,
                                                              &self._normalised_weights,
                                                              &0_usize);

    }

    fn _propagation_step(&mut self, t:usize){
        let anc:& mut Vec<usize> = & mut self._ancestors[t-1];
        let particles = & self._particles[t-1];//&self._particles.column_mut(t-1).to_vec();
        let xpred:Vec<f64> = (self._f)(particles, t - 1);

        let cur_particles: & mut Vec<f64> =  & mut self._particles[t];
        let mut rand_norm:f64;
        for n in 0..self._n_particles {
            rand_norm = self._rng.sample(StandardNormal);
            cur_particles[n] = xpred[anc[n]] + self._q_sqrt * rand_norm;
        }
    }
    // generate particles
    fn _generate_particles(&mut self, y: &[f64], x0: f64) {
        // Initialization of variables
        self._seq_len = y.len(); // Number of states
        self._particles = vec![vec![0.0; self._n_particles]; self._seq_len];
        self._ancestors = vec![vec![0; self._n_particles]; self._seq_len];
        self._normalised_weights = vec![0.0; self._n_particles];
        self._log_likelihood = 0.0;

        // deterministically set the initial state
        self._particles[0].iter_mut().for_each(|x| *x = x0);

        //weighting step at t=0
        self._weighting_step(y[0], 0);

        for t in 1..self._seq_len {
            //resampling step
            self._resample_step(t);

            // propagation step
            self._propagation_step(t);

            // weighting step
            self._weighting_step(y[t], t);
        }
        //set the last column of ancestors as [0,1,2,...N-1]
        self._ancestors[self._seq_len - 1] = (0..self._n_particles).collect();
    }

    // sample states
    pub fn sample_states(&mut self, y: &[f64], x0: f64, n_particles: usize)->Vec<f64> {
        self._n_particles = n_particles;

        self._generate_particles(y, x0);
        let x_sample:Vec<f64> = self._get_state_trajectory();

        x_sample
    }
}




