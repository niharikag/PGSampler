use rand::{Rng};
use std::process::exit;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StudentT;
use statrs::distribution::{StudentsT, Continuous};
use crate::utils;

type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;
pub struct Priors(pub f64, pub f64, pub f64, pub f64);

// Marginal conditional_SMC class
#[allow(dead_code)]
pub struct MargCSMC {
    _f: FDef,
    _g: GDef,
    //_p: Priors,
    _n_particles: usize,
    _a0: f64,
    _b0: f64,
    _c0: f64,
    _d0: f64,
    _alpha_x: f64,
    _alpha_y: f64,
    _beta_x: Vec<f64>,
    _beta_y: Vec<f64>,
    _sx: Vec<f64>,
    _sy: Vec<f64>,
    _seq_len: usize,
    _particles: Vec<Vec<f64>>,
    _normalised_weights: Vec<f64>,
    _ancestors: Vec<Vec<usize>>,
    _rng: StdRng
}

#[allow(dead_code)]
impl MargCSMC {
    pub fn new(f: FDef, g: GDef, p:Priors) -> Self {
        MargCSMC {
            _f: f,
            _g: g,
            //_p: p,
            _n_particles: 100,
            _a0: p.0,
            _b0: p.1,
            _c0: p.2,
            _d0: p.3,
            _alpha_x: 0.0,
            _alpha_y: 0.0,
            _beta_x: Default::default(),
            _beta_y: Default::default(),
            _sx: Default::default(),
            _sy: Default::default(),
            _seq_len: 0,
            _particles: Default::default(),
            _normalised_weights: Default::default(),
            _ancestors: Default::default(),
            _rng: StdRng::from_entropy()
        }
    }

    // sample state trajectory from a set of (weighted) particles
    pub fn _get_state_trajectory(&mut self) -> Vec<f64> {
        if self._seq_len == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }

        let mut x_star: Vec<f64> = vec![0.0; self._seq_len as usize]; //Default::default();
        let mut indx: usize = utils::multinomial_resampling(&mut self._rng,
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
        let mut logweights: Vec<f64> = vec![0.0; self._n_particles];
        let mut max_weight: f64;
        let nu = 2.0 * self._alpha_y;
        let particles = & self._particles[t];
        let ypred:Vec<f64> = (self._g)(particles);

        for i in 0..self._n_particles {
            let sig2 = self._beta_y[i] / self._alpha_y;
            let t_dist = StudentsT::new(y_t, sig2.sqrt(), nu).unwrap();
            logweights[i] = t_dist.pdf(ypred[i]).ln();
        }

        max_weight = logweights[0];
        for i in 0..self._n_particles
        {
            if max_weight < logweights[i] {
                max_weight = logweights[i];
            }
        }
        // log sum trick to avoid numerical underflow
        let weights:Vec<f64> = logweights.iter().map(|x| (x-max_weight).exp()).collect();
        let w_sum:f64 = weights.iter().sum();
        // compute normalized weights
        self._normalised_weights = weights.iter().map(|x| x /w_sum).collect();
    }

    // update statistics and parameters
    fn _update_stat_param(&mut self, y:&Vec<f64>, t: usize) {
        let particles = & self._particles[t];
        let ypred:Vec<f64> = (self._g)(particles);

        // update hyper params
        self._alpha_x = self._alpha_x + 0.5;
        self._alpha_y = self._alpha_y + 0.5;

        if t>0 {
            let anc:& mut Vec<usize> = & mut self._ancestors[t-1];
            let particles = & self._particles[t-1];
            let xpred:Vec<f64> = (self._f)(particles, t - 1);
            let cur_particles = & self._particles[t];

            for n in 0..self._n_particles {
                let indx_n: usize = anc[n];
                self._sx[n] = self._sx[indx_n] - 0.5 * (cur_particles[n] - xpred[indx_n]).powi(2);
                self._sy[n] = self._sy[indx_n] - 0.5 * (y[t] - ypred[n]).powi(2);
                self._beta_x[n] = self._b0 - self._sx[n];
                self._beta_y[n] = self._d0 - self._sy[n];
            }
        }
        else {
            for n in 0..self._n_particles {
                self._sy[n] =  -0.5 * (y[t] - ypred[n]).powi(2);
                self._beta_x[n] = self._b0;
                self._beta_y[n] = self._d0 - self._sy[n];
            }
        }
    }

    fn _resample_step(&mut self, t:usize){
        let mut anc = utils::multinomial_resampling(&mut self._rng,
                                                              &self._normalised_weights,
                                                              &0_usize);
        anc[self._n_particles-1] = self._n_particles-1;
        self._ancestors[t] = anc;
    }

    fn _propagation_step(&mut self, x_ref:f64, t:usize){
        let nu = 2.0 * self._alpha_x;
        let t_dist = StudentT::new(nu).unwrap();
        let particles = & self._particles[t-1];
        let xpred:Vec<f64> = (self._f)(particles, t - 1);
        let cur_particles: & mut Vec<f64> =  & mut self._particles[t];

        for n in 0..(self._n_particles-1) {
            let indx_n:usize = self._ancestors[t-1][n];
            let sig2 = self._beta_x[indx_n] / self._alpha_x;
            let rand_t:f64 = self._rng.sample(t_dist);

            cur_particles[n] = xpred[n] + sig2.sqrt() * rand_t;
        }
        cur_particles[self._n_particles-1] = x_ref;
    }
    // generate particles
    fn _generate_particles(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>) {
        // Initialization of variables
        self._alpha_x = self._a0;
        self._alpha_y = self._c0;
        self._beta_x = vec![self._b0; self._n_particles];
        self._beta_y = vec![self._d0; self._n_particles];
        self._sx = vec![0.0; self._n_particles];
        self._sy = vec![0.0; self._n_particles];

        self._seq_len = y.len(); // Number of states
        self._particles = vec![vec![0.0; self._n_particles]; self._seq_len];
        self._ancestors = vec![vec![0; self._n_particles]; self._seq_len];
        self._normalised_weights = vec![0.0; self._n_particles];

        // deterministically set the initial state
        self._particles[0].iter_mut().for_each(|x| *x = x0);

        //weighting step at t=0
        self._weighting_step(y[0], 0);
        self._update_stat_param(y, 0);

        for t in 1..self._seq_len {
            //resampling step
            //println!("{}", t);
            self._resample_step(t-1);

            // propagation step
            self._propagation_step(x_ref[t], t);

            // weighting step
            self._weighting_step(y[t], t);
            self._update_stat_param(y, t);
        }
        //set the last column of ancestors as [0,1,2,...N-1]
        self._ancestors[self._seq_len - 1] = (0..self._n_particles).collect();
    }

    // generate particles
    pub fn sample_states(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, n_particles: usize,
                         max_iter:usize)->Vec<f64> {
        let mut x_ref_new:Vec<f64> = x_ref.clone();
        self._n_particles = n_particles;
        for _i in 0..max_iter{
            self._generate_particles(y, x0, &x_ref_new);
            x_ref_new = self._get_state_trajectory();
        }
        x_ref_new
    }
}