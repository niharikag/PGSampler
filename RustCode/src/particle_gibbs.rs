use std::process::exit;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use crate::conditional_smc;
use rand::distributions::Distribution;
use statrs::distribution::{InverseGamma};
use crate::utils;

type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;


// conditional_SMC class
#[allow(dead_code)]
pub struct PG {
    _q_est: Vec<f64>,
    _r_est: Vec<f64>,
    _x: Vec<f64>,
    _f: FDef,
    _g: GDef,
    _rng: StdRng,
    _csmc:conditional_smc::CSMC
}

#[allow(dead_code)]
impl PG {
    pub fn new(f: FDef, g: GDef, anc_resamp: bool) -> Self {
        PG {
            _q_est: Default::default(),
            _r_est: Default::default(),
            _x: Default::default(),
            _f: f,
            _g: g,
            _rng: StdRng::from_entropy(),
            _csmc: conditional_smc::CSMC::new(f, g,0.0, 0.0, anc_resamp)
        }
    }

    // returns estimated states
    pub fn get_states(&self) -> Vec<f64> {
        if self._x.len() == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }
        //let x = self._x[self._x.len()-1].clone();
        self._x.clone()
    }
    // returns estimated q
    pub fn get_q(& self) -> Vec<f64> {
        if self._q_est.len() == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }
        self._q_est.clone()
    }

    // returns estimated r
    pub fn get_r(& self) -> Vec<f64> {
        if self._r_est.len() == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }
        self._r_est.clone()
    }

    pub fn generate_samples(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>,
        q_init: f64, r_init: f64, prior_a: f64, prior_b: f64, n_particles:usize, max_iter:usize){
        let seq_len:usize = y.len();
        self._q_est = vec![0.0; max_iter];
        self._r_est = vec![0.0; max_iter];
        // initialize
        self._q_est[0] = q_init;
        self._r_est[0] = r_init;

        self._csmc.set_noise_param(self._q_est[0], self._r_est[0]);

        let mut x_est = self._csmc.sample_states(y, x0, x_ref, n_particles, 1);

        for iter in 1..max_iter{
            let mut err_q:f64 = 0.0;

            for t in 1..seq_len {
                //let xpred = (self._f)(&x_est[t-1], t-1);
                let xpred = utils::_state_transition_func(x_est[t-1], t-1);
                err_q += (xpred - x_est[t]).powi(2);
            }
            let inv_gamma:InverseGamma = InverseGamma::new( prior_a + ((seq_len-1) as f64)/2.0,
                                       prior_b + err_q/2.0 ).unwrap();
            self._q_est[iter] = inv_gamma.sample(&mut self._rng);

            let mut err_r:f64 = 0.0;
            let ypred: Vec<f64> = (self._g)(&x_est);

            for t in 0..seq_len {
                err_r += (ypred[t] - y[t]).powi(2);
            }

            let inv_gamma:InverseGamma = InverseGamma::new( prior_a + (seq_len as f64)/2.0,
                                           prior_b + err_r/2.0 ).unwrap();
            self._r_est[iter] = inv_gamma.sample(&mut self._rng);

            self._csmc.set_noise_param(self._q_est[iter], self._r_est[iter]);
            x_est = self._csmc.sample_states(y, x0, &x_est, n_particles, 1);
        }
       self._x = x_est;
    }
}