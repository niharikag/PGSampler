use std::process::exit;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand::distributions::Distribution;
use statrs::distribution::{InverseGamma};
use crate::marg_csmc;
use crate::utils;

type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;

// conditional_SMC class
#[allow(dead_code)]
pub struct CollapsedPG {
    _q_est: Vec<f64>,
    _r_est: Vec<f64>,
    _x_est: Vec<f64>,
    _f: FDef,
    _g: GDef,
    _rng: StdRng,
    _mcsmc:marg_csmc::MargCSMC
}

#[allow(dead_code)]
impl CollapsedPG {
    pub fn new(f: FDef, g: GDef) -> Self {
        CollapsedPG {
            _q_est: Default::default(),
            _r_est: Default::default(),
            _x_est: Default::default(),
            _f: f,
            _g:g,
            _rng: StdRng::from_entropy(),
            _mcsmc: marg_csmc::MargCSMC::new(f, g, marg_csmc::Priors(1.0,1.0,1.0,1.0))
        }
    }

    // returns the last state trajectory
    pub fn get_states(&mut self) -> Vec<f64> {
        if self._x_est.len() == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }
        self._x_est.clone()
    }
    // returns estimated q
    pub fn get_q(&mut self) -> Vec<f64> {
        if self._q_est.len() == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }
        self._q_est.clone()
    }

    // returns estimated r
    pub fn get_r(&mut self) -> Vec<f64> {
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
        self._x_est = vec![0.0; seq_len];
        self._q_est[0] = q_init;
        self._r_est[0] = r_init;

        self._x_est = self._mcsmc.sample_states(y, x0, x_ref, n_particles, 1);

        for iter in 1..max_iter{
            let mut err_q:f64 = 0.0;

            for t in 1..seq_len {
                let xpred = utils::_state_transition_func(self._x_est[t-1], t-1);
                err_q += (xpred - self._x_est[t]).powi(2);
            }
            let n = InverseGamma::new( prior_a + ((seq_len-1) as f64)/2.0,
                                       prior_b + err_q/2.0 ).unwrap();
            self._q_est[iter] = n.sample(&mut self._rng);
            //println!("{}", n.sample(&mut self._rng));
            let mut err_r:f64 = 0.0;

            for t in 0..seq_len {
                let ypred = utils::_transfer_func(self._x_est[t]);
                err_r += (ypred - y[t]).powi(2);
            }

            let n = InverseGamma::new( prior_a + (seq_len as f64)/2.0,
                                           prior_b + err_r/2.0 ).unwrap();
            self._r_est[iter] = n.sample(&mut self._rng);

            self._x_est = self._mcsmc.sample_states(y, x0, &self._x_est, n_particles, 1);
        }
        println!("hello");
    }
}