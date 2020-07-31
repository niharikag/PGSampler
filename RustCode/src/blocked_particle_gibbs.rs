use rand::Rng;
//use crate::conditional_smc::CSMC;
use crate::utils;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use std::process::exit;
use rand_distr::StandardNormal;


// consitional_SMC class
#[allow(dead_code)]
pub struct BlockedSMC {
    _f: fn(f64, usize) -> f64,
    _g: fn(f64) -> f64,
    _q:f64,
    _r:f64,
    _seq_len: usize,
    _n_particles: usize,
    _particles: Vec<Vec<f64>>,
    _normalised_weights: Vec<Vec<f64>>,
    _ancestors: Vec<Vec<usize>>,
    _pos_block: i32,
    _rng: StdRng
}

#[allow(dead_code)]
impl BlockedSMC {
    pub fn new(f: fn(f64, usize) -> f64, g: fn(f64) -> f64, q: f64, r: f64) -> Self {
        BlockedSMC {
            _f: f,
            _g: g,
            _q: q,
            _r: r,
            _seq_len: 0,
            _n_particles: 0,
            _particles: Default::default(),
            _normalised_weights: Default::default(),
            _ancestors: Default::default(),
            _pos_block: 0,
            _rng: StdRng::from_entropy()
        }
    }

    // sample state trajectory from a set of (weighted) particles
    fn _get_state_trajectory(&mut self) -> Vec<f64> {
        if self._seq_len == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }

        let mut x_star: Vec<f64> = vec![0.0; self._seq_len]; //Default::default();
        let mut indx: usize = utils::multinomial_resampling(&mut self._rng,
                                                             &self._normalised_weights[self._seq_len - 1],
                                                             &1_usize)[0];
        for t in (0..self._seq_len).rev() {
            indx = self._ancestors[t][indx];
            x_star[t] = self._particles[t][indx];
        }
        x_star
    }

    // computes weights of particles set at a given time t, and observation y
    fn _weighting_step(&mut self, y:&Vec<f64>, t: usize) {
        let mut logweights: Vec<f64> = vec![0.0; self._n_particles];
        let mut max_weight: f64;
        let mut ypred;

        for i in 0..self._n_particles {
            ypred = (self._g)(self._particles[t][i]);
            logweights[i] = -(1.0 / (2.0 * self._r)) * (ypred - y[t]).powi(2)
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
        self._normalised_weights[t] = weights.iter().map(|x| x /w_sum).collect();
    }

    // re-weighting step for last state (of a block)
    fn _reweighting_step(&mut self, y:&Vec<f64>, t: usize, x_last:f64, start_time:usize) {
        let mut logweights: Vec<f64> = vec![0.0; self._n_particles];
        let mut max_weight: f64;
        let mut ypred:f64;
        let mut xpred:f64;

        for i in 0..self._n_particles {
            ypred = (self._g)(self._particles[t][i]);
            logweights[i] = -(1.0 / (2.0 * self._r)) * (ypred - y[t]).powi(2);
            xpred = (self._f)(self._particles[t][i], start_time+t);
            logweights[i] += -(1.0 / (2.0 * self._q)) * (xpred - x_last).powi(2);
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
        self._normalised_weights[t] = weights.iter().map(|x| x /w_sum).collect();
    }

    fn _resample_step(&mut self, t:usize){
        self._ancestors[t-1] = utils::multinomial_resampling(&mut self._rng,
                                                              &self._normalised_weights[t-1],
                                                              &0);
        self._ancestors[t-1][self._n_particles-1] = self._n_particles-1;
    }

    fn _propagation_step(&mut self, t:usize, start_time:usize){
        let q:f64 = self._q.sqrt();

        for n in 0..(self._n_particles-1) {
            let rand_norm:f64 = self._rng.sample(StandardNormal);
            let indx_n:usize = self._ancestors[t-1][n];
            let xpred = (self._f)(self._particles[t-1][indx_n], start_time+t - 1);
            self._particles[t][n] = xpred + q * rand_norm;
        }

    }
    // generate particles
    fn _generate_particles(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, pos_block:i32,
                               x_last:f64, start_time:usize) {
        // Initialization of variables
        //self._n_particles = n_particles;
        self._seq_len = y.len(); // Number of states
        self._particles = vec![vec![0.0; self._n_particles]; self._seq_len];
        self._ancestors = vec![vec![0; self._n_particles]; self._seq_len];
        self._normalised_weights = vec![vec![0.0; self._n_particles]; self._seq_len];

        // deterministically set the initial state
        if pos_block == 0{
            self._particles[0].iter_mut().for_each(|x| *x = x0);
        }
        else {
            for n in 0..self._n_particles{
                let rand_norm:f64 = self._rng.sample(StandardNormal);
                self._particles[0][n] = x0 + self._q.sqrt()* rand_norm;
            }
        }

        // set the last particle as the given reference trajectory
        for t in 0..self._seq_len{
            self._particles[t][self._n_particles-1] = x_ref[t];
            self._ancestors[t][self._n_particles-1] = self._n_particles-1;
        }

        //weighting step at t=0
        self._weighting_step(y, 0);

        for t in 1..self._seq_len {
            //resampling step
            self._resample_step(t);

            // propagation step
            self._propagation_step(t, start_time);

            // weighting step
            self._weighting_step(y, t);
        }

        // re-weighting step, except for the last block
        if pos_block >= 0{
            self._reweighting_step(y, self._seq_len-1, x_last, start_time);
        }

        //set the last column of ancestors as [0,1,2,...N-1]
        self._ancestors[self._seq_len - 1] = (0..self._n_particles).collect();
    }

    // generate particles
    pub fn sample_states(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, pos_block:i32,
                          x_last:f64, start_time:usize,  n_particles: usize)->Vec<f64> {
        //let mut x_ref_new:Vec<f64> = x_ref.clone();
        self._n_particles = n_particles;

        self._generate_particles(y, x0, x_ref, pos_block, x_last, start_time);
        let x_ref_new = self._get_state_trajectory();

        x_ref_new
    }
}

#[allow(dead_code)]
pub fn run_blocked_smc(q: f64, r: f64, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, pos_block:i32,
                          x_last:f64, start_time:usize, n_particles: usize)->Vec<f64> {
    let f = utils::_state_transition_func;
    let g: fn(f64) -> f64 = utils::_transfer_func;
    let mut b_smc: BlockedSMC = BlockedSMC::new(f, g, q, r);

    let x_states: Vec<f64> = b_smc.sample_states(y, x0, x_ref, pos_block, x_last,
                                                 start_time, n_particles);

    x_states
}

#[allow(dead_code)]
pub fn iterate_blocked_pg(q: f64, r: f64, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, n_particles: usize,
                    l:usize, p:usize, max_iter:usize)->Vec<f64> {

    let seq_len:usize = y.len();
    let start_ids:Vec<usize> = (0..seq_len).step_by(l-p).collect();
    let mut x_ref_new: Vec<f64> = vec![0.0;seq_len];
    let num_blocks:usize;
    let y_temp = y.clone();
    let mut x_ref_temp = x_ref.clone();

    if start_ids[start_ids.len()-1] == seq_len-1{
        num_blocks = start_ids.len()-1;
    }
    else {
        num_blocks = start_ids.len();
    }

    for _m in 0..max_iter {
        for i in 0..num_blocks {
            let s = start_ids[i];
            let u:usize;
            let pos_block:i32;
            let x_init:f64;
            let x_last:f64;

            if s + l-1 < seq_len-1{
                u = s + l-1;
            }
            else {
                u = seq_len-1
            }
            let mut y_b: Vec<f64> = vec![0.0;u-s+1];
            let mut x_ref_b: Vec<f64> = vec![0.0;u-s+1];

            if i+1 == num_blocks{
               //last block
                pos_block = -1;
                x_init = x_ref_temp[s-1];
                x_last = 0.0; //None
            }else if i > 0 {
                // intermediate block
                pos_block = 1;
                x_init = x_ref_temp[s-1];
                x_last = x_ref_temp[u+1];
            }else {
                // first block
                x_init = x0;
                x_last = x_ref_temp[u+1];
                pos_block = 0;
            }
            //y_temp[s..u + 1].clone_from_slice(&y_b);
            y_b.copy_from_slice(&y_temp[s..u + 1]);
            x_ref_b.copy_from_slice(&x_ref_temp[s..u + 1]);

            //println!("{:?}", x_ref_temp);
            let result = run_blocked_smc(q, r, &y_b, x_init, &x_ref_b,
                                         pos_block, x_last, s, n_particles);
            //println!("{}",u);
            for (i,n) in (s..u).enumerate() {
                x_ref_new[n] = result[i];
            }
            //println!("{:?}",result);
        }

        x_ref_temp = x_ref_new.clone();
    }

    x_ref_new.clone()
}