use rand::Rng;
use crate::utils;
use std::process::exit;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use rusty_machine::learning::gp;
use rusty_machine::learning::toolkit::kernel;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::{Matrix, Vector};

// conditional_SMC class
#[allow(dead_code)]
pub struct CSMC {
    _g: fn(f64) -> f64,
    _q:f64,
    _r:f64,
    _seq_len: usize,
    _n_particles: usize,
    _particles: Vec<Vec<f64>>,
    _normalised_weights: Vec<Vec<f64>>,
    _ancestors: Vec<Vec<usize>>,
    _gaussp: gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean>,
    _rng: StdRng
}

#[allow(dead_code)]
impl CSMC {
    pub fn new(g: fn(f64) -> f64, q: f64, r: f64) -> Self {
        CSMC {
            _g: g,
            _q: q,
            _r: r,
            _seq_len: 0,
            _n_particles: 0,
            _particles: Default::default(),
            _normalised_weights: Default::default(),
            _ancestors: Default::default(),
            _gaussp: Default::default(), //gp::GaussianProcess::new(ker, mean, 1e-3f64),
            _rng: StdRng::from_entropy()
        }
    }

     // set the gaussian process class
    pub fn set_gaussp(&mut self, gaussp: gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean>) {
        self._gaussp = gaussp;
    }

    // sample state trajectory from a set of (weighted) particles
    fn _get_state_trajectory(&mut self) -> Vec<f64> {
        if self._seq_len == 0 {
            println!("call generateWeightedParticles first");
            exit(0);
        }

        let mut x_star: Vec<f64> = vec![0.0; self._seq_len as usize]; //Default::default();
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

    fn _resample_step(&mut self, t:usize){
        self._ancestors[t-1] = utils::multinomial_resampling(&mut self._rng,
                                                              &self._normalised_weights[t-1],
                                                              &0_usize);
        self._ancestors[t-1][self._n_particles-1] = self._n_particles-1;
    }

    fn _propagation_step(&mut self, t:usize){
        let q:f64 = self._q.sqrt();
        let test_inputs = Matrix::new(self._n_particles,1, self._particles[t-1].to_vec());
        let xpred = self._gaussp.predict(&test_inputs).unwrap();

        for n in 0..(self._n_particles-1) {
            let rand_norm:f64 = self._rng.sample(StandardNormal);
            let indx_n:usize = self._ancestors[t-1][n];
            //let xpred = (self._f)(self._particles[t-1][indx_n], t - 1);
            self._particles[t][n] = xpred[indx_n] + q * rand_norm;
        }

        //self._particles[t][self._n_particles-1] = x_ref[t];
    }

    // generate particles
    pub fn _generate_particles(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>) {
        // Initialization of variables
        //self._n_particles = n_particles;
        self._seq_len = y.len(); // Number of states
        self._particles = vec![vec![0.0; self._n_particles]; self._seq_len];
        self._ancestors = vec![vec![0; self._n_particles]; self._seq_len];
        self._normalised_weights = vec![vec![0.0; self._n_particles]; self._seq_len];

        // deterministically set the initial state
        self._particles[0].iter_mut().for_each(|x| *x = x0);

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
            self._propagation_step(t);

            // weighting step
            self._weighting_step(y, t);
            //println!("{:?}", self._ancestors);
        }
        //set the last column of ancestors as [0,1,2,...N-1]
        self._ancestors[self._seq_len - 1] = (0..self._n_particles).collect();
    }

    // sample states
    pub fn sample_states(&mut self, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>,
                          n_particles: usize, max_iter:usize)->Vec<f64> {
        let mut x_ref_new:Vec<f64> = x_ref.clone();
        self._n_particles = n_particles;

        for _i in 0..max_iter{
            self._generate_particles(y, x0, &x_ref_new);
            x_ref_new = self._get_state_trajectory();
        }
        x_ref_new
    }
}

#[allow(dead_code)]
pub fn learn_guassp(x: &Vec<f64>)->gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean>{
    let seq_len:usize = x.len();
    let gp_input = Matrix::new(seq_len-1,1, x[0..seq_len - 1].to_vec()); //Vector::new(x[0..seq_len-1].to_vec()); //
    let targets = Vector::new(x[1..seq_len].to_vec()); //Vector::new(seq_len,1, x[1..seq_len].to_vec());

    // A squared exponential kernel with lengthscale 2, and amplitude 1.
    let ker = kernel::SquaredExp::new(2., 1.);

    // The zero function
    let zero_mean = gp::ConstMean::default();

    // Construct a GP with the specified kernel, mean, and a noise of 0.5.
    let mut gp = gp::GaussianProcess::new(ker, zero_mean, 1e-3f64);

    // Train the model!
    gp.train(&gp_input, &targets).unwrap();

    gp
}

#[allow(dead_code)]
pub fn gp_ssm(g: fn(f64) -> f64, q: f64, r:f64, y:&Vec<f64>, x0:f64, n_particles:usize, max_iter:usize)
    ->(Vec<f64>, gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean>){
    let seq_len:usize = y.len();
    let x_ref:Vec<f64> = vec![0.0; seq_len];
    let mut x_ref_new:Vec<f64> = x_ref.clone();
    let mut csmc:CSMC = CSMC::new(g, q, r);
    let mut gpr_model: gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean>;

    for _k in 1..max_iter{
        gpr_model = learn_guassp(&x_ref_new);
        csmc.set_gaussp(gpr_model);
        x_ref_new = csmc.sample_states(y, x0, &x_ref_new, n_particles, 1)
    }
    let gpr_model_new:gp::GaussianProcess<kernel::SquaredExp, gp::ConstMean> = learn_guassp(&x_ref_new);
    (x_ref_new, gpr_model_new)
}




