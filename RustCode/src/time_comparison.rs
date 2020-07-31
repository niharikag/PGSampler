use crate::particle_gibbs::PG;
use crate::collapsed_particle_gibbs::CollapsedPG;
use crate::utils;
use crate::gp_ssm::gp_ssm;

use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::{SystemTime};

#[allow(dead_code)]
static SEED: [u8; 32] = [0;32];

#[allow(dead_code)]
fn test_pg(max_iter: &usize){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 500;
    let n_particles:usize = 500;
    //let max_iter:usize = 1000;
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let r_init:f64 = 0.1;
    let q_init:f64 = 1.0;
    let prior_a:f64 = 0.01;
    let prior_b:f64 = 0.01;
    let anc_resamp:bool = true;

    let (_x,y) = utils::_generate_data(&mut rng_mut,
                                      utils::_state_transition_func,
                                      utils::_transfer_func, q, r, x0, seq_len);

    let mut pg:PG = PG::new(utils::state_trans_func,
                            utils::transfer_func,
                               anc_resamp);
    let now = SystemTime::now();
    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, *max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by PG: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
}

#[allow(dead_code)]
pub fn time_pg(){
    let mat_iter:[usize;4] = [1000, 5000, 10000, 20000];
    for iter in mat_iter.iter(){
        test_pg(&iter);
    }
}


#[allow(dead_code)]
fn test_collapsed_pg(max_iter: &usize){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 500;
    let n_particles:usize = 500;
    //let max_iter:usize = 1000;
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let r_init:f64 = 0.1;
    let q_init:f64 = 1.0;
    let prior_a:f64 = 0.01;
    let prior_b:f64 = 0.01;

    let (_x,y) = utils::_generate_data(&mut rng_mut,
                                      utils::_state_transition_func,
                                      utils::_transfer_func, q, r, x0, seq_len);

    let mut pg:CollapsedPG = CollapsedPG::new(utils::state_trans_func,utils::transfer_func);
    let now = SystemTime::now();
    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, *max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by Collapsed PG: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
}

#[allow(dead_code)]
pub fn time_collapsed_pg(){
    let mat_iter:[usize;4] = [1000, 5000, 10000, 20000];
    for iter in mat_iter.iter(){
        test_collapsed_pg(&iter);
    }
}

#[allow(dead_code)]
pub fn g(x:f64)->f64 {
    x
}

#[allow(dead_code)]
pub fn f(x:f64, _t:usize)->f64 {
    x.tanh()
}


fn test_gpssm_pg(max_iter: &usize){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 500;
    let n_particles:usize = 500;
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;

    let (_x,y) = utils::_generate_data(&mut rng_mut, f, g, q, r, x0, seq_len);

    let now = SystemTime::now();
    let (_x_est, _gaussp) = gp_ssm(g, q, r, &y, x0, n_particles, *max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by GPSSM: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
}

#[allow(dead_code)]
pub fn time_gpssm_pg(){
    let mat_iter:[usize;4] = [1000, 5000, 10000, 20000];
    //let mat_iter:[usize;2] = [1000, 5000];
    for iter in mat_iter.iter(){
        test_gpssm_pg(&iter);
    }
}
