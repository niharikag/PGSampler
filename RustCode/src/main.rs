mod acf_plot;
mod time_comparison;
mod constants;
mod gp_ssm;
mod marg_csmc;
mod collapsed_particle_gibbs;
mod blocked_particle_gibbs;
mod interacting_particle_gibbs;
mod conditional_smc;
mod particle_gibbs;
mod plots;
mod base_smc;
mod utils;
mod file_utils;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::{SystemTime};
use itertools_num::linspace;

use crate::base_smc::SMCBase;
use crate::conditional_smc::CSMC;
use crate::particle_gibbs::PG;
use crate::plots::{plot_err_line};
use crate::file_utils::write_to_file;
use crate::interacting_particle_gibbs::interacting_pg_sampler;
use crate::blocked_particle_gibbs::iterate_blocked_pg;
use crate::marg_csmc::{MargCSMC, Priors};
use crate::collapsed_particle_gibbs::CollapsedPG;
use crate::gp_ssm::gp_ssm;
use rusty_machine::learning::SupModel;
use rusty_machine::linalg::{Matrix};


//static SEED: [u8; 32] = [1;32];
#[allow(dead_code)]
static SEED: [u8; 32] = [0;32];

#[allow(dead_code)]
pub fn g(x:f64)->f64 {
    x
}

#[allow(dead_code)]
pub fn f(x:f64, _t:usize)->f64 {
    x.tanh()
}

#[allow(dead_code)]
fn _test_gp_ssm(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    let n_particles:usize = 100;
    let max_iter:usize = 5000;
    let x0:f64 = 1.0;
    let q:f64 = 0.1;
    let r:f64 = 0.1;

    let (x,y) = utils::_generate_data(&mut rng_mut, f,
                                      g, q, r, x0, seq_len);
    //println!("{:?}",x);

    let now = SystemTime::now();
    let (x_est,gaussp) = gp_ssm(g, q, r, &y, x0,
                                n_particles, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by cSMC: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
   }
    //println!("data:x {:?}", x);
    //println!("x est {:?}", x_est);
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i];
    }
    //plot_err_line(&_x_err);
    let x_input:Vec<f64> =  linspace::<f64>(-1.0, 1.0, 100).collect();
    //let x_new:Vec<f64>= x_input.iter().map(|x| x.tanh() ).collect();
    //plot_a_line(&x_input, &x_new);
    let test_input = Matrix::new(100,1, x_input.to_vec());
    let x_predited = gaussp.predict(&test_input).unwrap();

    //plot_a_line(&x_input, &x_predited.into_vec());
    write_to_file(&x_predited.into_vec(), "output/gpssm_x.txt".to_string());
}


fn _test_collapsed_pg(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    let n_particles:usize = 100;
    let max_iter:usize = 10000;
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let r_init:f64 = 0.1;
    let q_init:f64 = 1.0;
    let prior_a:f64 = 0.01;
    let prior_b:f64 = 0.01;

    let (x,y) = utils::_generate_data(&mut rng_mut,
                                      utils::_state_transition_func,
                                      utils::_transfer_func, q, r, x0, seq_len);

    let mut pg:CollapsedPG = CollapsedPG::new(utils::state_trans_func,utils::transfer_func);

    let now = SystemTime::now();
    pg.generate_samples(&y, 0.0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by PG: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
    let x_est: Vec<f64> = pg.get_states();
    let q_est: Vec<f64> = pg.get_q();
    //plots::plot_hist(&q_est);
    write_to_file(&q_est, "output/collapsed_pg_q.txt".to_string());
    let r_est: Vec<f64> = pg.get_r();
    write_to_file(&r_est, "output/collapsed_pg_r.txt".to_string());

    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i];
    }
    plot_err_line(&_x_err);
}


fn _test_marg_csmc(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    let n_particles:usize = 100;
    let max_iter:usize = 10000;
    let x0:f64 = 0.0;

    let (x,y) = utils::_generate_data(&mut rng_mut,utils::_state_transition_func,
                                      utils::_transfer_func, 0.1, 0.1, x0, seq_len);
    let p = Priors(1.0,1.0,1.0,1.0);
    let mut csmc:MargCSMC = MargCSMC::new(utils::state_trans_func,utils::transfer_func,
                                          p);
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let now = SystemTime::now();
    let x_est = csmc.sample_states(&y, x0, &x_ref, n_particles, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by marg cSMC: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
   }
    //println!("data:x {:?}", x);
    //println!("x est {:?}", x_est);
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i];
    }
    plot_err_line(&_x_err);
}


fn _test_blocked_pg(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    let n_particles:usize = 50;
    let block_size:usize = 30;
    let n_overlap:usize = 1;
    let max_iter:usize = 1000;
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let (x,y) = utils::_generate_data(&mut rng_mut,utils::_state_transition_func,
                                      utils::_transfer_func, 0.1, 0.1, 0.0, seq_len);

    let mut csmc:CSMC = conditional_smc::CSMC::new(utils::state_trans_func,utils::transfer_func,
                                                      0.1, 0.1, false);
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let x_est: Vec<f64> = csmc.sample_states(&y, 0.0, &x_ref, n_particles, max_iter);
    let now = SystemTime::now();

    let x_blocked_pg =  iterate_blocked_pg(q, r, &y, x0, &x_est, n_particles,
                                           block_size, n_overlap, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by iterate_blocked_pg: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_blocked_pg[i];
    }
    plot_err_line(&_x_err);

}


fn _test_base_smc(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let n_particles:usize = 200;
    let seq_len:usize = 100;
    let (x,y) = utils::_generate_data(&mut rng_mut,utils::_state_transition_func,
                                      utils::_transfer_func, q, r, x0, seq_len);

    let mut bsmc:SMCBase = base_smc::SMCBase::new(utils::state_trans_func,
                                                  utils::transfer_func, q, r);

    let x_est:Vec<f64> = bsmc.sample_states(&y, x0, n_particles);
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i]
    }
    //plot_lines(&x, &x_est);
    plot_err_line(&_x_err);
    let ll:f64 = bsmc.get_log_likelihood();
    println!("log likelihood: {}", ll);
}


fn _test_csmc(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 500;
    let n_particles:usize = 500;
    let max_iter:usize = 1000;
    let (x,y) = utils::_generate_data(&mut rng_mut,utils::_state_transition_func,
                                      utils::_transfer_func, 0.1, 0.1, 0.0, seq_len);

    let mut csmc:CSMC = conditional_smc::CSMC::new(utils::state_trans_func,utils::transfer_func,
                                                      0.1, 0.1, false);
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let now = SystemTime::now();
    let x_est = csmc.sample_states(&y, 0.0, &x_ref, n_particles, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by cSMC: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
   }
    //println!("data:x {:?}", x);
    //println!("x est {:?}", x_est);
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i];
    }
    plot_err_line(&_x_err);
    let ll:f64 = csmc.get_log_likelihood();
    println!("log likelihood: {}", ll);
}


fn _test_i_pg(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    let n_particles:usize = 100;
    let max_iter:usize = 1000;
    //let x_ref: Vec<f64> = vec![0.0; seq_len];
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;

    let (_x,y) = utils::_generate_data(&mut rng_mut,utils::_state_transition_func,
                                      utils::_transfer_func, 0.1, 0.1, 0.0, seq_len);

    let now = SystemTime::now();
    let x_est:Vec<f64> = interacting_pg_sampler(q, r, &y, x0, n_particles,4, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by cSMC: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = _x[i] - x_est[i]
    }
    //plot_lines(&x, &x_est);
    plot_err_line(&_x_err);
}


fn _test_pg(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 500;
    let n_particles:usize = 500;
    let max_iter:usize = 50000;
    let x_ref: Vec<f64> = vec![0.0; seq_len];
    let q:f64 = 0.1;
    let r:f64 = 1.0;
    let x0:f64 = 0.0;
    let r_init:f64 = 0.1;
    let q_init:f64 = 1.0;
    let prior_a:f64 = 0.01;
    let prior_b:f64 = 0.01;
    let anc_resamp:bool = false;

    let (x,y) = utils::_generate_data(&mut rng_mut,
                                      utils::_state_transition_func,
                                      utils::_transfer_func, q, r, x0, seq_len);

    let mut pg:PG = particle_gibbs::PG::new(utils::state_trans_func,
                                            utils::transfer_func,
                                            anc_resamp);
    let now = SystemTime::now();
    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, max_iter);
    match now.elapsed() {
       Ok(elapsed) => {
           println!("time taken by PG: {}", elapsed.as_secs());
       }
       Err(e) => {
           // an error occurred!
           println!("Error: {:?}", e);
       }
    }
    let x_est: Vec<f64> = pg.get_states();
    let q_est: Vec<f64> = pg.get_q();
    //plots::plot_hist(&q_est);
    write_to_file(&q_est, "output/pg_q.txt".to_string());
    let r_est: Vec<f64> = pg.get_r();
    write_to_file(&r_est, "output/pg_r.txt".to_string());

    let mut _x_err:Vec<f64> = vec![0.0;seq_len];

    for i in 0..seq_len{
        _x_err[i] = x[i] - x_est[i];
    }
    plot_err_line(&_x_err);
}


fn _test_utils(){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let s:Vec<f64> =  Vec::from([0.4, 0.3, 0.2, 0.1, 0.1]);
    let res_multi = utils::multinomial_resampling(&mut rng_mut, &s, &0_usize);
    let res_systematic = utils::_systematic_resampling(&mut rng_mut, &s, 0);
    let res_stratified = utils::_stratified_resampling(&mut rng_mut, &s, 0);
    println!("res multi {:?}", res_multi);
    println!("res systematic {:?}", res_systematic);
    println!("res stratified {:?}", res_stratified);
    let ess_value = utils::_ess(&s);
    println!("ess {}", ess_value);
    //println!("s original vector {:?}", s);
}


fn main() {
    //_test_utils();
    //_test_base_smc();
    //_test_csmc();
    //_test_pg();
    //_test_i_pg();
    //_test_blocked_pg();
    //_test_marg_csmc();
    //_test_collapsed_pg();
    _test_gp_ssm();
    //time_comparison::time_pg();
    //acf_plot::acf_pg();
    //time_comparison::time_collapsed_pg();
    //time_comparison::time_gpssm_pg();
    println!("hello!");
}
