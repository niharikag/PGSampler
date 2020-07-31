use pyo3::prelude::*;

pub mod blocked_particle_gibbs;
pub mod utils;
pub mod particle_gibbs;
pub mod conditional_smc;
pub mod base_smc;
pub mod collapsed_particle_gibbs;
pub mod marg_csmc;

use crate::blocked_particle_gibbs::BlockedSMC;
use crate::collapsed_particle_gibbs::CollapsedPG;
use crate::particle_gibbs::PG;

#[pyfunction]
pub fn run_csmc(q: f64, r: f64, y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    n_particles:usize, anc_resamp: bool, max_iter:usize)-> PyResult<(Vec<f64>, f64)>{

    let f = utils::state_trans_func;
    let g = utils::transfer_func;
    let mut csmc = conditional_smc::CSMC::new(f, g, q, r, anc_resamp);

    let x_states: Vec<f64> = csmc.sample_states(&y, x0, &x_ref, n_particles, max_iter);
    let ll:f64 = csmc.get_log_likelihood();
    Ok((x_states, ll))
}

#[pyfunction]
pub fn run_bpf(q: f64, r: f64, y: Vec<f64>, x0: f64, n_particles:usize)-> PyResult<(Vec<f64>, f64)>{

    let f = utils::state_trans_func;
    let g = utils::transfer_func;
    let mut bsmc = base_smc::SMCBase::new(f, g, q, r);

    let x_states: Vec<f64> = bsmc.sample_states(&y, x0, n_particles);
    let ll:f64 = bsmc.get_log_likelihood();
    Ok((x_states, ll))
}

#[pyfunction]
fn run_blocked_smc(q: f64, r: f64, y: Vec<f64>, x0: f64, x_ref: Vec<f64>, pos_block:i32,
                          x_last:f64, start_time:usize, n_particles: usize)->PyResult<Vec<f64>> {
    let f = utils::_state_transition_func;
    let g: fn(f64) -> f64 = utils::_transfer_func;
    let mut b_smc: BlockedSMC = BlockedSMC::new(f, g, q, r);

    let x_states: Vec<f64> = b_smc.sample_states(&y, x0, &x_ref, pos_block, x_last,
                                                 start_time, n_particles);

    Ok(x_states)
}

#[pyfunction]
pub fn run_particle_gibbs(y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    q_init: f64, r_init: f64, prior_a: f64, prior_b: f64,
    n_particles:usize, anc_resamp: bool, max_iter:usize)-> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)>{

    let mut pg:PG = particle_gibbs::PG::new(utils::state_trans_func,
                                        utils::transfer_func,
                                        anc_resamp);

    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, max_iter);
    Ok((pg.get_states(), pg.get_q(), pg.get_r()))
}


#[pyfunction]
pub fn run_collapsed_pg(y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    q_init: f64, r_init: f64, prior_a: f64, prior_b: f64,
    n_particles:usize, max_iter:usize)-> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)>{

    let mut pg:CollapsedPG = CollapsedPG::new(utils::state_trans_func, utils::transfer_func);

    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, n_particles, max_iter);
    Ok((pg.get_states(), pg.get_q(), pg.get_r()))
}

#[pymodule]
fn rust_library(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "run_blocked_smc")]
    fn run_blocked_smc_py(q: f64, r: f64, y: Vec<f64>, x0: f64, x_ref: Vec<f64>, pos_block:i32,
                          x_last:f64, start_time:usize, n_particles: usize)->PyResult<Vec<f64>> {
        let out = run_blocked_smc(q, r, y, x0, x_ref,pos_block, x_last, start_time,n_particles);
        out
    }

    #[pyfn(m, "run_particle_gibbs")]
    fn run_particle_gibbs_py(y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    q_init: f64, r_init: f64, prior_a: f64, prior_b: f64,
    n_particles:usize, anc_resamp:bool, max_iter:usize) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {

        let out = run_particle_gibbs(y, x0, x_ref, q_init, r_init, prior_a, prior_b,
                                   n_particles, anc_resamp, max_iter);
        out
    }

    #[pyfn(m, "run_csmc")]
    pub fn run_csmc_py(q: f64, r: f64, y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    n_particles:usize, anc_resamp: bool, max_iter:usize)-> PyResult<(Vec<f64>, f64)> {
        let out = run_csmc(q, r, y, x0, x_ref, n_particles, anc_resamp, max_iter);

        out
    }

    #[pyfn(m, "run_bpf")]
    pub fn run_bpf_py(q: f64, r: f64, y: Vec<f64>, x0: f64, n_particles:usize)
        -> PyResult<(Vec<f64>, f64)>{
        let out = run_bpf(q, r, y, x0, n_particles);

        out
    }

    #[pyfn(m, "run_collapsed_pg")]
    pub fn run_collapsed_pg_py(y: Vec<f64>, x0: f64, x_ref: Vec<f64>,
    q_init: f64, r_init: f64, prior_a: f64, prior_b: f64,
    n_particles:usize, max_iter:usize)-> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)>{
        let out = run_collapsed_pg(y, x0, x_ref, q_init, r_init, prior_a, prior_b,
                                   n_particles, max_iter);
        out
    }
    Ok(())
}
