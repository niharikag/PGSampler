use crate::base_smc::SMCBase;
use crate::conditional_smc::CSMC;
use crate::utils;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rayon::prelude::*;

#[allow(dead_code)]
const N_NODES:usize = 16;
#[allow(dead_code)]
const N_NODES_SMC:usize = 8;
#[allow(dead_code)]
const N_NODES_CSMC:usize = 8;

type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;

struct SamplerResult{
    xref: Vec<f64>,
    ll: f64
}

#[allow(dead_code)]
impl SamplerResult {
    fn new() -> SamplerResult {
        SamplerResult {xref:Vec::new(), ll:0.0}
    }
    fn new_from(x:Vec<f64>, y:f64) -> SamplerResult {
        SamplerResult {xref:x, ll:y}
    }
}

impl Default for SamplerResult {
    #[inline]
    fn default() -> SamplerResult {
        SamplerResult {
            xref:Vec::new(),
            ll:0.0,
        }
    }
}

#[allow(dead_code)]
fn run_bpf(q: f64, r: f64, y: &Vec<f64>, x0: f64, n_particles: usize)->SamplerResult {
    let f = utils::state_trans_func;
    let g = utils::transfer_func;
    let mut bsmc: SMCBase = SMCBase::new(f, g, q, r);
    let x_states: Vec<f64> = bsmc.sample_states(&y, x0, n_particles);
    let x_ll:f64 = bsmc.get_log_likelihood();

    SamplerResult::new_from(x_states, x_ll)
}

#[allow(dead_code)]
fn run_csmc(q: f64, r: f64, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, n_particles: usize)->SamplerResult {
    let f: FDef = utils::state_trans_func;
    let g: GDef = utils::transfer_func;
    let anc_resamp: bool = false;
    let mut csmc: CSMC = CSMC::new(f, g, q, r, anc_resamp);

    let x_states: Vec<f64> = csmc.sample_states(&y, x0, x_ref, n_particles, 1);
    let x_ll:f64 = csmc.get_log_likelihood();

    SamplerResult::new_from(x_states, x_ll)
}

#[allow(dead_code)]
pub fn interacting_pg_sampler(q: f64, r: f64, y: &Vec<f64>, x0: f64, n_particles: usize,
                    _n_nodes:usize, max_iter:usize)->Vec<f64> {
    // fixed number of nodes as of now
    // let n_nodes_smc: usize = n_nodes/2;
    // let n_nodes_csmc: usize = n_nodes/2;
    let mut ll_smc = vec![0.0; N_NODES_SMC];
    let mut ll_csmc = vec![0.0; N_NODES_CSMC];
    let mut rng =  StdRng::from_entropy();
    let mut r_smc:[SamplerResult; N_NODES_SMC] = Default::default();
    let mut r_csmc:[SamplerResult; N_NODES_SMC] = Default::default();

    & mut r_csmc.par_iter_mut()
        .for_each(|x|
        {
           *x = run_bpf(q, r, y, x0, n_particles);
        });

    for _m in 0..max_iter {
        & mut r_smc.par_iter_mut()
            .for_each(|x|
            {
               *x = run_bpf(q, r, y, x0, n_particles);
            });

        & mut r_csmc.par_iter_mut()
            .for_each(|x|
            {
               *x = run_csmc(q, r, y, x0, &x.xref, n_particles);
            });

        for i in 0..N_NODES_SMC {
            ll_smc[i]  = r_smc[i].ll;
        }

        //TBD: run CSMC nodes in parallel
        for i in 0..N_NODES_CSMC {
            ll_csmc[i]  = r_csmc[i].ll;
        }

        let mut weights = ll_smc.clone();

        for i in 0..N_NODES_CSMC {
            weights.push(ll_csmc[i]);

            let w_sum:f64 = weights.iter().sum();

            // compute normalized weights
            let normalised_weights:Vec<f64> = weights.iter().map(|x| x /w_sum).collect();

            let index = utils::multinomial_resampling(& mut rng, &normalised_weights, &1_usize)[0];

            if index != N_NODES_CSMC{
                r_csmc[index].xref = r_smc[index].xref.clone();
                //x_refs_new.push(x_smc[index].clone());
            }
            weights.pop();
        }
    }

    let w_sum:f64 = ll_csmc.iter().sum();

    // compute normalized weights
    let normalised_weights:Vec<f64> = ll_csmc.iter().map(|x| x /w_sum).collect();
    let index = utils::multinomial_resampling(& mut rng, &normalised_weights, &1_usize)[0];
    r_csmc[index].xref.clone()
}