use crate::base_smc::SMCBase;
use crate::conditional_smc::CSMC;
use crate::utils;
use rand::{SeedableRng};
use rand::rngs::StdRng;



struct Result(Vec<f64>, f64);
type FDef = fn(&Vec<f64>, usize) -> Vec<f64>;
type GDef = fn(&Vec<f64>) -> Vec<f64>;


#[allow(dead_code)]
fn run_bpf(q: f64, r: f64, y: &Vec<f64>, x0: f64, n_particles: usize)->Result {
    let f = utils::state_trans_func;
    let g = utils::transfer_func;
    let mut bsmc: SMCBase = SMCBase::new(f, g, q, r);
    let x_states: Vec<f64> = bsmc.sample_states(&y, x0, n_particles);
    let x_ll:f64 = bsmc.get_log_likelihood();

    Result(x_states, x_ll)
}

#[allow(dead_code)]
fn run_csmc(q: f64, r: f64, y: &Vec<f64>, x0: f64, x_ref: &Vec<f64>, n_particles: usize)->(Vec<f64>, f64) {
    let f: FDef = utils::state_trans_func;
    let g: GDef = utils::transfer_func;
    let anc_resamp: bool = false;
    let mut csmc: CSMC = CSMC::new(f, g, q, r, anc_resamp);

    let x_states: Vec<f64> = csmc.sample_states(&y, x0, x_ref, n_particles, 1);
    let x_ll:f64 = csmc.get_log_likelihood();

    (x_states, x_ll)
}

#[allow(dead_code)]
pub fn interacting_pg_sampler(q: f64, r: f64, y: &Vec<f64>, x0: f64, n_particles: usize,
                    n_nodes:usize, max_iter:usize)->Vec<f64> {

    let n_nodes_smc: usize = n_nodes/2;
    let n_nodes_csmc: usize = n_nodes/2;
    let mut ll_smc = vec![0.0; n_nodes_smc];
    let mut ll_csmc = vec![0.0; n_nodes_csmc];
    let mut x_refs: Vec<Vec<f64>> = Vec::new();
    let mut rng =  StdRng::from_entropy();

    //TBD: run SMC nodes in parallel
    for _i in 0..n_nodes_csmc {
        let r_smc = run_bpf(q, r, y, x0, n_particles);
        x_refs.push(r_smc.0);
    }

    for _m in 0..max_iter {
        let mut x_smc: Vec<Vec<f64>> = Vec::new();
        let mut x_csmc: Vec<Vec<f64>> = Vec::new();
        //TBD: run SMC nodes in parallel
        for i in 0..n_nodes_smc {
            let r_smc = run_bpf(q, r, y, x0, n_particles);
            x_smc.push(r_smc.0);
            ll_smc[i]  = r_smc.1;
        }

        //TBD: run CSMC nodes in parallel
        for i in 0..n_nodes_csmc {
            let r_smc = run_csmc(q, r, y, x0, &x_refs[i], n_particles);
            x_csmc.push(r_smc.0);
            ll_csmc[i]  = r_smc.1;
        }

        let mut weights = ll_smc.clone();
        let mut x_refs_new: Vec<Vec<f64>> = Vec::new();

        for i in 0..n_nodes_csmc {
            weights.push(ll_csmc[i]);

            let w_sum:f64 = weights.iter().sum();

            // compute normalized weights
            let normalised_weights:Vec<f64> = weights.iter().map(|x| x /w_sum).collect();

            let index = utils::multinomial_resampling(& mut rng, &normalised_weights, &1_usize)[0];

            if index == n_nodes_csmc{
                x_refs_new.push(x_csmc[i].clone());
            }
            else {
                x_refs_new.push(x_smc[index].clone());
            }
            weights.pop();
        }
        x_refs = x_refs_new;
    }

    let w_sum:f64 = ll_csmc.iter().sum();

    // compute normalized weights
    let normalised_weights:Vec<f64> = ll_csmc.iter().map(|x| x /w_sum).collect();
    let index = utils::multinomial_resampling(& mut rng, &normalised_weights, &1_usize)[0];
    x_refs[index].clone()
}