use crate::particle_gibbs::PG;
use crate::utils;
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::file_utils::write_to_file_append;

#[allow(dead_code)]
static SEED: [u8; 32] = [0;32];

#[allow(dead_code)]
fn run_pg(n_particles: &usize){
    let mut rng_mut:StdRng = SeedableRng::from_seed(SEED);
    let seq_len:usize = 100;
    //let n_particles:usize = 100;
    let max_iter:usize = 1000;
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

    pg.generate_samples(&y, x0, &x_ref, q_init, r_init, prior_a, prior_b, *n_particles, max_iter);

    let q_est: Vec<f64> = pg.get_q();
    //plots::plot_hist(&q_est);
    write_to_file_append(&q_est, "output/pg_q.txt".to_string(), true).expect("error");
    let r_est: Vec<f64> = pg.get_r();
    write_to_file_append(&r_est, "output/pg_r.txt".to_string(), true).expect("error");
}

#[allow(dead_code)]
pub fn acf_pg(){
    let n_particles:[usize;4] = [10, 50, 100, 200];
    for n in n_particles.iter(){
        run_pg(n);
    }
}
