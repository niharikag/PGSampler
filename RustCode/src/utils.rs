use rand::Rng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
//use random_choice::random_choice;
//use rand::seq::index::sample;


#[allow(dead_code)]
pub fn binary_search(arr: &[f64], target: &f64) -> usize
{
    let mut size = arr.len();
    if size == 0 {
        return 0_usize;
    }
    let mut base = 0_usize;

    while size > 1 {
        // mid: [base..size)
        let half = size / 2;
        let mid = base + half;
        if arr[mid] <= *target {
            base = mid
        }
        size -= half;
    }

    if arr[base] == *target {
        base
    } else {
        // Return the expected position in the array.
        base + (arr[base] < *target) as usize
    }
}

#[allow(dead_code)]
pub fn multinomial_resampling(_rng:&mut StdRng, w: &[f64], size: &usize)->Vec<usize> {
    let n: usize;
    if *size == 0_usize {
        n = w.len();
    }
    else {
        n = *size;
    }

    /*
    let cumsum: Vec<f64> = w.iter().scan(0.0, |acc, &x| {
            *acc = *acc + x;
            Some(*acc)
        }).collect();

    let mut res:Vec<usize> = vec![0; n];

    for i in 0..n {
        let u:f64 = rng.gen();
        res[i] = binary_search(&cumsum, &u);
    }
     */

    // it works
    let w_len = w.len();

    let mut res:Vec<usize> = vec![0; n];
    let samples:Vec<usize> = (0..w_len).collect();
    let choices = random_choice::random_choice().random_choice_f64(&samples, &w, n);
     for i in 0..n {
        res[i] = *choices[i];
    }

    res
}


pub fn _systematic_resampling(rng:&mut StdRng, w: &Vec<f64>, size: usize) ->Vec<usize> {
    let n: usize;
    if size == 0 {
        n = w.len();
    }
    else {
        n = size;
    }

    // compute cumulative sum
    let cumsum: Vec<f64> = w.iter().scan(0.0, |acc, &x| {
            *acc = *acc + x;
            Some(*acc)
        }).collect();

    let mut u: f64;
    let mut res: Vec<usize> = vec![0; n];

    let u_rand: f64 = rng.gen();

    for i in 0..n {
        u = (i as f64 + u_rand) / (n as f64);
        let location = cumsum.binary_search_by(|v| { v.partial_cmp(&u)
            .expect("Couldn't compare values") });

        match location {
            Ok(location) => res[i] = location,
            Err(location) => res[i] = location,
        }
    }

    res
}


pub fn _stratified_resampling(rng:&mut StdRng, w: &Vec<f64>, size: usize) ->Vec<usize> {
    let n: usize;
    if size == 0 {
        n = w.len();
    }
    else {
        n = size;
    }

    let cumsum: Vec<f64> = w.iter().scan(0.0, |acc, &x| {
            *acc = *acc + x;
            Some(*acc)
        }).collect();

    let mut u: f64;
    let mut u_rand:f64;
    let mut res:Vec<usize> = vec![0; n];

    for i in 0..n {
        u_rand = rng.gen();
        u =(i as f64 + u_rand)/( n as f64) ;
        let location = cumsum.binary_search_by(|v| {v.partial_cmp(&u)
            .expect("Couldn't compare values")});

        match location{
             Ok(location) => res[i] = location,
             Err(location) => res[i] = location,
         }
    }

    res
}

// transfer function that takes input as vector of f64
#[inline]
pub fn transfer_func(x:&Vec<f64>)->Vec<f64> {
    let out:Vec<f64> = x.iter().map(|x|
    x.powi(2)/20.0).collect();
    out
}

// transfer function that takes input as f64
pub fn _transfer_func(x:f64)->f64 {
    x.powi(2)/20.0
}

// state transition function that takes input as vector of f64
#[inline]
pub fn state_trans_func(x:&Vec<f64>, t:usize)->Vec<f64> {
    let time_t:f64 = t as f64+1.0;
    let out:Vec<f64> = x.iter().map(|x|
    0.5*x + 25.0*x/(1.0+ x.powi(2)) + 8.0*(1.2*time_t).cos()).collect();
    out
}

// state transition function that takes input as f64
pub fn _state_transition_func(x:f64, t:usize)->f64 {
    let time_t:f64 = t as f64+1.0;
    0.5*x + 25.0*x/(1.0+ x.powi(2)) + 8.0*(1.2*time_t).cos()
}

pub fn _generate_data(rng:&mut StdRng, f: fn(f64, usize) -> f64, g: fn(f64) -> f64, q:f64, r:f64,
                      x0:f64, seq_len:usize) ->( Vec<f64>,  Vec<f64>){
    let mut x: Vec<f64> = vec![0.0; seq_len];
    let mut y: Vec<f64> = vec![0.0; seq_len];
    let sd_val: f64 = rng.sample(StandardNormal);
    x[0] = x0;
    //let mut sd_val: f64 = thread_rng().sample(StandardNormal);
    y[0] = g(x[0]) + r.sqrt() * sd_val;

    for t in 1..seq_len as usize{
        let sd_val: f64 = rng.sample(StandardNormal);
        x[t] = f(x[t - 1], t-1) + q.sqrt() * sd_val;
        let rn_val: f64 = rng.sample(StandardNormal);
        y[t] = g(x[t]) + r.sqrt() * rn_val;
    }
    (x, y)
}


pub fn _ess(x: &Vec<f64>) ->f64 {
    let mut _cloned = ndarray::arr1(&x);
    let temp = _cloned.dot(&_cloned);
    1.0/temp
}