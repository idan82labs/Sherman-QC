//! Rectangular linear sum assignment: a port of scipy's rectangular_lsap.cpp (Crouse's shortest augmenting
//! path), including its tie-breaking, so assignments equal scipy.optimize.linear_sum_assignment.

fn augmenting_path(
    nc: usize,
    cost: &[f64],
    u: &[f64],
    v: &[f64],
    path: &mut [isize],
    row4col: &[isize],
    spc: &mut [f64],
    mut i: usize,
    sr: &mut [bool],
    sc: &mut [bool],
    remaining: &mut [usize],
) -> Option<(usize, f64)> {
    let mut min_val = 0.0f64;
    let mut num_remaining = nc;
    for it in 0..nc {
        remaining[it] = nc - it - 1;
    }
    sr.iter_mut().for_each(|x| *x = false);
    sc.iter_mut().for_each(|x| *x = false);
    spc.iter_mut().for_each(|x| *x = f64::INFINITY);
    let mut sink: isize = -1;
    while sink == -1 {
        let mut index: isize = -1;
        let mut lowest = f64::INFINITY;
        sr[i] = true;
        for it in 0..num_remaining {
            let j = remaining[it];
            let r = min_val + cost[i * nc + j] - u[i] - v[j];
            if r < spc[j] {
                path[j] = i as isize;
                spc[j] = r;
            }
            if spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1) {
                lowest = spc[j];
                index = it as isize;
            }
        }
        min_val = lowest;
        if min_val == f64::INFINITY {
            return None;
        }
        let j = remaining[index as usize];
        if row4col[j] == -1 {
            sink = j as isize;
        } else {
            i = row4col[j] as usize;
        }
        sc[j] = true;
        num_remaining -= 1;
        remaining[index as usize] = remaining[num_remaining];
    }
    Some((sink as usize, min_val))
}

/// Returns (row indices, col indices) like scipy (rows ascending when not transposed).
pub fn linear_sum_assignment(nr0: usize, nc0: usize, cost0: &[f64]) -> (Vec<usize>, Vec<usize>) {
    if nr0 == 0 || nc0 == 0 {
        return (vec![], vec![]);
    }
    let transpose = nc0 < nr0;
    let (nr, nc, cost): (usize, usize, Vec<f64>) = if transpose {
        let mut t = vec![0.0; nr0 * nc0];
        for i in 0..nr0 {
            for j in 0..nc0 {
                t[j * nr0 + i] = cost0[i * nc0 + j];
            }
        }
        (nc0, nr0, t)
    } else {
        (nr0, nc0, cost0.to_vec())
    };
    let mut u = vec![0.0f64; nr];
    let mut v = vec![0.0f64; nc];
    let mut spc = vec![0.0f64; nc];
    let mut path = vec![-1isize; nc];
    let mut col4row = vec![-1isize; nr];
    let mut row4col = vec![-1isize; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];
    for cur_row in 0..nr {
        let (sink, min_val) = match augmenting_path(
            nc, &cost, &u, &v, &mut path, &row4col, &mut spc, cur_row, &mut sr, &mut sc, &mut remaining,
        ) {
            Some(x) => x,
            None => return (vec![], vec![]),
        };
        u[cur_row] += min_val;
        for i in 0..nr {
            if sr[i] && i != cur_row {
                u[i] += min_val - spc[col4row[i] as usize];
            }
        }
        for j in 0..nc {
            if sc[j] {
                v[j] -= min_val - spc[j];
            }
        }
        let mut j = sink as isize;
        loop {
            let i = path[j as usize];
            row4col[j as usize] = i;
            std::mem::swap(&mut col4row[i as usize], &mut j);
            if i as usize == cur_row {
                break;
            }
        }
    }
    if transpose {
        let mut order: Vec<usize> = (0..nr).collect();
        order.sort_by_key(|&k| col4row[k]);
        let a: Vec<usize> = order.iter().map(|&k| col4row[k] as usize).collect();
        let b: Vec<usize> = order.clone();
        (a, b)
    } else {
        ((0..nr).collect(), col4row.iter().map(|&x| x as usize).collect())
    }
}
