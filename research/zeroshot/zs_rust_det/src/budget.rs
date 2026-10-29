//! Port of zs_lane_v3/budget.py (revision 2): the face budget and its normal-aware quadric vertex clustering.
//! numpy's chunked bincount accumulations (2 000 000 faces per chunk), np.unique / lexsort first-occurrence
//! rules and the batched np.linalg.solve (dgesv per cluster) are reproduced, so the decimated mesh equals the
//! one Python writes (open3d writes it as float64 PLY, which reads back unchanged).
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use rayon::prelude::*;
use crate::num::*;

const CHUNK: usize = 2_000_000;
const LADDER: f64 = 1.12;
const MAX_STEPS: usize = 12;
const REG: f64 = 0.05;

pub struct Decimated {
    pub v: Vec<V3>,
    pub f: Vec<[u32; 3]>,
    pub info: serde_json::Value,
}

fn chunks(n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut s = 0;
    while s < n {
        out.push((s, (s + CHUNK).min(n)));
        s += CHUNK;
    }
    out
}

/// acc_j += np.bincount(idx, weights=w_j, minlength=len(acc_j)) for each j. bincount sums each bin's weights in
/// input order from +0.0; untouched bins would add +0.0, which leaves an accumulator unchanged (it never holds -0.0)
fn bincount_add(accs: &mut [&mut Vec<f64>], idx: &[u32], ws: &[&[f64]]) {
    if rayon::current_num_threads() == 1 {
        // one thread: bincount's own loop into a per-bin temporary (touched bins only), then acc += it
        let nb = accs[0].len();
        let mut tmp = vec![0.0f64; nb];
        let mut seen = vec![false; nb];
        let mut touched: Vec<u32> = Vec::new();
        for &b in idx {
            if !seen[b as usize] {
                seen[b as usize] = true;
                touched.push(b);
            }
        }
        for (j, w) in ws.iter().enumerate() {
            for (k, &b) in idx.iter().enumerate() {
                tmp[b as usize] += w[k];
            }
            let acc = &mut accs[j];
            for &b in &touched {
                acc[b as usize] += tmp[b as usize];
                tmp[b as usize] = 0.0;
            }
        }
        return;
    }
    // several threads: the bins are split into disjoint ranges and the item positions are bucketed by range in
    // input order (contiguous pieces, each kept in order); each range then runs bincount's loop over its own items,
    // so every bin sees its weights in input order exactly as in the sequential loop
    let nb = accs[0].len();
    let n = idx.len();
    let nt = rayon::current_num_threads();
    let parts = nt * 4;
    let step = ((nb + parts - 1) / parts).max(1);
    let nparts = (nb + step - 1) / step;
    let piece = ((n + nt - 1) / nt).max(1);
    let buckets: Vec<Vec<Vec<u32>>> = idx
        .par_chunks(piece)
        .enumerate()
        .map(|(p, c)| {
            let mut bk: Vec<Vec<u32>> = vec![Vec::new(); nparts];
            for (q, &b) in c.iter().enumerate() {
                bk[b as usize / step].push((p * piece + q) as u32);
            }
            bk
        })
        .collect();
    let nw = ws.len();
    let mut per_range: Vec<Vec<&mut [f64]>> = (0..nparts).map(|_| Vec::with_capacity(nw)).collect();
    for acc in accs.iter_mut() {
        for (r, ch) in acc.chunks_mut(step).enumerate() {
            per_range[r].push(ch);
        }
    }
    per_range.into_par_iter().enumerate().for_each(|(r, mut outs)| {
        let lo = r * step;
        let len = outs[0].len();
        let mut tmp = vec![0.0f64; len];
        let mut seen = vec![false; len];
        for bk in &buckets {
            for &k in &bk[r] {
                seen[idx[k as usize] as usize - lo] = true;
            }
        }
        for (j, w) in ws.iter().enumerate() {
            for bk in &buckets {
                for &k in &bk[r] {
                    tmp[idx[k as usize] as usize - lo] += w[k as usize];
                }
            }
            let out = &mut outs[j];
            for i in 0..len {
                if seen[i] {
                    out[i] += tmp[i];
                    tmp[i] = 0.0;
                }
            }
        }
    });
}

pub fn decimate(v: &[V3], f: &[[u32; 3]], max_faces: usize) -> Decimated {
    let nv = v.len();
    // _normals
    let tn = crate::prof::Timer::new("budget.normals");
    let mut col = [vec![0.0f64; nv], vec![0.0f64; nv], vec![0.0f64; nv]];
    for (s, e) in chunks(f.len()) {
        let cr: Vec<V3> = f[s..e]
            .par_iter()
            .map(|t| {
                let p0 = v[t[0] as usize];
                cross3(&sub3(&v[t[1] as usize], &p0), &sub3(&v[t[2] as usize], &p0))
            })
            .collect();
        let w: Vec<Vec<f64>> = (0..3).map(|ax| cr.par_iter().map(|c| c[ax]).collect()).collect();
        for k in 0..3 {
            let idx: Vec<u32> = f[s..e].par_iter().map(|t| t[k]).collect();
            let [c0, c1, c2] = &mut col;
            bincount_add(&mut [c0, c1, c2], &idx, &[&w[0], &w[1], &w[2]]);
        }
    }
    let nh: Vec<(V3, bool)> = (0..nv)
        .into_par_iter()
        .map(|i| {
            let c = [col[0][i], col[1][i], col[2][i]];
            let n = norm3(&c);
            let d = n.max(1e-30);
            ([c[0] / d, c[1] / d, c[2] / d], n > 0.0)
        })
        .collect();
    drop(col);
    let vn: Vec<V3> = nh.par_iter().map(|x| x.0).collect();
    let has: Vec<bool> = nh.par_iter().map(|x| x.1).collect();
    drop(nh);
    let nbin: Vec<i64> = (0..nv)
        .into_par_iter()
        .map(|i| {
            if !has[i] {
                return 0;
            }
            let a = [vn[i][0].abs(), vn[i][1].abs(), vn[i][2].abs()];
            let mut dom = 0;
            for k in 1..3 {
                if a[k] > a[dom] {
                    dom = k;
                }
            }
            (dom as i64) * 2 + if vn[i][dom] < 0.0 { 1 } else { 0 }
        })
        .collect();
    drop(tn);
    let tg = crate::prof::Timer::new("budget.grid");
    let ne = f.len().min(200_000);
    let el: Vec<f64> = f[..ne].iter().map(|t| norm3(&sub3(&v[t[1] as usize], &v[t[0] as usize]))).collect();
    let edge = mean(&el);
    let mut h = edge * (1.0f64.max(f.len() as f64 / max_faces as f64)).sqrt();
    let mut lo = [f64::INFINITY; 3];
    for p in v {
        for k in 0..3 {
            if p[k] < lo[k] {
                lo[k] = p[k];
            }
        }
    }
    let mut steps = Vec::new();
    let mut inv: Vec<u32> = Vec::new();
    let mut ncl = 0usize;
    let mut g: Vec<[u32; 3]> = Vec::new();
    for _ in 0..MAX_STEPS {
        let gk: Vec<[i64; 3]> = v
            .par_iter()
            .map(|p| [((p[0] - lo[0]) / h).floor() as i64, ((p[1] - lo[1]) / h).floor() as i64, ((p[2] - lo[2]) / h).floor() as i64])
            .collect();
        let m: Vec<i64> = (0..3).map(|k| gk.par_iter().map(|x| x[k]).max().unwrap_or(0) + 1).collect();
        let keys: Vec<i64> = (0..nv).into_par_iter().map(|i| (((gk[i][0] * m[1]) + gk[i][1]) * m[2] + gk[i][2]) * 6 + nbin[i]).collect();
        let mut uk = keys.clone();
        uk.par_sort_unstable();
        uk.dedup();
        inv = keys.par_iter().map(|k| uk.binary_search(k).unwrap() as u32).collect();
        ncl = uk.len();
        // _clustered_faces
        let gg: Vec<[u32; 3]> = f
            .par_iter()
            .filter_map(|t| {
                let c = [inv[t[0] as usize], inv[t[1] as usize], inv[t[2] as usize]];
                if c[0] != c[1] && c[1] != c[2] && c[0] != c[2] { Some(c) } else { None }
            })
            .collect();
        let mut order: Vec<usize> = (0..gg.len()).collect();
        let srt: Vec<[u32; 3]> = gg
            .par_iter()
            .map(|c| {
                let mut s = *c;
                s.sort_unstable();
                s
            })
            .collect();
        order.par_sort_by(|&a, &b| srt[a].cmp(&srt[b])); // stable: first of equal rows keeps its rank
        let mut keep_idx: Vec<usize> = Vec::new();
        for (r, &o) in order.iter().enumerate() {
            if r == 0 || srt[o] != srt[order[r - 1]] {
                keep_idx.push(o);
            }
        }
        keep_idx.sort_unstable();
        g = keep_idx.iter().map(|&i| gg[i]).collect();
        steps.push(serde_json::json!([crate::z2graph::py_round(h, 5), ncl, g.len()]));
        if g.len() <= max_faces {
            break;
        }
        h *= LADDER;
    }
    drop(tg);
    let tq = crate::prof::Timer::new("budget.quadrics");
    let c = ncl;
    let mut cnt = vec![0.0f64; c];
    let mut sx = vec![0.0f64; c];
    let mut sy = vec![0.0f64; c];
    let mut sz = vec![0.0f64; c];
    {
        let px: Vec<f64> = v.par_iter().map(|p| p[0]).collect();
        let py: Vec<f64> = v.par_iter().map(|p| p[1]).collect();
        let pz: Vec<f64> = v.par_iter().map(|p| p[2]).collect();
        let ones = vec![1.0f64; nv];
        bincount_add(&mut [&mut cnt, &mut sx, &mut sy, &mut sz], &inv, &[&ones, &px, &py, &pz]);
    }
    let mean_c: Vec<V3> = (0..c).into_par_iter().map(|k| [sx[k] / cnt[k], sy[k] / cnt[k], sz[k] / cnt[k]]).collect();
    drop((sx, sy, sz));
    let mut acc: Vec<Vec<f64>> = vec![vec![0.0f64; c]; 9];
    for (s, e) in chunks(f.len()) {
        let fs = &f[s..e];
        let tv: Vec<[f64; 9]> = fs
            .par_iter()
            .map(|t| {
                let p0 = v[t[0] as usize];
                let cr = cross3(&sub3(&v[t[1] as usize], &p0), &sub3(&v[t[2] as usize], &p0));
                let w = norm3(&cr);
                let wd = w.max(1e-30);
                let n = [cr[0] / wd, cr[1] / wd, cr[2] / wd];
                let d = n[2].dfma(p0[2], n[0] * p0[0]) + n[1] * p0[1];
                [
                    w * n[0] * n[0], w * n[0] * n[1], w * n[0] * n[2], w * n[1] * n[1], w * n[1] * n[2], w * n[2] * n[2],
                    w * n[0] * d, w * n[1] * d, w * n[2] * d,
                ]
            })
            .collect();
        let cols: Vec<Vec<f64>> = (0..9).map(|j| tv.par_iter().map(|x| x[j]).collect()).collect();
        drop(tv);
        let wr: Vec<&[f64]> = cols.iter().map(|x| x.as_slice()).collect();
        for k in 0..3 {
            let cl: Vec<u32> = fs.par_iter().map(|t| inv[t[k] as usize]).collect();
            let mut ar: Vec<&mut Vec<f64>> = acc.iter_mut().collect();
            bincount_add(&mut ar, &cl, &wr);
        }
    }
    let b: Vec<Vec<f64>> = acc.split_off(6);
    let a = acc;
    let xs: Vec<(V3, bool)> = (0..c).into_par_iter().map(|k| {
        let lam = REG * ((a[0][k] + a[3][k]) + a[5][k]) / 3.0 + 1e-12;
        let rhs = [b[0][k] + lam * mean_c[k][0], b[1][k] + lam * mean_c[k][1], b[2][k] + lam * mean_c[k][2]];
        let m = [
            a[0][k] + lam, a[1][k], a[2][k],
            a[1][k], a[3][k] + lam, a[4][k],
            a[2][k], a[4][k], a[5][k] + lam,
        ];
        let sol = accel::solve(&m, &rhs, 3).unwrap_or_else(|| vec![f64::NAN; 3]);
        let xi = [sol[0], sol[1], sol[2]];
        if norm3(&sub3(&xi, &mean_c[k])) > h { (mean_c[k], true) } else { (xi, false) }
    }).collect();
    drop(tq);
    let nfar = xs.iter().filter(|x| x.1).count();
    let x: Vec<V3> = xs.into_iter().map(|x| x.0).collect();
    let info = serde_json::json!({
        "faces_in": f.len(), "vertices_in": nv, "faces_out": g.len(), "vertices_out": c,
        "cell_mm": crate::z2graph::py_round(h, 5), "mean_edge_mm": crate::z2graph::py_round(edge, 5), "ladder": steps,
        "clamped_frac": crate::z2graph::py_round(nfar as f64 / c as f64, 5)
    });
    Decimated { v: x, f: g, info }
}

// ================================================================================================================
// Revision 3 (zs_lane_v3 budget.py at fingerprint dc23a905...): the density-invariant fixed-grid resampling
// (grid-line crossings -> one QEF vertex per (cell, skin) -> one quad per crossing). Revision 2's decimate above
// stays as it was; `apply` picks the revision from the config (budget.cell_mm present = revision 3).
// ================================================================================================================

/// The budget step's settings, from the lane config: `cell_mm` = None runs revision 2, Some(c) revision 3.
#[derive(Clone, Debug)]
pub struct BudgetCfg {
    pub max_faces: usize,
    pub cell_mm: Option<f64>,
}

/// The one entry point of the budget step (lane.rs and the selftest call only this).
pub fn apply(v: &[V3], f: &[[u32; 3]], cfg: &BudgetCfg) -> Decimated {
    match cfg.cell_mm {
        None => decimate(v, f, cfg.max_faces),
        Some(c) => decimate_rev3(v, f, cfg.max_faces, c),
    }
}

const R3_LADDER: f64 = 1.12;
const R3_MAX_STEPS: usize = 24;
const R3_SKIN_COS: f64 = -0.5;
const R3_PERM: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];
const R3_QUAD: [(i64, i64); 4] = [(-1, -1), (0, -1), (0, 0), (-1, 0)];

/// one crossing; `rank` = its position in Python's concatenation order (axis, chunk, ob, oc, face in chunk), the
/// tie-break that makes the parallel sort equal to numpy's stable lexsort
#[derive(Clone, Copy)]
struct Crossing {
    a: u8,
    bb: i64,
    cc: i64,
    g: i64,
    p: V3,
    n: V3,
    rank: (u32, u32, u32, u32),
}

#[inline]
fn fl(x: f64, h: f64) -> i64 {
    (x / h).floor() as i64
}

/// the accepted crossings of one face with the grid lines parallel to axis a (Python's ob / oc loops for this face)
#[inline]
fn face_crossings(v: &[V3], t: &[u32; 3], a: usize, h: f64, mut emit: impl FnMut(i64, i64, f64, f64, f64, u32, u32)) {
    let (b, c) = R3_PERM[a];
    let (p0, p1, p2) = (&v[t[0] as usize], &v[t[1] as usize], &v[t[2] as usize]);
    let gb = [fl(p0[b], h), fl(p1[b], h), fl(p2[b], h)];
    let gc = [fl(p0[c], h), fl(p1[c], h), fl(p2[c], h)];
    let (lo_b, hi_b) = (gb[0].min(gb[1]).min(gb[2]), gb[0].max(gb[1]).max(gb[2]));
    let (lo_c, hi_c) = (gc[0].min(gc[1]).min(gc[2]), gc[0].max(gc[1]).max(gc[2]));
    if !(hi_b > lo_b && hi_c > lo_c) {
        return;
    }
    let (sb, sc) = (hi_b - lo_b, hi_c - lo_c);
    let x0b = p0[b];
    let x0c = p0[c];
    let e1b = p1[b] - x0b;
    let e1c = p1[c] - x0c;
    let e2b = p2[b] - x0b;
    let e2c = p2[c] - x0c;
    let den0 = e1b * e2c - e2b * e1c;
    if !(den0.abs() > 1e-18) {
        return;
    }
    let den = den0;
    for ob in 1..=sb {
        for oc in 1..=sc {
            let bv = lo_b + ob;
            let cv = lo_c + oc;
            let qb = bv as f64 * h;
            let qc = cv as f64 * h;
            let l1 = ((qb - x0b) * e2c - e2b * (qc - x0c)) / den;
            let l2 = (e1b * (qc - x0c) - (qb - x0b) * e1c) / den;
            let l0 = 1.0 - l1 - l2;
            if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                emit(bv, cv, l0, l1, l2, ob as u32, oc as u32);
            }
        }
    }
}

fn r3_count(v: &[V3], f: &[[u32; 3]], h: f64) -> usize {
    (0..3usize)
        .map(|a| {
            f.par_iter()
                .map(|t| {
                    let mut n = 0usize;
                    face_crossings(v, t, a, h, |_, _, _, _, _, _, _| n += 1);
                    n
                })
                .sum::<usize>()
        })
        .sum()
}

fn r3_crossings(v: &[V3], f: &[[u32; 3]], h: f64) -> Vec<Crossing> {
    let mut all: Vec<Crossing> = Vec::new();
    for a in 0..3usize {
        let (b, c) = R3_PERM[a];
        let part: Vec<Crossing> = f
            .par_iter()
            .enumerate()
            .flat_map_iter(|(fi, t)| {
                let mut out: Vec<Crossing> = Vec::new();
                let (p0, p1, p2) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
                face_crossings(v, t, a, h, |bv, cv, l0, l1, l2, ob, oc| {
                    let mut p = [0.0f64; 3];
                    for k in 0..3 {
                        p[k] = (l0 * p0[k] + l1 * p1[k]) + l2 * p2[k];
                    }
                    p[b] = bv as f64 * h;
                    p[c] = cv as f64 * h;
                    let cr = cross3(&sub3(&p1, &p0), &sub3(&p2, &p0));
                    let d = norm3(&cr).max(1e-30);
                    let n = [cr[0] / d, cr[1] / d, cr[2] / d];
                    let chunk = (fi / CHUNK) as u32;
                    out.push(Crossing { a: a as u8, bb: bv, cc: cv, g: fl(p[a], h), p, n, rank: (chunk, ob, oc, (fi % CHUNK) as u32) });
                });
                out
            })
            .collect();
        all.extend(part);
    }
    // numpy lexsort keys (a, B, C, G, P, N), compared with < (so -0.0 == 0.0), then Python's concatenation order
    let fcmp = |x: f64, y: f64| x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal);
    all.par_sort_unstable_by(|x, y| {
        x.a.cmp(&y.a)
            .then(x.bb.cmp(&y.bb))
            .then(x.cc.cmp(&y.cc))
            .then(x.g.cmp(&y.g))
            .then(fcmp(x.p[0], y.p[0]))
            .then(fcmp(x.p[1], y.p[1]))
            .then(fcmp(x.p[2], y.p[2]))
            .then(fcmp(x.n[0], y.n[0]))
            .then(fcmp(x.n[1], y.n[1]))
            .then(fcmp(x.n[2], y.n[2]))
            .then(x.rank.cmp(&y.rank))
    });
    all
}

/// resample(cx, h): (X, T, info fields crossings / vertices_out / faces_out / clamped_frac)
fn r3_resample(cx: &[Crossing], h: f64) -> (Vec<V3>, Vec<[u32; 3]>, usize, f64) {
    let n = cx.len();
    let nr = 4 * n;
    let cell = |r: usize| -> [i64; 3] {
        let x = &cx[r / 4];
        let (b, c) = R3_PERM[x.a as usize];
        let (db, dc) = R3_QUAD[r % 4];
        let mut k = [0i64; 3];
        k[x.a as usize] = x.g;
        k[b] = x.bb + db;
        k[c] = x.cc + dc;
        k
    };
    let mut lo = [i64::MAX; 3];
    let mut hi = [i64::MIN; 3];
    for r in 0..nr {
        let k = cell(r);
        for d in 0..3 {
            lo[d] = lo[d].min(k[d]);
            hi[d] = hi[d].max(k[d]);
        }
    }
    let dim = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1];
    let ckey: Vec<i64> = (0..nr)
        .into_par_iter()
        .map(|r| {
            let k = cell(r);
            ((k[0] - lo[0]).wrapping_mul(dim[1]).wrapping_add(k[1] - lo[1])).wrapping_mul(dim[2]).wrapping_add(k[2] - lo[2])
        })
        .collect();
    // first row of each cell (np.unique return_index: the smallest flattened index)
    let mut ord: Vec<(i64, u32)> = ckey.par_iter().enumerate().map(|(r, &k)| (k, r as u32)).collect();
    ord.par_sort_unstable();
    let mut first_of = vec![0u32; nr];
    {
        let mut s = 0usize;
        while s < nr {
            let mut e = s + 1;
            while e < nr && ord[e].0 == ord[s].0 {
                e += 1;
            }
            let f0 = ord[s].1;
            for q in s..e {
                first_of[ord[q].1 as usize] = f0;
            }
            s = e;
        }
    }
    drop(ord);
    let vkey: Vec<i64> = (0..nr)
        .into_par_iter()
        .map(|r| {
            let nr_ = &cx[r / 4].n;
            let nf = &cx[first_of[r] as usize / 4].n;
            let skin = if crate::z2seg::einsum3(nr_, nf) < R3_SKIN_COS { 1 } else { 0 };
            ckey[r].wrapping_mul(2).wrapping_add(skin)
        })
        .collect();
    drop(first_of);
    drop(ckey);
    let mut uk = vkey.clone();
    uk.par_sort_unstable();
    uk.dedup();
    let c = uk.len();
    let vinv: Vec<u32> = vkey.par_iter().map(|k| uk.binary_search(k).unwrap() as u32).collect();
    drop(vkey);
    drop(uk);
    // QEF sums in flattened row order (bincount)
    let col = |f: &(dyn Fn(&Crossing) -> f64 + Sync)| -> Vec<f64> { (0..nr).into_par_iter().map(|r| f(&cx[r / 4])).collect() };
    let ones = vec![1.0f64; nr];
    let px = col(&|x| x.p[0]);
    let py = col(&|x| x.p[1]);
    let pz = col(&|x| x.p[2]);
    let mut cnt = vec![0.0f64; c];
    let mut sx = vec![0.0f64; c];
    let mut sy = vec![0.0f64; c];
    let mut sz = vec![0.0f64; c];
    bincount_add(&mut [&mut cnt, &mut sx, &mut sy, &mut sz], &vinv, &[&ones, &px, &py, &pz]);
    drop((ones, px, py, pz));
    let mean: Vec<V3> = (0..c).into_par_iter().map(|k| [sx[k] / cnt[k], sy[k] / cnt[k], sz[k] / cnt[k]]).collect();
    drop((sx, sy, sz, cnt));
    let dd = |x: &Crossing| crate::z2seg::einsum3(&x.n, &x.p);
    let wcols: Vec<Vec<f64>> = vec![
        col(&|x| x.n[0] * x.n[0]),
        col(&|x| x.n[0] * x.n[1]),
        col(&|x| x.n[0] * x.n[2]),
        col(&|x| x.n[1] * x.n[1]),
        col(&|x| x.n[1] * x.n[2]),
        col(&|x| x.n[2] * x.n[2]),
        col(&|x| x.n[0] * dd(x)),
        col(&|x| x.n[1] * dd(x)),
        col(&|x| x.n[2] * dd(x)),
    ];
    let mut acc: Vec<Vec<f64>> = vec![vec![0.0f64; c]; 9];
    {
        let wr: Vec<&[f64]> = wcols.iter().map(|x| x.as_slice()).collect();
        let mut ar: Vec<&mut Vec<f64>> = acc.iter_mut().collect();
        bincount_add(&mut ar, &vinv, &wr);
    }
    drop(wcols);
    let xs: Vec<(V3, bool)> = (0..c)
        .into_par_iter()
        .map(|k| {
            let a = |j: usize| acc[j][k];
            let lam = REG * ((a(0) + a(3)) + a(5)) / 3.0 + 1e-12;
            let rhs = [a(6) + lam * mean[k][0], a(7) + lam * mean[k][1], a(8) + lam * mean[k][2]];
            let m = [a(0) + lam, a(1), a(2), a(1), a(3) + lam, a(4), a(2), a(4), a(5) + lam];
            let sol = accel::solve(&m, &rhs, 3).unwrap_or_else(|| vec![f64::NAN; 3]);
            let xi = [sol[0], sol[1], sol[2]];
            if norm3(&sub3(&xi, &mean[k])) > h { (mean[k], true) } else { (xi, false) }
        })
        .collect();
    drop(acc);
    let nfar = xs.iter().filter(|x| x.1).count();
    let x: Vec<V3> = xs.into_iter().map(|x| x.0).collect();
    // faces: one quad per crossing, oriented by the triangle normal along the axis, split on the shorter diagonal
    let tt: Vec<[u32; 3]> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut q = [vinv[4 * i], vinv[4 * i + 1], vinv[4 * i + 2], vinv[4 * i + 3]];
            if cx[i].n[cx[i].a as usize] < 0.0 {
                q = [q[3], q[2], q[1], q[0]];
            }
            let d02 = norm3(&sub3(&x[q[0] as usize], &x[q[2] as usize]));
            let d13 = norm3(&sub3(&x[q[1] as usize], &x[q[3] as usize]));
            if d02 <= d13 { [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] } else { [[q[0], q[1], q[3]], [q[1], q[2], q[3]]] }
        })
        .filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2])
        .collect();
    let srt: Vec<[u32; 3]> = tt
        .par_iter()
        .map(|t| {
            let mut s = *t;
            s.sort_unstable();
            s
        })
        .collect();
    let mut order: Vec<usize> = (0..tt.len()).collect();
    order.par_sort_by(|&a, &b| srt[a].cmp(&srt[b]));
    let mut keep: Vec<usize> = Vec::new();
    for (r, &o) in order.iter().enumerate() {
        if r == 0 || srt[o] != srt[order[r - 1]] {
            keep.push(o);
        }
    }
    keep.par_sort_unstable();
    let t: Vec<[u32; 3]> = keep.iter().map(|&i| tt[i]).collect();
    let clamped = if c > 0 { crate::z2graph::py_round(nfar as f64 / c as f64, 5) } else { 0.0 };
    (x, t, n, clamped)
}

pub fn decimate_rev3(v: &[V3], f: &[[u32; 3]], max_faces: usize, cell_mm: f64) -> Decimated {
    let hk = |k: usize| cell_mm * std::hint::black_box(R3_LADDER).dpowf(std::hint::black_box(k as f64));
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    let mut count = |k: usize, counts: &mut std::collections::BTreeMap<usize, usize>| -> usize {
        *counts.entry(k).or_insert_with(|| crate::prof::time("budget.r3_count", || r3_count(v, f, hk(k))))
    };
    let n0 = count(0, &mut counts);
    let mut k = if 2 * n0 <= max_faces {
        0
    } else {
        let r = ((2 * n0) as f64 / max_faces as f64).dln() / (2.0 * std::hint::black_box(R3_LADDER).dln());
        (R3_MAX_STEPS as f64).min(r.ceil()) as usize
    };
    while k < R3_MAX_STEPS && 2 * count(k, &mut counts) > max_faces {
        k += 1;
    }
    while k > 0 && 2 * count(k - 1, &mut counts) <= max_faces {
        k -= 1;
    }
    let h = hk(k);
    let cx = crate::prof::time("budget.r3_crossings", || r3_crossings(v, f, h));
    let steps: Vec<serde_json::Value> = counts
        .iter()
        .map(|(&j, &cj)| serde_json::json!([crate::z2graph::py_round(hk(j), 5), cj, 2 * cj]))
        .collect();
    let (x, t, ncx, clamped) = crate::prof::time("budget.r3_resample", || r3_resample(&cx, h));
    let info = serde_json::json!({
        "faces_in": f.len(), "vertices_in": v.len(), "faces_out": t.len(), "vertices_out": x.len(),
        "cell_mm": crate::z2graph::py_round(h, 5), "crossings": ncx, "ladder": steps, "clamped_frac": clamped
    });
    Decimated { v: x, f: t, info }
}
