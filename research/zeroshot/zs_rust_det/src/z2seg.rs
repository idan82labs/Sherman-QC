//! Port of z2_self_recipe/segment.py from the kNN graph on (normal smoothing, seeds, seed release, plane fits,
//! plane-guided growth, coplanar merge, purification, region geometry, strip merge, skin pairing into flanges),
//! with E3 geom.robust_plane / fit_plane. Loading, the D1 thickness probe, the random thinning and the kNN query
//! stay in Python (inputs here).
//!
//! numpy evaluation orders reproduced: einsum over a length-3 axis = fma(a2, b2, a0 * b0) + a1 * b1; a dot with a
//! strided 3-vector (an eigenvector column view) = (p0 + p2) + p1, contiguous = (p0 + p1) + p2; (n, 3) @ (3,) with
//! n > 1 is gemv = (p0 + p1) + p2 whatever the strides, with n == 1 it is a dot; X.T @ X and eigh go through
//! Accelerate (accel.rs). Each Plane carries whether its normal is numpy's strided eigenvector view.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use crate::kdtree::KdTree;
use crate::num::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize, Clone, Debug)]
pub struct SegParams {
    pub smooth_iters: usize,
    pub seed_deg: f64,
    pub flat_deg: f64,
    pub seed_offset_mm: f64,
    pub min_seed_points: usize,
    pub grow_deg: f64,
    pub grow_dist_mm: f64,
    pub grow_iters: usize,
    pub merge_deg: f64,
    pub merge_dist_mm: f64,
    pub min_region_area_mm2: f64,
    pub pair_deg: f64,
    pub wall_width_t: f64,
    pub seed_release_deg: f64,
}

#[derive(Clone, Debug)]
pub struct Plane {
    pub n: V3,
    pub strided: bool,
    pub d: f64,
    pub rms: f64,
    pub count: i64,
    pub c: Option<V3>,
}

#[inline]
pub fn einsum3(a: &V3, b: &V3) -> f64 {
    a[2].dfma(b[2], a[0] * b[0]) + a[1] * b[1]
}

#[inline]
pub fn dotn(a: &V3, sa: bool, b: &V3, sb: bool) -> f64 {
    if sa || sb {
        dot3_strided(a, b)
    } else {
        dot3(a, b)
    }
}

impl Plane {
    pub fn flipped(&self) -> Plane {
        Plane { n: neg3(&self.n), strided: false, d: -self.d, rms: self.rms, count: self.count, c: self.c }
    }
    /// Plane.distance(pts): pts @ n - d (gemv, or a dot for a single point)
    pub fn distance_one(&self, p: &V3, single: bool) -> f64 {
        if single {
            dotn(p, false, &self.n, self.strided) - self.d
        } else {
            dot3(p, &self.n) - self.d
        }
    }
    /// Plane.offset(delta): n.copy() (contiguous), d + delta, centroid + delta * n
    pub fn offset(&self, delta: f64) -> Plane {
        Plane { n: self.n, strided: false, d: self.d + delta, rms: self.rms, count: self.count,
                c: self.c.map(|c| [c[0] + delta * self.n[0], c[1] + delta * self.n[1], c[2] + delta * self.n[2]]) }
    }
    /// Plane.distance(pts) for several points (gemv, or a dot when there is one point)
    pub fn distances(&self, pts: &[V3]) -> Vec<f64> {
        let single = pts.len() == 1;
        pts.iter().map(|p| self.distance_one(p, single)).collect()
    }
    pub fn to_json(&self) -> Value {
        json!({"n": self.n.to_vec(), "strided": self.strided, "d": self.d, "rms": self.rms, "count": self.count,
               "centroid": self.c.map(|c| c.to_vec())})
    }
}

fn gemv_mean(ns: &[V3], idx: &[usize], n: &V3) -> f64 {
    // float(np.mean(Ns[idx] @ n)): gemv rows (or a dot for one row), pairwise mean
    if idx.len() == 1 {
        return dot3(&ns[idx[0]], n); // contiguous row . n (n contiguous or strided handled by caller)
    }
    let v: Vec<f64> = idx.iter().map(|&i| dot3(&ns[i], n)).collect();
    mean(&v)
}

fn gemv_mean_pl(ns: &[V3], idx: &[usize], pl: &Plane) -> f64 {
    if idx.len() == 1 {
        return dotn(&ns[idx[0]], false, &pl.n, pl.strided);
    }
    gemv_mean(ns, idx, &pl.n)
}

/// E3 geom.fit_plane(p, w)
fn fit_plane(p: &[V3], w: Option<&[f64]>) -> Option<Plane> {
    let n = p.len();
    if n < 3 {
        return None;
    }
    let ones;
    let w: &[f64] = match w {
        Some(x) => x,
        None => {
            ones = vec![1.0f64; n];
            &ones
        }
    };
    let sw = pw_sum(w);
    if !(sw > 0.0) {
        return None;
    }
    // c = (p * w[:, None]).sum(axis=0) / sw  (sequential over rows)
    let mut c = [0.0f64; 3];
    for i in 0..n {
        c[0] += p[i][0] * w[i];
        c[1] += p[i][1] * w[i];
        c[2] += p[i][2] * w[i];
    }
    let c = [c[0] / sw, c[1] / sw, c[2] / sw];
    let mut q = Vec::with_capacity(n * 3);
    for i in 0..n {
        let s = w[i].sqrt();
        q.push((p[i][0] - c[0]) * s);
        q.push((p[i][1] - c[1]) * s);
        q.push((p[i][2] - c[2]) * s);
    }
    let mut cov = accel::xtx(&q, n, 3);
    for x in cov.iter_mut() {
        *x /= sw;
    }
    let (_vals, vecs) = accel::eigh(&cov, 3);
    let mut nv = [vecs[0], vecs[3], vecs[6]];
    let mut strided = true;
    let mut k = 0;
    for j in 1..3 {
        if nv[j].abs() > nv[k].abs() {
            k = j;
        }
    }
    if nv[k] < 0.0 {
        nv = neg3(&nv);
        strided = false;
    }
    let d = dotn(&nv, strided, &c, false);
    // res = (p - c) @ n ; rms = sqrt(max(0, (w * res * res).sum() / sw))
    let rr: Vec<f64> = if n == 1 {
        vec![]
    } else {
        (0..n)
            .map(|i| {
                let r = dot3(&sub3(&p[i], &c), &nv);
                (w[i] * r) * r
            })
            .collect()
    };
    let rms = 0.0f64.max(pw_sum(&rr) / sw).sqrt();
    Some(Plane { n: nv, strided, d, rms, count: n as i64, c: Some(c) })
}

/// E3 geom.robust_plane(pts, iterations)
pub fn robust_plane(p: &[V3], iterations: usize) -> Option<Plane> {
    let scale_floor = 0.02;
    let cutoff = 3.0;
    let mut plane = fit_plane(p, None)?;
    let single = p.len() == 1;
    for _ in 0..iterations {
        let r: Vec<f64> = p.iter().map(|q| plane.distance_one(q, single)).collect();
        let med = median(&r);
        let ad: Vec<f64> = r.iter().map(|x| (x - med).abs()).collect();
        let mad = median(&ad) * 1.4826;
        let s = mad.max(scale_floor);
        let w: Vec<f64> = r
            .iter()
            .map(|x| {
                let u = (x - med) / (cutoff * s);
                if u.abs() < 1.0 {
                    let y = 1.0 - u * u;
                    y * y
                } else {
                    0.0
                }
            })
            .collect();
        if pw_sum(&w) < 3.0 {
            break;
        }
        let mut new = match fit_plane(p, Some(&w)) {
            Some(x) => x,
            None => break,
        };
        if dotn(&new.n, new.strided, &plane.n, plane.strided) < 0.0 {
            new = new.flipped();
        }
        plane = new;
    }
    let r: Vec<f64> = p.iter().map(|q| plane.distance_one(q, single)).collect();
    let ar: Vec<f64> = r.iter().map(|x| x.abs()).collect();
    let lim = cutoff * scale_floor.max(median(&ar) * 1.4826);
    let inl: Vec<f64> = r.iter().filter(|x| x.abs() < lim).map(|x| x * x).collect();
    plane.rms = if !inl.is_empty() {
        mean(&inl).sqrt()
    } else {
        let all: Vec<f64> = r.iter().map(|x| x * x).collect();
        mean(&all).sqrt()
    };
    plane.count = inl.len() as i64;
    Some(plane)
}

pub struct Ctx<'a> {
    pub p: &'a [V3],
    pub nbr: &'a [u32],
    pub k: usize,
}

fn unit_rows(v: &V3) -> V3 {
    let n = norm3(v).max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn smooth_normals(n_in: &[V3], nbr: &[u32], k: usize, iters: usize) -> Vec<V3> {
    let mut ns = n_in.to_vec();
    for _ in 0..iters {
        let prev = ns.clone();
        ns = (0..prev.len())
            .into_par_iter()
            .map(|i| {
                let me = prev[i];
                let mut acc = [0.0f64; 3];
                for j in 0..k {
                    let nb = prev[nbr[i * k + j] as usize];
                    let w = if einsum3(&nb, &me) > 0.5 { 1.0 } else { 0.0 };
                    acc[0] += w * nb[0];
                    acc[1] += w * nb[1];
                    acc[2] += w * nb[2];
                }
                unit_rows(&[me[0] + acc[0], me[1] + acc[1], me[2] + acc[2]])
            })
            .collect();
    }
    ns
}

/// _components(ok, nbr, mask, min_pts, offset): labels (or -1)
/// roots of the edge graph's components, each root = the component's smallest node (any union order gives the
/// same partition, so the concurrent version equals the sequential one)
fn component_roots_par(ok: &[bool], nbr: &[u32], k: usize, mask: &[bool]) -> Vec<u32> {
    use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
    let n = mask.len();
    let parent: Vec<AtomicU32> = (0..n as u32).map(AtomicU32::new).collect();
    let find = |mut x: u32| -> u32 {
        loop {
            let p = parent[x as usize].load(Relaxed);
            if p == x {
                return x;
            }
            let gp = parent[p as usize].load(Relaxed);
            if gp != p {
                let _ = parent[x as usize].compare_exchange(p, gp, Relaxed, Relaxed);
            }
            x = gp;
        }
    };
    (0..n).into_par_iter().for_each(|i| {
        if !mask[i] {
            return;
        }
        for j in 0..k {
            let e = i * k + j;
            let t = nbr[e] as usize;
            if !(ok[e] && mask[t]) {
                continue;
            }
            let (mut a, mut b) = (i as u32, t as u32);
            loop {
                a = find(a);
                b = find(b);
                if a == b {
                    break;
                }
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                if parent[hi as usize].compare_exchange(hi, lo, Relaxed, Relaxed).is_ok() {
                    break;
                }
            }
        }
    });
    (0..n as u32).into_par_iter().map(|i| find(i)).collect()
}

fn components(ok: &[bool], nbr: &[u32], k: usize, mask: &[bool], min_pts: usize, offset: i64) -> Vec<i64> {
    if rayon::current_num_threads() > 1 {
        let root = component_roots_par(ok, nbr, k, mask);
        return label_components(&root, mask, min_pts, offset);
    }
    let n = mask.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for i in 0..n {
        if !mask[i] {
            continue;
        }
        for j in 0..k {
            let e = i * k + j;
            let t = nbr[e] as usize;
            if ok[e] && mask[t] {
                let a = find(&mut parent, i as u32);
                let b = find(&mut parent, t as u32);
                if a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    parent[hi as usize] = lo;
                }
            }
        }
    }
    // component label = rank of the component's smallest node (scipy connected_components numbering)
    let mut root = vec![0u32; n];
    for i in 0..n {
        root[i] = find(&mut parent, i as u32);
    }
    label_components(&root, mask, min_pts, offset)
}

fn label_components(root: &[u32], mask: &[bool], min_pts: usize, offset: i64) -> Vec<i64> {
    let n = mask.len();
    let mut size = vec![0usize; n];
    for i in 0..n {
        size[root[i] as usize] += 1;
    }
    let mut lab_of_root = vec![-1i64; n];
    let mut next = offset;
    // uniq over kept labels in comp order: comp order == order of smallest node == first occurrence by index
    let mut out = vec![-1i64; n];
    for i in 0..n {
        let r = root[i] as usize;
        if mask[i] && size[r] >= min_pts {
            if lab_of_root[r] < 0 {
                lab_of_root[r] = next;
                next += 1;
            }
            out[i] = lab_of_root[r];
        }
    }
    out
}

fn groups_of(label: &[i64], m: usize) -> Vec<Vec<usize>> {
    let mut g = vec![Vec::new(); m];
    for (i, &l) in label.iter().enumerate() {
        if l >= 0 && (l as usize) < m {
            g[l as usize].push(i);
        }
    }
    g
}

fn ceil_div_f(a: usize, b: usize) -> usize {
    // int(math.ceil(a / b)) with Python true division
    ((a as f64) / (b as f64)).ceil() as usize
}

fn fit_planes(p: &[V3], label: &[i64], m: usize) -> Vec<Option<Plane>> {
    let g = groups_of(label, m);
    g.par_iter()
        .map(|idx| {
            if idx.len() < 3 {
                return None;
            }
            let sub: Vec<V3> = if idx.len() <= 20000 {
                idx.iter().map(|&i| p[i]).collect()
            } else {
                let st = ceil_div_f(idx.len(), 20000);
                idx.iter().step_by(st).map(|&i| p[i]).collect()
            };
            robust_plane(&sub, 3)
        })
        .collect()
}

fn orient(planes: &mut [Option<Plane>], ns: &[V3], label: &[i64]) {
    let g = groups_of(label, planes.len());
    for (r, pl) in planes.iter_mut().enumerate() {
        if let Some(p) = pl {
            let idx: Vec<usize> = g[r].iter().take(2000).cloned().collect();
            if !idx.is_empty() && gemv_mean_pl(ns, &idx, p) < 0.0 {
                *p = p.flipped();
            }
        }
    }
}

fn grow(ctx: &Ctx, ns: &[V3], label0: &[i64], planes: &[Option<Plane>], prm: &SegParams) -> Vec<i64> {
    let m = planes.len();
    let k = ctx.k;
    let mut nvec: Vec<V3> = planes.iter().map(|p| p.as_ref().map(|x| x.n).unwrap_or([0.0; 3])).collect();
    let mut dvec: Vec<f64> = planes.iter().map(|p| p.as_ref().map(|x| x.d).unwrap_or(0.0)).collect();
    let g = groups_of(label0, m);
    for r in 0..m {
        if planes[r].is_none() {
            continue;
        }
        let idx: Vec<usize> = g[r].iter().take(2000).cloned().collect();
        // nvec rows are contiguous copies (np.array of the normals)
        if !idx.is_empty() && gemv_mean(ns, &idx, &nvec[r]) < 0.0 {
            nvec[r] = neg3(&nvec[r]);
            dvec[r] = -dvec[r];
        }
    }
    let cos_g = prm.grow_deg.to_radians().dcos();
    let mut label = label0.to_vec();
    for _ in 0..prm.grow_iters {
        let un: Vec<usize> = (0..label.len()).filter(|&i| label[i] < 0).collect();
        if un.is_empty() || m == 0 {
            break;
        }
        let upd: Vec<(usize, i64)> = un
            .par_iter()
            .filter_map(|&i| {
                let nl: Vec<i64> = (0..k).map(|j| label[ctx.nbr[i * k + j] as usize]).collect();
                if !nl.iter().any(|&x| x >= 0) {
                    return None;
                }
                let mut best = -1i64;
                let mut bestd = f64::INFINITY;
                for &r in &nl {
                    let v = r >= 0;
                    let rr = if r < 0 { 0 } else { r as usize };
                    let dist = (einsum3(&ctx.p[i], &nvec[rr]) - dvec[rr]).abs();
                    let cs = einsum3(&ns[i], &nvec[rr]);
                    if v && dist < prm.grow_dist_mm && cs > cos_g && dist < bestd {
                        best = r;
                        bestd = dist;
                    }
                }
                if best >= 0 { Some((i, best)) } else { None }
            })
            .collect();
        if upd.is_empty() {
            break;
        }
        for (i, r) in upd {
            label[i] = r;
        }
    }
    label
}

fn purify(ctx: &Ctx, ns: &[V3], label: &mut Vec<i64>, planes: Vec<Option<Plane>>, prm: &SegParams) -> Vec<Option<Plane>> {
    let cos_g = prm.grow_deg.to_radians().dcos();
    let mut changed = false;
    let g = groups_of(label, planes.len());
    for (r, pl) in planes.iter().enumerate() {
        let pl = match pl {
            Some(p) => p,
            None => continue,
        };
        let idx = &g[r];
        if idx.is_empty() {
            continue;
        }
        let head: Vec<usize> = idx.iter().take(2000).cloned().collect();
        let (nv, nstr) = if gemv_mean_pl(ns, &head, pl) >= 0.0 { (pl.n, pl.strided) } else { (neg3(&pl.n), false) };
        let good: Vec<bool> = if idx.len() == 1 {
            vec![dotn(&ns[idx[0]], false, &nv, nstr) > cos_g]
        } else {
            idx.iter().map(|&i| dot3(&ns[i], &nv) > cos_g).collect()
        };
        let gm = mean_bool(&good);
        if gm < 0.5 {
            for &i in idx {
                label[i] = -1;
            }
            changed = true;
        } else if !good.iter().all(|&x| x) {
            for (j, &i) in idx.iter().enumerate() {
                if !good[j] {
                    label[i] = -1;
                }
            }
            changed = true;
        }
    }
    let mut planes = planes;
    if changed {
        let m = planes.len();
        planes = fit_planes(ctx.p, label, m);
        orient(&mut planes, ns, label);
        *label = grow(ctx, ns, label, &planes, prm);
        planes = fit_planes(ctx.p, label, m);
        orient(&mut planes, ns, label);
    }
    planes
}

fn merge(ctx: &Ctx, label: &mut Vec<i64>, planes: Vec<Option<Plane>>, prm: &SegParams) -> Vec<Option<Plane>> {
    let m = planes.len();
    if m == 0 {
        return planes;
    }
    let k = ctx.k;
    let n = ctx.p.len();
    let mut keys: Vec<i64> = Vec::new();
    for i in 0..n {
        let li = label[i];
        for j in 0..k {
            let lj = label[ctx.nbr[i * k + j] as usize];
            if li >= 0 && lj >= 0 && li != lj {
                keys.push(li.min(lj) * m as i64 + li.max(lj));
            }
        }
    }
    keys.sort_unstable();
    let mut kc: Vec<(i64, usize)> = Vec::new();
    for key in keys {
        match kc.last_mut() {
            Some((kk, c)) if *kk == key => *c += 1,
            _ => kc.push((key, 1)),
        }
    }
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let cos_m = prm.merge_deg.to_radians().dcos();
    let g = groups_of(label, m);
    let cents: Vec<Option<V3>> = (0..m)
        .map(|r| {
            if planes[r].is_some() && !g[r].is_empty() {
                let s = seq_sum_rows3(g[r].iter().map(|&i| ctx.p[i]));
                let c = g[r].len() as f64;
                Some([s[0] / c, s[1] / c, s[2] / c])
            } else {
                None
            }
        })
        .collect();
    for (key, c) in kc {
        let x = (key / m as i64) as usize;
        let y = (key % m as i64) as usize;
        let (px, py) = match (&planes[x], &planes[y]) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if c < 3 {
            continue;
        }
        let cs = dotn(&px.n, px.strided, &py.n, py.strided);
        if cs < cos_m {
            continue;
        }
        let (cx, cy) = match (cents[x], cents[y]) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        if px.distance_one(&cy, true).abs() > prm.merge_dist_mm || py.distance_one(&cx, true).abs() > prm.merge_dist_mm {
            continue;
        }
        let rx = find(&mut parent, x);
        let ry = find(&mut parent, y);
        if rx != ry {
            parent[ry] = rx;
        }
    }
    let roots: Vec<usize> = (0..m).map(|x| find(&mut parent, x)).collect();
    let mut uniq = roots.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let mut newid = vec![0i64; m];
    for x in 0..m {
        newid[x] = uniq.binary_search(&roots[x]).unwrap() as i64;
    }
    for l in label.iter_mut() {
        if *l >= 0 {
            *l = newid[*l as usize];
        }
    }
    fit_planes(ctx.p, label, uniq.len())
}

/// _region_geometry: (width, length)
fn region_geometry(pts: &[V3], pl: &Plane) -> (f64, f64) {
    let n = pts.len();
    let s = seq_sum_rows3(pts.iter().cloned());
    let c = [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64];
    let single = n == 1;
    let mut q = Vec::with_capacity(n * 3);
    for p in pts {
        let x = sub3(p, &c);
        let h = if single { dotn(&x, false, &pl.n, pl.strided) } else { dot3(&x, &pl.n) };
        q.push(x[0] - h * pl.n[0]);
        q.push(x[1] - h * pl.n[1]);
        q.push(x[2] - h * pl.n[2]);
    }
    if n < 3 {
        return (0.0, 0.0);
    }
    let mut cov = accel::xtx(&q, n, 3);
    for x in cov.iter_mut() {
        *x /= n as f64;
    }
    let (_v, vecs) = accel::eigh(&cov, 3);
    let em = [vecs[2], vecs[5], vecs[8]];
    let en = [vecs[1], vecs[4], vecs[7]];
    let mut a: Vec<f64> = (0..n).map(|i| dot3(&[q[3 * i], q[3 * i + 1], q[3 * i + 2]], &em)).collect();
    let mut b: Vec<f64> = (0..n).map(|i| dot3(&[q[3 * i], q[3 * i + 1], q[3 * i + 2]], &en)).collect();
    a.sort_by(total_cmp);
    b.sort_by(total_cmp);
    let length = quantiles_sorted(&a, 0.99) - quantiles_sorted(&a, 0.01);
    let width = quantiles_sorted(&b, 0.99) - quantiles_sorted(&b, 0.01);
    (width, length)
}

#[derive(Clone)]
pub struct Region {
    pub rid: usize,
    pub idx: Vec<usize>,
    pub plane: Plane,
    pub area: f64,
    pub width: f64,
    pub length: f64,
    pub flange: i64,
}

/// _strip_axis(Q, n): principal in-plane axis (strided eigenvector view)
fn strip_axis(q: &[V3], pl: &Plane) -> (V3, V3) {
    let n = q.len();
    let s = seq_sum_rows3(q.iter().cloned());
    let c = [s[0] / n as f64, s[1] / n as f64, s[2] / n as f64];
    let single = n == 1;
    let mut x = Vec::with_capacity(n * 3);
    for p in q {
        let r = sub3(p, &c);
        let h = if single { dotn(&r, false, &pl.n, pl.strided) } else { dot3(&r, &pl.n) };
        x.push(r[0] - h * pl.n[0]);
        x.push(r[1] - h * pl.n[1]);
        x.push(r[2] - h * pl.n[2]);
    }
    let m = accel::xtx(&x, n, 3);
    let (_w, v) = accel::eigh(&m, 3);
    (c, [v[2], v[5], v[8]])
}

fn merge_strips(p: &[V3], ns: &[V3], regions: Vec<Region>, t: f64) -> Vec<Region> {
    let narrow: Vec<usize> = (0..regions.len()).filter(|&i| regions[i].width < 4.0 * t + 2.0).collect();
    if narrow.len() < 2 {
        return regions;
    }
    // info: (Q.mean(0), _strip_axis(Q, n), Q); the mean Q.mean(0) equals strip_axis's centroid computation
    let info: Vec<(V3, V3, Vec<V3>)> = narrow
        .iter()
        .map(|&i| {
            let q: Vec<V3> = regions[i].idx.iter().map(|&j| p[j]).collect();
            let (c, u) = strip_axis(&q, &regions[i].plane);
            (c, u, q)
        })
        .collect();
    let nn = narrow.len();
    let mut parent: Vec<usize> = (0..nn).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let c10 = 10.0f64.to_radians().dcos();
    let c5 = 5.0f64.to_radians().dcos();
    for ia in 0..nn {
        let a = &regions[narrow[ia]];
        let (ca, ua, qa) = &info[ia];
        let za: Vec<f64> = if qa.len() == 1 {
            vec![dot3_strided(&sub3(&qa[0], ca), ua)]
        } else {
            qa.iter().map(|q| dot3(&sub3(q, ca), ua)).collect()
        };
        for ib in (ia + 1)..nn {
            let b = &regions[narrow[ib]];
            if dotn(&a.plane.n, a.plane.strided, &b.plane.n, b.plane.strided) < c10 {
                continue;
            }
            let (cb, ub, qb) = &info[ib];
            if dot3_strided(ua, ub).abs() < c5 {
                continue;
            }
            let dv = sub3(cb, ca);
            let s = dot3_strided(&dv, ua);
            let lat = [dv[0] - ua[0] * s, dv[1] - ua[1] * s, dv[2] - ua[2] * s];
            let ln = dotn(&lat, false, &a.plane.n, a.plane.strided);
            let rest = [lat[0] - a.plane.n[0] * ln, lat[1] - a.plane.n[1] * ln, lat[2] - a.plane.n[2] * ln];
            if ln.abs() > 0.6 || norm3(&rest) > 1.0 {
                continue;
            }
            let zb: Vec<f64> = if qb.len() == 1 {
                vec![dot3_strided(&sub3(&qb[0], ca), ua)]
            } else {
                qb.iter().map(|q| dot3(&sub3(q, ca), ua)).collect()
            };
            let zmin = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
            let zmax = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let g1 = zmin(&zb) - zmax(&za);
            let g2 = zmin(&za) - zmax(&zb);
            let gap = if g1 >= g2 { g1 } else { g2 };
            if gap > 6.0 {
                continue;
            }
            let ra = find(&mut parent, ia);
            let rb = find(&mut parent, ib);
            if ra != rb {
                parent[rb] = ra;
            }
        }
    }
    // groups keyed by root, in narrow order
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for ia in 0..nn {
        let r = find(&mut parent, ia);
        match groups.iter_mut().find(|(k, _)| *k == r) {
            Some((_, v)) => v.push(regions[narrow[ia]].rid),
            None => groups.push((r, vec![regions[narrow[ia]].rid])),
        }
    }
    if groups.iter().all(|(_, g)| g.len() == 1) {
        return regions;
    }
    let pos_of_rid = |rid: usize| narrow.iter().position(|&i| regions[i].rid == rid);
    let byid = |rid: usize| regions.iter().find(|r| r.rid == rid).unwrap();
    let mut out: Vec<Region> = Vec::new();
    let mut done = std::collections::HashSet::new();
    for r in &regions {
        if done.contains(&r.rid) {
            continue;
        }
        let g = pos_of_rid(r.rid).map(|ia| {
            let root = find(&mut parent.clone(), ia);
            groups.iter().find(|(k, _)| *k == root).unwrap().1.clone()
        });
        match g {
            Some(g) if g.len() > 1 => {
                let mut idx: Vec<usize> = Vec::new();
                for &k in &g {
                    idx.extend_from_slice(&byid(k).idx);
                }
                let sub: Vec<V3> = if idx.len() <= 20000 {
                    idx.iter().map(|&i| p[i]).collect()
                } else {
                    let st = ceil_div_f(idx.len(), 20000);
                    idx.iter().step_by(st).map(|&i| p[i]).collect()
                };
                let mut pl = robust_plane(&sub, 3).unwrap();
                let head: Vec<usize> = idx.iter().take(2000).cloned().collect();
                if gemv_mean_pl(ns, &head, &pl) < 0.0 {
                    pl = pl.flipped();
                }
                let pts: Vec<V3> = idx.iter().map(|&i| p[i]).collect();
                let (w, l) = region_geometry(&pts, &pl);
                let mut area = 0.0f64;
                for &k in &g {
                    area += byid(k).area;
                }
                out.push(Region { rid: out.len(), idx, plane: pl, area, width: w, length: l, flange: -1 });
                for &k in &g {
                    done.insert(k);
                }
            }
            _ => {
                out.push(Region { rid: out.len(), idx: r.idx.clone(), plane: r.plane.clone(), area: r.area,
                                  width: r.width, length: r.length, flange: -1 });
                done.insert(r.rid);
            }
        }
    }
    out
}

/// numpy.linspace(0, n - 1, num).astype(int)
pub fn linspace_idx(n: usize, num: usize) -> Vec<usize> {
    let stop = (n - 1) as f64;
    let div = (num - 1) as f64;
    let step = stop / div;
    (0..num)
        .map(|k| {
            let y = if k == num - 1 { stop } else { (k as f64) * step + 0.0 };
            y as usize
        })
        .collect()
}

pub struct Flange {
    pub fid: usize,
    pub regions: Vec<usize>,
    pub idx: Vec<usize>,
    pub normal: V3,
    pub area: f64,
    pub two_sided: bool,
    pub width: f64,
    pub length: f64,
}

fn pair_flanges(p: &[V3], ns: &[V3], regions: &mut Vec<Region>, t: f64, prm: &SegParams, n: usize) -> (Vec<Flange>, Vec<i64>) {
    let m = regions.len();
    for reg in regions.iter_mut() {
        let head: Vec<usize> = reg.idx.iter().take(3000).cloned().collect();
        if gemv_mean_pl(ns, &head, &reg.plane) < 0.0 {
            reg.plane = reg.plane.flipped();
        }
    }
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(pp: &mut [usize], mut x: usize) -> usize {
        while pp[x] != x {
            pp[x] = pp[pp[x]];
            x = pp[x];
        }
        x
    }
    let mut paired = vec![false; m];
    let cos_p = prm.pair_deg.to_radians().dcos();
    let samples: Vec<Vec<usize>> = regions
        .iter()
        .map(|r| {
            if r.idx.len() <= 4000 {
                r.idx.clone()
            } else {
                linspace_idx(r.idx.len(), 4000).iter().map(|&k| r.idx[k]).collect()
            }
        })
        .collect();
    let trees: Vec<KdTree> = samples.par_iter().map(|s| KdTree::new(&s.iter().map(|&i| p[i]).collect::<Vec<_>>())).collect();
    let ub = 1.6 * t + 1.0;
    let near_tol = (0.35 * t).max(0.3);
    let pairs: Vec<(usize, usize)> = (0..m).flat_map(|i| ((i + 1)..m).map(move |j| (i, j))).collect();
    let verdict: Vec<bool> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let a = &regions[i];
            let b = &regions[j];
            if dotn(&a.plane.n, a.plane.strided, &b.plane.n, b.plane.strided) > -cos_p {
                return false;
            }
            let sj = &samples[j];
            let single = sj.len() == 1;
            let near: Vec<bool> = sj.iter().map(|&k| (a.plane.distance_one(&p[k], single) + t).abs() < near_tol).collect();
            if near.iter().filter(|&&x| x).count() < 20 {
                return false;
            }
            let ub2 = ub * ub;
            let close: Vec<bool> = sj.iter().map(|&k| trees[i].knn(&p[k], 1).first().map(|x| x.0 < ub2).unwrap_or(false)).collect();
            let ov = near.iter().zip(&close).filter(|(a, b)| **a && **b).count();
            let nclose = close.iter().filter(|&&x| x).count();
            let lim = 20.0f64.max(0.05 * (samples[i].len().min(sj.len()) as f64));
            !((ov as f64) < lim || (ov as f64) < 0.6 * nclose as f64)
        })
        .collect();
    for (&(i, j), &v) in pairs.iter().zip(&verdict) {
        if v {
            paired[i] = true;
            paired[j] = true;
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri != rj {
                parent[rj] = ri;
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in 0..m {
        let r = find(&mut parent, i);
        match groups.iter_mut().find(|(k, _)| *k == r) {
            Some((_, v)) => v.push(i),
            None => groups.push((r, vec![i])),
        }
    }
    let mut flanges = Vec::new();
    let mut fop = vec![-1i64; n];
    for (_root, members) in groups {
        let two = members.iter().any(|&i| paired[i]);
        let mut big = members[0];
        for &i in &members[1..] {
            if regions[i].area > regions[big].area {
                big = i;
            }
        }
        if !two && regions[big].width < prm.wall_width_t * t {
            continue;
        }
        let mut idx = Vec::new();
        for &i in &members {
            idx.extend_from_slice(&regions[i].idx);
        }
        let mut area = 0.0f64;
        for &i in &members {
            area += regions[i].area;
        }
        let fid = flanges.len();
        for &i in &members {
            regions[i].flange = fid as i64;
        }
        for &i in &idx {
            fop[i] = fid as i64;
        }
        flanges.push(Flange { fid, regions: members.clone(), idx, normal: regions[big].plane.n, area, two_sided: two,
                              width: regions[big].width, length: regions[big].length });
    }
    (flanges, fop)
}

pub struct SegOut {
    pub ns: Vec<V3>,
    pub label: Vec<i64>,
    pub regions: Vec<Region>,
    pub flanges: Vec<Flange>,
    pub fop: Vec<i64>,
    pub timing: Vec<(String, f64)>,
}

/// segment() from the kNN graph on.
pub fn segment(p: &[V3], n_in: &[V3], nbr: &[u32], k: usize, t: f64, pa: f64, prm: &SegParams) -> SegOut {
    let mut tm = Vec::new();
    let t1 = std::time::Instant::now();
    let n = p.len();
    let ns = smooth_normals(n_in, nbr, k, prm.smooth_iters);
    tm.push(("normals".to_string(), t1.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let cflat = prm.flat_deg.to_radians().dcos();
    let cseed = prm.seed_deg.to_radians().dcos();
    // cosn, same, flat, ok
    let cosn: Vec<f64> = (0..n * k).into_par_iter().map(|e| einsum3(&ns[nbr[e] as usize], &ns[e / k])).collect();
    let flat: Vec<bool> = (0..n)
        .map(|i| {
            let mut dev = f64::INFINITY;
            for j in 0..k {
                let c = cosn[i * k + j];
                let v = if c > 0.5 { c } else { 1.0 };
                if v < dev {
                    dev = v;
                }
            }
            dev > cflat
        })
        .collect();
    let ok: Vec<bool> = (0..n * k)
        .into_par_iter()
        .map(|e| {
            let i = e / k;
            let j = nbr[e] as usize;
            let c = cosn[e];
            let off = einsum3(&sub3(&p[j], &p[i]), &ns[i]).abs();
            (c > 0.5) && (c > cseed) && (off < prm.seed_offset_mm) && flat[i] && flat[j]
        })
        .collect();
    let all = vec![true; n];
    let mut label = components(&ok, nbr, k, &all, prm.min_seed_points, 0);
    let cos_s = prm.seed_release_deg.to_radians().dcos();
    for _ in 0..3 {
        let m = label.iter().cloned().max().map(|x| if x >= 0 { x as usize + 1 } else { 0 }).unwrap_or(0);
        let mut planes = fit_planes(p, &label, m);
        orient(&mut planes, &ns, &label);
        let g = groups_of(&label, m);
        let mut rel = vec![false; n];
        for (r, pl) in planes.iter().enumerate() {
            let pl = match pl {
                Some(x) => x,
                None => continue,
            };
            let idx = &g[r];
            for &i in idx {
                let v = if idx.len() == 1 { dotn(&ns[i], false, &pl.n, pl.strided) } else { dot3(&ns[i], &pl.n) };
                if v < cos_s {
                    rel[i] = true;
                }
            }
        }
        if rel.iter().filter(|&&x| x).count() < prm.min_seed_points {
            break;
        }
        for i in 0..n {
            if rel[i] {
                label[i] = -1;
            }
        }
        let same: Vec<bool> = (0..n * k)
            .map(|e| {
                let i = e / k;
                ok[e] && label[i] == label[nbr[e] as usize] && label[i] >= 0
            })
            .collect();
        let keptmask: Vec<bool> = label.iter().map(|&l| l >= 0).collect();
        let kept = components(&same, nbr, k, &keptmask, prm.min_seed_points, 0);
        let m2 = kept.iter().cloned().max().map(|x| if x >= 0 { x + 1 } else { 0 }).unwrap_or(0);
        let new = components(&ok, nbr, k, &rel, prm.min_seed_points, m2);
        for i in 0..n {
            label[i] = if new[i] >= 0 { new[i] } else { kept[i] };
        }
    }
    // remap to 0..len(uniq)
    let mut uniq: Vec<i64> = label.iter().cloned().filter(|&l| l >= 0).collect();
    uniq.sort_unstable();
    uniq.dedup();
    for l in label.iter_mut() {
        if *l >= 0 {
            *l = uniq.binary_search(l).unwrap() as i64;
        }
    }
    let nu = uniq.len();
    tm.push(("seeds".to_string(), t1.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let ctx = Ctx { p, nbr, k };
    let planes = fit_planes(p, &label, nu);
    let mut label = grow(&ctx, &ns, &label, &planes, prm);
    let mut planes = fit_planes(p, &label, nu);
    orient(&mut planes, &ns, &label);
    let planes = merge(&ctx, &mut label, planes, prm);
    let planes = purify(&ctx, &ns, &mut label, planes, prm);
    tm.push(("grow_merge".to_string(), t1.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let g = groups_of(&label, planes.len());
    let mut regions: Vec<Region> = Vec::new();
    for (r, pl) in planes.iter().enumerate() {
        let idx = &g[r];
        let pl = match pl {
            Some(x) if !idx.is_empty() => x,
            _ => continue,
        };
        let area = (idx.len() as f64) * pa;
        if area < prm.min_region_area_mm2 {
            for &i in idx {
                label[i] = -1;
            }
            continue;
        }
        let pts: Vec<V3> = idx.iter().map(|&i| p[i]).collect();
        let (w, l) = region_geometry(&pts, pl);
        regions.push(Region { rid: regions.len(), idx: idx.clone(), plane: pl.clone(), area, width: w, length: l,
                              flange: -1 });
    }
    let mut regions = merge_strips(p, &ns, regions, t);
    let mut label = vec![-1i64; n];
    for reg in &regions {
        for &i in &reg.idx {
            label[i] = reg.rid as i64;
        }
    }
    let (flanges, fop) = pair_flanges(p, &ns, &mut regions, t, prm, n);
    tm.push(("flanges".to_string(), t1.elapsed().as_secs_f64()));
    SegOut { ns, label, regions, flanges, fop, timing: tm }
}
