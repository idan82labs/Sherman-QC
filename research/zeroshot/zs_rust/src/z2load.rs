//! Port of z2_self_recipe.segment.load_scan for meshes: d1_two_skin.mesh_from_arrays, d1_two_skin.estimate_thickness
//! (surface sampling with numpy's RNG, inward probes cast as Embree does, robust histogram mode), the finite /
//! normal filter and the random thinning to max_points.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::nprng::Rng;
use crate::num::*;
use crate::raycast::Scene;
use crate::z3prep::{convolve_same, histogram_edges};

pub struct D1Mesh<'a> {
    pub v: &'a [V3],
    pub f: &'a [[u32; 3]],
    pub fnrm: Vec<V3>,
    pub fa: Vec<f64>,
    pub vn: Vec<V3>,
}

/// two_skin.mesh_from_arrays (every face kept; fn = cr / max(|cr|, 1e-300))
pub fn d1_mesh<'a>(v: &'a [V3], f: &'a [[u32; 3]]) -> D1Mesh<'a> {
    let mut cr_all = Vec::with_capacity(f.len());
    let mut fnrm = Vec::with_capacity(f.len());
    let mut fa = Vec::with_capacity(f.len());
    for t in f {
        let p0 = v[t[0] as usize];
        let cr = cross3(&sub3(&v[t[1] as usize], &p0), &sub3(&v[t[2] as usize], &p0));
        let nrm = norm3(&cr);
        fa.push(0.5 * nrm);
        let d = nrm.max(1e-300);
        fnrm.push([cr[0] / d, cr[1] / d, cr[2] / d]);
        cr_all.push(cr);
    }
    let mut vn = vec![[0.0f64; 3]; v.len()];
    for k in 0..3 {
        for (t, c) in f.iter().zip(&cr_all) {
            let x = &mut vn[t[k] as usize];
            x[0] += c[0];
            x[1] += c[1];
            x[2] += c[2];
        }
    }
    for x in vn.iter_mut() {
        let n = norm3(x).max(1e-300);
        *x = [x[0] / n, x[1] / n, x[2] / n];
    }
    D1Mesh { v, f, fnrm, fa, vn }
}

/// two_skin.robust_mode(values, lo, hi, bin_mm, smooth_mm) -> (mode, edges)
fn robust_mode(values: &[f64], lo: f64, hi: f64, bin_mm: f64, smooth_mm: f64) -> f64 {
    let edges = arange(lo, hi + bin_mm, bin_mm);
    let counts = histogram_edges(values, &edges);
    let k = 1i64.max(round_even(3.0 * smooth_mm / bin_mm) as i64);
    let x: Vec<f64> = (-k..=k).map(|i| (i as f64) * bin_mm).collect();
    let w: Vec<f64> = x.iter().map(|&xi| {
        let q = xi / smooth_mm;
        (-0.5 * (q * q)).dexp()
    }).collect();
    let ws = pw_sum(&w);
    let kern: Vec<f64> = w.iter().map(|&a| a / ws).collect();
    let sm = convolve_same(&counts, &kern);
    let i = argmax(&sm);
    0.5 * (edges[i] + edges[i + 1])
}

/// two_skin.estimate_thickness(mesh, n_samples, seed=0): (t_mm or None, confidence)
pub fn estimate_thickness(m: &D1Mesh, scene: &Scene, n: usize, seed: u64) -> (Option<f64>, f64) {
    let (dmin, dmax) = (0.25f64, 12.0f64);
    let mut rng = Rng::new(seed);
    let tot = pw_sum(&m.fa);
    let p: Vec<f64> = m.fa.iter().map(|&a| a / tot).collect();
    let fi = rng.choice_p(&p, n);
    let r1 = rng.random_n(n);
    let r2 = rng.random_n(n);
    let mut rays = Vec::with_capacity(n);
    let mut nrm = Vec::with_capacity(n);
    for k in 0..n {
        let s = r1[k].sqrt();
        let (a, b, c) = (1.0 - s, s * (1.0 - r2[k]), s * r2[k]);
        let t = m.f[fi[k]];
        let (v0, v1, v2) = (m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]);
        let pt = [
            (a * v0[0] + b * v1[0]) + c * v2[0],
            (a * v0[1] + b * v1[1]) + c * v2[1],
            (a * v0[2] + b * v1[2]) + c * v2[2],
        ];
        let nn = m.fnrm[fi[k]];
        let d = neg3(&nn);
        let eps = 0.02;
        rays.push([
            (pt[0] + eps * d[0]) as f32,
            (pt[1] + eps * d[1]) as f32,
            (pt[2] + eps * d[2]) as f32,
            d[0] as f32,
            d[1] as f32,
            d[2] as f32,
        ]);
        nrm.push(nn);
    }
    let hits = scene.cast_many(&rays);
    let nf = m.f.len() as i64;
    let mut dist = vec![f64::INFINITY; n];
    let mut ok = vec![false; n];
    for k in 0..n {
        let (th, pid) = hits[k];
        let th = th as f64;
        let miss = !th.is_finite() || pid < 0 || pid >= nf;
        if miss {
            continue;
        }
        let hn = m.fnrm[pid as usize];
        let nv = nrm[k];
        let cos_hit = (hn[0] * nv[0] + hn[1] * nv[1]) + hn[2] * nv[2];
        let d = neg3(&nv);
        let exits = (hn[0] * d[0] + hn[1] * d[1]) + hn[2] * d[2] > 0.0;
        dist[k] = th + 0.02;
        ok[k] = exits && cos_hit < -0.85 && dist[k] > dmin && dist[k] < dmax;
    }
    let d: Vec<f64> = (0..n).filter(|&k| ok[k]).map(|k| dist[k]).collect();
    if d.len() < 200 {
        return (None, 0.0);
    }
    let mode = robust_mode(&d, dmin, dmax, 0.02, 0.06);
    let near: Vec<f64> = d.iter().cloned().filter(|x| (x - mode).abs() < 0.2 * mode).collect();
    let t = median(&near);
    let near_m: Vec<bool> = d.iter().map(|x| (x - t).abs() < 0.2 * t).collect();
    let nv: Vec<f64> = d.iter().zip(&near_m).filter(|(_, &b)| b).map(|(x, _)| *x).collect();
    let t = median(&nv);
    let ad: Vec<f64> = nv.iter().map(|x| (x - t).abs()).collect();
    let mad = median(&ad);
    let two_sided = (0..n).filter(|&k| ok[k] && (dist[k] - t).abs() < 0.25 * t).count() as f64 / n as f64;
    let peak = mean_bool(&near_m);
    let conf = peak * 1.0f64.min(two_sided / 0.25) * (-(mad / t.max(1e-6)) / 0.15).dexp();
    (Some(t), conf)
}

pub struct Loaded {
    pub p: Vec<V3>,
    pub n: Vec<V3>,
    pub t: f64,
    pub t_conf: f64,
    pub area: f64,
    pub pa: f64,
}

/// segment.load_scan(path, prm) for a mesh: (P, N, t, t_conf, area, area per point)
pub fn load_scan(v: &[V3], f: &[[u32; 3]], max_points: usize, shared: Option<&Scene>) -> Loaded {
    let m = crate::prof::time("z2.d1_mesh", || d1_mesh(v, f));
    let own;
    let scene: &Scene = match shared {
        Some(s) => s,
        None => {
            own = crate::prof::time("z2.bvh_build", || Scene::new(m.v, m.f));
            &own
        }
    };
    let ns = 120_000usize.min(20_000usize.max(m.f.len() / 2));
    let (tm, conf) = crate::prof::time("z2.thickness_rays", || estimate_thickness(&m, scene, ns, 0));
    let t = match tm {
        Some(x) if x != 0.0 => x,
        _ => 1.5,
    };
    let area = pw_sum(&m.fa);
    let mut p = Vec::with_capacity(m.v.len());
    let mut n = Vec::with_capacity(m.v.len());
    for (x, nn) in m.v.iter().zip(&m.vn) {
        if x.iter().all(|c| c.is_finite()) && norm3(nn) > 0.5 {
            p.push(*x);
            n.push(*nn);
        }
    }
    if p.len() > max_points {
        let mut keep: Vec<i64> = Rng::new(0).choice_noreplace(p.len(), max_points);
        keep.sort_unstable();
        p = keep.iter().map(|&i| p[i as usize]).collect();
        n = keep.iter().map(|&i| n[i as usize]).collect();
    }
    let pa = area / p.len() as f64;
    Loaded { p, n, t, t_conf: conf, area, pa }
}
