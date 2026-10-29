//! Port of the parts of zs_lane_v2/z3s.py before the slicing: estimate_thickness (Embree ray casts reproduced in
//! raycast.rs, numpy RNG in nprng.rs), gauss_samples (scipy cKDTree kNN in ckdtree.rs), normal_peaks and
//! axis_families.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use crate::ckdtree::CKDTree;
use crate::kdtree::KdTree;
use crate::mesh::Mesh;
use crate::nprng::Rng;
use crate::num::*;
use crate::raycast::Scene;
use rayon::prelude::*;

pub struct Thickness {
    pub t_mm: f64,
    pub confidence: f64,
    pub two_sided: f64,
}

/// np.histogram(values, bins=edges) for explicit (sorted) edges: [e_i, e_i+1), the last bin closed.
pub fn histogram_edges(values: &[f64], edges: &[f64]) -> Vec<f64> {
    let nb = edges.len() - 1;
    let mut a = values.to_vec();
    a.sort_by(total_cmp);
    // _search_sorted_inclusive: searchsorted(edges[:-1], 'left') ++ searchsorted(edges[-1:], 'right')
    let left = |v: f64| -> usize { a.partition_point(|&x| x < v) };
    let right = |v: f64| -> usize { a.partition_point(|&x| x <= v) };
    let mut cum = Vec::with_capacity(nb + 1);
    for e in &edges[..nb] {
        cum.push(left(*e));
    }
    cum.push(right(edges[nb]));
    (0..nb).map(|i| (cum[i + 1] - cum[i]) as f64).collect()
}

/// np.convolve(a, k, mode="same") for a symmetric kernel k (|k| odd), sequential sums in increasing index order
pub fn convolve_same(a: &[f64], k: &[f64]) -> Vec<f64> {
    let (n, m) = (a.len(), k.len());
    if n < m {
        // numpy swaps the operands when the kernel is longer; same-mode length is max(n, m)
        return convolve_same_general(k, a);
    }
    convolve_same_general(a, k)
}

fn convolve_same_general(a: &[f64], k: &[f64]) -> Vec<f64> {
    // numpy _pyarray_correlate(a, k[::-1], mode=1): n_left = m / 2 partial windows on the left
    let (n, m) = (a.len() as i64, k.len() as i64);
    let kr: Vec<f64> = k.iter().rev().cloned().collect();
    let n_left = m / 2;
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let start = i - n_left;
        let mut acc = 0.0f64;
        let mut first = true;
        for j in 0..m {
            let ai = start + j;
            if ai < 0 || ai >= n {
                continue;
            }
            let p = a[ai as usize] * kr[j as usize];
            if first {
                acc = p;
                first = false;
            } else {
                acc += p;
            }
        }
        out.push(acc);
    }
    out
}

/// z3s.estimate_thickness(mesh, n_samples=20000, seed=0)
pub fn estimate_thickness(mesh: &Mesh, scene: &Scene) -> Thickness {
    let nf = mesh.f.len();
    let tot = pw_sum(&mesh.fa);
    let p: Vec<f64> = mesh.fa.iter().map(|&a| a / tot).collect();
    let mut rng = Rng::new(0);
    let idx = rng.choice_p_noreplace(&p, 20000.min(nf));
    let rays: Vec<[f32; 6]> = idx
        .iter()
        .map(|&fi| {
            let t = mesh.f[fi];
            let (a, b, c) = (mesh.v[t[0] as usize], mesh.v[t[1] as usize], mesh.v[t[2] as usize]);
            let cen = [((0.0 + a[0]) + b[0] + c[0]) / 3.0, ((0.0 + a[1]) + b[1] + c[1]) / 3.0, ((0.0 + a[2]) + b[2] + c[2]) / 3.0];
            let d = neg3(&mesh.fnrm[fi]);
            [
                (cen[0] + d[0] * 1e-3) as f32,
                (cen[1] + d[1] * 1e-3) as f32,
                (cen[2] + d[2] * 1e-3) as f32,
                d[0] as f32,
                d[1] as f32,
                d[2] as f32,
            ]
        })
        .collect();
    let hits = scene.cast_many(&rays);
    let mut good_t = Vec::new();
    let n = idx.len();
    for k in 0..n {
        let (th, pid) = hits[k];
        let th = th as f64;
        let ok = th.is_finite() && pid >= 0 && (pid as usize) < nf;
        if !ok {
            continue;
        }
        let a = mesh.fnrm[pid as usize];
        let b = mesh.fnrm[idx[k]];
        let anti = a[2].dfma(b[2], a[0] * b[0]) + a[1] * b[1] < -0.9;
        if anti && th > 0.2 && th < 12.0 {
            good_t.push(th);
        }
    }
    let gm = good_t.len() as f64 / n as f64;
    if good_t.len() < 50 {
        return Thickness { t_mm: 1.5, confidence: 0.0, two_sided: gm };
    }
    let edges = arange(0.2, 12.0, 0.02);
    let h = histogram_edges(&good_t, &edges);
    let k5 = [0.2f64; 5];
    let hs = convolve_same(&h, &k5);
    let kk = argmax(&hs);
    let mode = 0.5 * (edges[kk] + edges[kk + 1]);
    let near: Vec<f64> = good_t.iter().cloned().filter(|x| (x - mode).abs() < 0.15 * mode + 0.05).collect();
    let t = if !near.is_empty() { median(&near) } else { mode };
    Thickness { t_mm: t, confidence: near.len() as f64 / good_t.len() as f64, two_sided: gm }
}

pub struct Gauss {
    pub n: Vec<V3>,
    pub w: Vec<f64>,
}

/// gauss_samples(mesh, n_max=150000, k=16)
pub fn gauss_samples(mesh: &Mesh) -> Gauss {
    let nv = mesh.v.len();
    let mut va = vec![0.0f64; nv];
    for j in 0..3 {
        for (t, a) in mesh.f.iter().zip(&mesh.fa) {
            va[t[j] as usize] += a / 3.0;
        }
    }
    let idx: Vec<usize> = if nv <= 150000 {
        (0..nv).collect()
    } else {
        Rng::new(1).choice_noreplace(nv, 150000).iter().map(|&x| x as usize).collect()
    };
    let tree = CKDTree::new(&mesh.v);
    let k = 16.min(nv);
    let ns: Vec<V3> = idx
        .par_iter()
        .map(|&i| {
            let nb = tree.query(&mesh.v[i], k, f64::INFINITY);
            let mut s = [0.0f64; 3];
            for (j, &(_, q)) in nb.iter().enumerate() {
                let v = mesh.vn[q];
                if j == 0 {
                    s = v;
                } else {
                    s = [s[0] + v[0], s[1] + v[1], s[2] + v[2]];
                }
            }
            let l = norm3(&s).max(1e-12);
            [s[0] / l, s[1] / l, s[2] / l]
        })
        .collect();
    let vsum = pw_sum(&va);
    let sub: Vec<f64> = idx.iter().map(|&i| va[i]).collect();
    let ssum = pw_sum(&sub).max(1e-12);
    let f = vsum / ssum;
    let w: Vec<f64> = sub.iter().map(|&x| x * f).collect();
    Gauss { n: ns, w }
}

fn fib_hemisphere(n: usize) -> Vec<V3> {
    let c = 3.0 - 5.0f64.sqrt();
    (0..n)
        .map(|k| {
            let i = k as f64 + 0.5;
            let z = i / n as f64;
            let phi = i * std::f64::consts::PI * c;
            let r = (1.0 - z * z).sqrt();
            [r * lcos(phi), r * lsin(phi), z]
        })
        .collect()
}

#[derive(Clone)]
pub struct Peak {
    pub normal: V3,
    pub area: f64,
}

pub struct PeakParams {
    pub peak_sep_deg: f64,
    pub peak_win_deg: f64,
    pub peak_min_area_frac: f64,
    pub peak_min_area_mm2: f64,
    pub peak_min_area_rel: f64,
    pub max_families: usize,
    pub family_merge_deg: f64,
}

pub fn normal_peaks(g: &Gauss, prm: &PeakParams) -> Vec<Peak> {
    let (n, a) = (&g.n, &g.w);
    let grid = fib_hemisphere(6000);
    let ng = grid.len();
    let cw = prm.peak_win_deg.to_radians().dcos();
    let mut both = grid.clone();
    both.extend(grid.iter().map(neg3));
    let tree = KdTree::new(&both);
    let chord = (2.0 - 2.0 * cw).sqrt();
    let lists: Vec<Vec<usize>> = n.par_iter().map(|q| tree.ball(q, chord)).collect();
    let mut dens = vec![0.0f64; ng];
    for (j, lst) in lists.iter().enumerate() {
        for &gi in lst {
            dens[gi % ng] += a[j];
        }
    }
    let total = pw_sum(a);
    // np.argsort(-dens): ties broken by index (numpy's introsort is unstable; exact ties above the floor are
    // not expected)
    let mut order: Vec<usize> = (0..ng).collect();
    order.sort_by(|&x, &y| total_cmp(&(-dens[x]), &(-dens[y])));
    let cs = prm.peak_sep_deg.to_radians().dcos();
    let mut floor = prm.peak_min_area_mm2;
    if prm.peak_min_area_rel > 0.0 {
        floor = floor.min(prm.peak_min_area_rel * total);
    }
    let thr = (prm.peak_min_area_frac * total).max(floor);
    let c2 = (2.0 * prm.peak_win_deg).to_radians().dcos();
    let mut peaks: Vec<V3> = Vec::new();
    let mut out = Vec::new();
    for &gi in &order {
        if dens[gi] < thr {
            break;
        }
        let d = grid[gi];
        if peaks.iter().any(|q| dot3(&d, q).abs() > cs) {
            continue;
        }
        let dots: Vec<f64> = n.iter().map(|x| dot3(x, &d)).collect();
        let m: Vec<usize> = (0..n.len()).filter(|&i| dots[i].abs() > cw).collect();
        if m.len() < 3 {
            continue;
        }
        // (n[m] * (sign(dots[m]) * a[m])[:, None]).sum(0), sequential
        let mut v = [0.0f64; 3];
        for &i in &m {
            let sg = if dots[i] > 0.0 { 1.0 } else if dots[i] < 0.0 { -1.0 } else { 0.0 };
            let wv = sg * a[i];
            v[0] += n[i][0] * wv;
            v[1] += n[i][1] * wv;
            v[2] += n[i][2] * wv;
        }
        let l = norm3(&v);
        let v = [v[0] / l, v[1] / l, v[2] / l];
        if peaks.iter().any(|q| dot3(&v, q).abs() > cs) {
            continue;
        }
        let ad: Vec<f64> = n.iter().map(|x| dot3(x, &v).abs()).collect();
        let core_v: Vec<f64> = (0..n.len()).filter(|&i| ad[i] > cw).map(|i| a[i]).collect();
        let ring_v: Vec<f64> = (0..n.len()).filter(|&i| ad[i] <= cw && ad[i] > c2).map(|i| a[i]).collect();
        let core = pw_sum(&core_v);
        let ring = pw_sum(&ring_v);
        if core < 1.2 * ring {
            continue;
        }
        peaks.push(v);
        out.push(Peak { normal: v, area: dens[gi] });
        if out.len() >= 40 {
            break;
        }
    }
    out
}

pub struct Family {
    pub axis: V3,
    pub weight: f64,
    pub arc_area: f64,
    pub score: f64,
}

pub fn axis_families(g: &Gauss, peaks: &[Peak], prm: &PeakParams) -> Vec<Family> {
    let smin = 8.0f64.to_radians().dsin();
    let mut cands: Vec<(f64, V3)> = Vec::new();
    for i in 0..peaks.len() {
        for j in (i + 1)..peaks.len() {
            let c = cross3(&peaks[i].normal, &peaks[j].normal);
            let s = norm3(&c);
            if s < smin {
                continue;
            }
            let c = [c[0] / s, c[1] / s, c[2] / s];
            let w = if peaks[i].area <= peaks[j].area { peaks[i].area } else { peaks[j].area };
            cands.push((w, c));
        }
    }
    cands.sort_by(|x, y| total_cmp(&(-x.0), &(-y.0)));
    let cm = prm.family_merge_deg.to_radians().dcos();
    let mut fams: Vec<(V3, f64)> = Vec::new();
    for (w, c) in cands {
        let mut hit = false;
        for f in fams.iter_mut() {
            if dot3(&c, &f.0).abs() > cm {
                f.1 += w;
                hit = true;
                break;
            }
        }
        if !hit {
            fams.push((c, w));
        }
    }
    let (n, a) = (&g.n, &g.w);
    let c3 = 3.0f64.to_radians().dcos();
    // on_peak = (|n @ pk.T| > cos 3deg).any(1): n @ pk.T is a gemm (Accelerate)
    let on_peak: Vec<bool> = if peaks.is_empty() {
        vec![false; n.len()]
    } else {
        let pk: Vec<f64> = peaks.iter().flat_map(|p| p.normal.iter().cloned()).collect();
        let prod = accel::gemm_nt(&n.iter().flat_map(|x| x.iter().cloned()).collect::<Vec<_>>(), n.len(), &pk, peaks.len(), 3);
        (0..n.len()).map(|i| (0..peaks.len()).any(|j| prod[i * peaks.len() + j].abs() > c3)).collect()
    };
    let s2 = 2.0f64.to_radians().dsin();
    let total = pw_sum(a);
    let mut out: Vec<Family> = fams
        .iter()
        .map(|(ax, w)| {
            let gc: Vec<bool> = n.iter().map(|x| dot3(x, ax).abs() < s2).collect();
            let arc: Vec<f64> = (0..n.len()).filter(|&i| gc[i] && !on_peak[i]).map(|i| a[i]).collect();
            let arc_area = pw_sum(&arc);
            Family { axis: *ax, weight: *w, arc_area, score: 0.0 }
        })
        .collect();
    for f in out.iter_mut() {
        f.score = f.weight / total + 2.0 * f.arc_area / total;
    }
    out.sort_by(|x, y| total_cmp(&(-x.score), &(-y.score)));
    out.truncate(prm.max_families);
    out
}
