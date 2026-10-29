//! Ray casting equal to open3d.t.geometry.RaycastingScene.cast_rays (Embree 4.3.3, RTC_SCENE_FLAG_ROBUST, aarch64):
//! float32 vertices and rays, Embree's modified Pluecker triangle test with its fused multiply-adds
//! (madd / msub -> fma on aarch64), rcp = 1 / x on aarch64, closest hit with t in [tnear, tfar] inclusive.
//! Any conservative BVH visits the same candidate triangles, so the hit distance is Embree's; on an exact tie of
//! t between two triangles (a shared edge) the reported primitive may differ from Embree's traversal order.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use rayon::prelude::*;

type V3f = [f32; 3];

#[inline]
fn fms(a: f32, b: f32, c: f32) -> f32 {
    a.dfma(b, -c)
}
#[inline]
fn dot(a: &V3f, b: &V3f) -> f32 {
    a[0].dfma(b[0], a[1].dfma(b[1], a[2] * b[2]))
}
#[inline]
fn cross(a: &V3f, b: &V3f) -> V3f {
    [fms(a[1], b[2], a[2] * b[1]), fms(a[2], b[0], a[0] * b[2]), fms(a[0], b[1], a[1] * b[0])]
}
#[inline]
fn sub(a: &V3f, b: &V3f) -> V3f {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn add(a: &V3f, b: &V3f) -> V3f {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn stable_normal(a: &V3f, b: &V3f, c: &V3f) -> V3f {
    let ab_x = a[2] * b[1];
    let ab_y = a[0] * b[2];
    let ab_z = a[1] * b[0];
    let bc_x = b[2] * c[1];
    let bc_y = b[0] * c[2];
    let bc_z = b[1] * c[0];
    let cab = [fms(a[1], b[2], ab_x), fms(a[2], b[0], ab_y), fms(a[0], b[1], ab_z)];
    let cbc = [fms(b[1], c[2], bc_x), fms(b[2], c[0], bc_y), fms(b[0], c[1], bc_z)];
    [
        if ab_x.abs() < bc_x.abs() { cab[0] } else { cbc[0] },
        if ab_y.abs() < bc_y.abs() { cab[1] } else { cbc[1] },
        if ab_z.abs() < bc_z.abs() { cab[2] } else { cbc[2] },
    ]
}

/// Pluecker test; Some(t) when the ray hits within [tnear, tfar]
#[inline]
fn intersect(o: &V3f, d: &V3f, p0: &V3f, p1: &V3f, p2: &V3f, tnear: f32, tfar: f32) -> Option<f32> {
    let v0 = sub(p0, o);
    let v1 = sub(p1, o);
    let v2 = sub(p2, o);
    let e0 = sub(&v2, &v0);
    let e1 = sub(&v0, &v1);
    let e2 = sub(&v1, &v2);
    let u = dot(&cross(&e0, &add(&v2, &v0)), d);
    let v = dot(&cross(&e1, &add(&v0, &v1)), d);
    let w = dot(&cross(&e2, &add(&v1, &v2)), d);
    let uvw = u + v + w;
    let eps = f32::EPSILON * uvw.abs();
    let mn = u.min(v).min(w);
    let mx = u.max(v).max(w);
    if !((mn >= -eps) || (mx <= eps)) {
        return None;
    }
    let ng = stable_normal(&e0, &e1, &e2);
    let dd = dot(&ng, d);
    let den = dd + dd;
    let tt = dot(&v0, &ng);
    let tt = tt + tt;
    let t = (1.0f32 / den) * tt;
    if !(tnear <= t && t <= tfar) || den == 0.0 {
        return None;
    }
    Some(t)
}

fn shift(mut nodes: Vec<Node>, off: u32) -> Vec<Node> {
    for n in nodes.iter_mut() {
        if n.left != 0 {
            n.left += off;
            n.right += off;
        }
    }
    nodes
}

fn build_rec(order: &mut [u32], base: usize, v: &[V3f], f: &[[u32; 3]], cent: &[V3f]) -> Vec<Node> {
    let n = order.len();
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for &fi in order.iter() {
        for &vi in &f[fi as usize] {
            let p = v[vi as usize];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let mut me = Node { lo, hi, start: base as u32, end: (base + n) as u32, left: 0, right: 0 };
    if n <= 4 {
        return vec![me];
    }
    let mut clo = [f32::INFINITY; 3];
    let mut chi = [f32::NEG_INFINITY; 3];
    for &fi in order.iter() {
        let c = cent[fi as usize];
        for k in 0..3 {
            clo[k] = clo[k].min(c[k]);
            chi[k] = chi[k].max(c[k]);
        }
    }
    let mut ax = 0;
    for k in 1..3 {
        if chi[k] - clo[k] > chi[ax] - clo[ax] {
            ax = k;
        }
    }
    if !(chi[ax] > clo[ax]) {
        return vec![me];
    }
    let m = n / 2;
    order.select_nth_unstable_by(m, |&a, &b| cent[a as usize][ax].partial_cmp(&cent[b as usize][ax]).unwrap_or(std::cmp::Ordering::Equal));
    let (lo_s, hi_s) = order.split_at_mut(m);
    let (l, r) = if n > 50_000 {
        rayon::join(|| build_rec(lo_s, base, v, f, cent), || build_rec(hi_s, base + m, v, f, cent))
    } else {
        (build_rec(lo_s, base, v, f, cent), build_rec(hi_s, base + m, v, f, cent))
    };
    let nl = l.len() as u32;
    me.left = 1;
    me.right = 1 + nl;
    let mut out = Vec::with_capacity(1 + l.len() + r.len());
    out.push(me);
    out.extend(shift(l, 1));
    out.extend(shift(r, 1 + nl));
    out
}

struct Node {
    lo: V3f,
    hi: V3f,
    start: u32,
    end: u32,
    left: u32,
    right: u32,
}

pub struct Scene {
    v: Vec<V3f>,
    f: Vec<[u32; 3]>,
    order: Vec<u32>,
    nodes: Vec<Node>,
}

impl Scene {
    pub fn new(v64: &[[f64; 3]], f: &[[u32; 3]]) -> Scene {
        let v: Vec<V3f> = v64.iter().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
        let n = f.len();
        let mut s = Scene { v, f: f.to_vec(), order: (0..n as u32).collect(), nodes: Vec::new() };
        if n > 0 {
            let cent: Vec<V3f> = f
                .par_iter()
                .map(|t| {
                    let a = s.v[t[0] as usize];
                    let b = s.v[t[1] as usize];
                    let c = s.v[t[2] as usize];
                    [(a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0, (a[2] + b[2] + c[2]) / 3.0]
                })
                .collect();
            s.build(0, n, &cent);
        }
        s
    }

    fn build(&mut self, s: usize, e: usize, cent: &[V3f]) -> u32 {
        // parallel recursive build over disjoint slices of `order` (the hits do not depend on the tree shape)
        let (v, f) = (&self.v, &self.f);
        let nodes = build_rec(&mut self.order[s..e], s, v, f, cent);
        self.nodes = nodes;
        0
    }

    /// conservative slab test in f64 (never rejects a box the ray touches within [0, tfar])
    fn box_hit(n: &Node, o: &[f64; 3], inv: &[f64; 3], tfar: f64) -> bool {
        let mut t0 = 0.0f64;
        let mut t1 = tfar;
        for k in 0..3 {
            let pad = 1e-5 * (1.0 + n.lo[k].abs().max(n.hi[k].abs()) as f64);
            let lo = n.lo[k] as f64 - pad;
            let hi = n.hi[k] as f64 + pad;
            if inv[k].is_infinite() {
                if o[k] < lo || o[k] > hi {
                    return false;
                }
                continue;
            }
            let a = (lo - o[k]) * inv[k];
            let b = (hi - o[k]) * inv[k];
            let (a, b) = if a <= b { (a, b) } else { (b, a) };
            if a > t0 {
                t0 = a;
            }
            if b < t1 {
                t1 = b;
            }
            if t0 > t1 * (1.0 + 1e-6) + 1e-6 {
                return false;
            }
        }
        true
    }

    /// (t_hit, primitive id) of one ray; (inf, -1) on a miss
    pub fn cast(&self, o: &V3f, d: &V3f) -> (f32, i64) {
        if self.nodes.is_empty() {
            return (f32::INFINITY, -1);
        }
        let o64 = [o[0] as f64, o[1] as f64, o[2] as f64];
        let inv = [1.0 / d[0] as f64, 1.0 / d[1] as f64, 1.0 / d[2] as f64];
        let mut tfar = f32::INFINITY;
        let mut pid: i64 = -1;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if !Self::box_hit(n, &o64, &inv, if tfar.is_finite() { tfar as f64 } else { f64::INFINITY }) {
                continue;
            }
            if n.left == 0 {
                for &fi in &self.order[n.start as usize..n.end as usize] {
                    let t = self.f[fi as usize];
                    if let Some(th) =
                        intersect(o, d, &self.v[t[0] as usize], &self.v[t[1] as usize], &self.v[t[2] as usize], 0.0, tfar)
                    {
                        if th < tfar || (th == tfar && (fi as i64) < pid) {
                            tfar = th;
                            pid = fi as i64;
                        }
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        (tfar, pid)
    }

    pub fn cast_many(&self, rays: &[[f32; 6]]) -> Vec<(f32, i64)> {
        rays.par_iter().map(|r| self.cast(&[r[0], r[1], r[2]], &[r[3], r[4], r[5]])).collect()
    }
}
