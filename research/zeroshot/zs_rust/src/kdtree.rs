//! 3-D kd-tree: ball queries (result sorted by point index) and k-nearest queries (sorted by distance, then
//! index). Distances are compared as squared Euclidean distances, inclusive at the radius (scipy cKDTree
//! query_ball_point semantics).
#![allow(dead_code)]
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::num::V3;

pub struct KdTree {
    pts: Vec<V3>,
    idx: Vec<u32>,
    nodes: Vec<Node>,
}

#[derive(Clone, Copy)]
struct Node {
    lo: [f64; 3],
    hi: [f64; 3],
    start: u32,
    end: u32,
    left: u32, // 0 = leaf
    right: u32,
}

const LEAF: usize = 16;

fn build_rec(idx: &mut [u32], base: usize, pts: &[V3]) -> Vec<Node> {
    let n = idx.len();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &i in idx.iter() {
        let p = &pts[i as usize];
        for d in 0..3 {
            if p[d] < lo[d] {
                lo[d] = p[d];
            }
            if p[d] > hi[d] {
                hi[d] = p[d];
            }
        }
    }
    let mut me = Node { lo, hi, start: base as u32, end: (base + n) as u32, left: 0, right: 0 };
    if n <= LEAF {
        return vec![me];
    }
    let mut dim = 0;
    for d in 1..3 {
        if hi[d] - lo[d] > hi[dim] - lo[dim] {
            dim = d;
        }
    }
    if !(hi[dim] > lo[dim]) {
        return vec![me];
    }
    let m = n / 2;
    idx.select_nth_unstable_by(m, |&a, &b| pts[a as usize][dim].partial_cmp(&pts[b as usize][dim]).unwrap_or(std::cmp::Ordering::Equal));
    let (ls, rs) = idx.split_at_mut(m);
    let (l, r) = if n > 50_000 {
        rayon::join(|| build_rec(ls, base, pts), || build_rec(rs, base + m, pts))
    } else {
        (build_rec(ls, base, pts), build_rec(rs, base + m, pts))
    };
    let nl = l.len() as u32;
    me.left = 1;
    me.right = 1 + nl;
    let mut out = Vec::with_capacity(1 + l.len() + r.len());
    out.push(me);
    for mut x in l {
        if x.left != 0 {
            x.left += 1;
            x.right += 1;
        }
        out.push(x);
    }
    for mut x in r {
        if x.left != 0 {
            x.left += 1 + nl;
            x.right += 1 + nl;
        }
        out.push(x);
    }
    out
}

impl KdTree {
    pub fn new(pts: &[V3]) -> KdTree {
        let n = pts.len();
        let mut t = KdTree { pts: pts.to_vec(), idx: (0..n as u32).collect(), nodes: Vec::with_capacity(2 * n / LEAF + 2) };
        if n > 0 {
            t.build(0, n);
        }
        t
    }

    fn build(&mut self, s: usize, e: usize) -> u32 {
        let pts = &self.pts;
        self.nodes = build_rec(&mut self.idx[s..e], s, pts);
        0
    }

    #[inline]
    fn d2(a: &V3, b: &V3) -> f64 {
        // scipy sqeuclidean_distance_double for m = 3, compiled with fp-contract: fma chain
        let x = a[0] - b[0];
        let y = a[1] - b[1];
        let z = a[2] - b[2];
        z.dfma(z, y.dfma(y, x * x))
    }

    #[inline]
    fn box_d2(n: &Node, q: &V3) -> f64 {
        let mut s = 0.0;
        for d in 0..3 {
            let v = if q[d] < n.lo[d] {
                n.lo[d] - q[d]
            } else if q[d] > n.hi[d] {
                q[d] - n.hi[d]
            } else {
                0.0
            };
            s += v * v;
        }
        s
    }

    /// Indices of points within r of q (inclusive), sorted ascending.
    pub fn ball(&self, q: &V3, r: f64) -> Vec<usize> {
        let mut out = Vec::new();
        self.ball_into(q, r, &mut out);
        out.sort_unstable();
        out
    }

    pub fn ball_into(&self, q: &V3, r: f64, out: &mut Vec<usize>) {
        if self.nodes.is_empty() {
            return;
        }
        let r2 = r * r;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if Self::box_d2(n, q) > r2 * (1.0 + 1e-12) + 1e-300 {
                continue;
            }
            if n.left == 0 {
                for &i in &self.idx[n.start as usize..n.end as usize] {
                    if Self::d2(&self.pts[i as usize], q) <= r2 {
                        out.push(i as usize);
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
    }

    /// Any point within r of q?
    pub fn any_within(&self, q: &V3, r: f64) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let r2 = r * r;
        let mut stack = vec![0u32];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni as usize];
            if Self::box_d2(n, q) > r2 * (1.0 + 1e-12) + 1e-300 {
                continue;
            }
            if n.left == 0 {
                for &i in &self.idx[n.start as usize..n.end as usize] {
                    if Self::d2(&self.pts[i as usize], q) <= r2 {
                        return true;
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.right);
            }
        }
        false
    }

    /// k nearest neighbours: (squared distance, index) sorted by (distance, index).
    pub fn knn(&self, q: &V3, k: usize) -> Vec<(f64, usize)> {
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        if self.nodes.is_empty() || k == 0 {
            return best;
        }
        let mut worst = f64::INFINITY;
        let mut stack: Vec<(f64, u32)> = vec![(0.0, 0)];
        while let Some((bd, ni)) = stack.pop() {
            if best.len() == k && bd > worst {
                continue;
            }
            let n = &self.nodes[ni as usize];
            if n.left == 0 {
                for &i in &self.idx[n.start as usize..n.end as usize] {
                    let d = Self::d2(&self.pts[i as usize], q);
                    let item = (d, i as usize);
                    if best.len() < k || lt(&item, best.last().unwrap()) {
                        let pos = best.partition_point(|x| lt(x, &item));
                        best.insert(pos, item);
                        if best.len() > k {
                            best.pop();
                        }
                        if best.len() == k {
                            worst = best[k - 1].0;
                        }
                    }
                }
            } else {
                let l = &self.nodes[n.left as usize];
                let r = &self.nodes[n.right as usize];
                let dl = Self::box_d2(l, q);
                let dr = Self::box_d2(r, q);
                // push farther first so nearer pops first
                if dl <= dr {
                    stack.push((dr, n.right));
                    stack.push((dl, n.left));
                } else {
                    stack.push((dl, n.left));
                    stack.push((dr, n.right));
                }
            }
        }
        best
    }

    /// nearest neighbour distance (Euclidean) if within `ub` (strictly less), else None.
    pub fn nearest_within(&self, q: &V3, ub: f64) -> Option<(f64, usize)> {
        let r = self.knn(q, 1);
        if let Some(&(d2, i)) = r.first() {
            let d = d2.sqrt();
            if d < ub {
                return Some((d, i));
            }
        }
        None
    }
}

#[inline]
fn lt(a: &(f64, usize), b: &(f64, usize)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}
