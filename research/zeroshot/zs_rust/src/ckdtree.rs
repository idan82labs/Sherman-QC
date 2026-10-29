//! Exact port of scipy.spatial.cKDTree (scipy 1.17 ckdtree/src/build.cxx + query.cxx) for 3-D points, p = 2,
//! eps = 0, non-periodic, leafsize 16, balanced_tree = True, compact_nodes = True. The tree build reproduces
//! libc++'s std::nth_element / std::partition, and the query its binary heaps, so the k nearest neighbours come
//! out in exactly scipy's order, ties included.
#![allow(dead_code)]
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::num::V3;

#[derive(Clone, Copy)]
struct Node {
    split_dim: i32,
    split: f64,
    start: usize,
    end: usize,
    less: usize,
    greater: usize,
}

pub struct CKDTree {
    data: Vec<V3>,
    idx: Vec<usize>,
    nodes: Vec<Node>,
    mins: V3,
    maxes: V3,
    leafsize: usize,
}

// ------------------------------------------------------------------ libc++ algorithms on index slices

fn sort3<F: Fn(usize, usize) -> bool>(v: &mut [usize], x: usize, y: usize, z: usize, c: &F) -> u32 {
    let mut r = 0;
    if !c(v[y], v[x]) {
        if !c(v[z], v[y]) {
            return r;
        }
        v.swap(y, z);
        r = 1;
        if c(v[y], v[x]) {
            v.swap(x, y);
            r = 2;
        }
        return r;
    }
    if c(v[z], v[y]) {
        v.swap(x, z);
        return 1;
    }
    v.swap(x, y);
    r = 1;
    if c(v[z], v[y]) {
        v.swap(y, z);
        r = 2;
    }
    r
}

fn min_element<F: Fn(usize, usize) -> bool>(v: &[usize], first: usize, last: usize, c: &F) -> usize {
    let mut m = first;
    for i in (first + 1)..last {
        if c(v[i], v[m]) {
            m = i;
        }
    }
    m
}

fn max_element<F: Fn(usize, usize) -> bool>(v: &[usize], first: usize, last: usize, c: &F) -> usize {
    let mut m = first;
    for i in (first + 1)..last {
        if c(v[m], v[i]) {
            m = i;
        }
    }
    m
}

fn selection_sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], mut first: usize, last: usize, c: &F) {
    let lm1 = last - 1;
    while first != lm1 {
        let i = min_element(v, first, last, c);
        if i != first {
            v.swap(first, i);
        }
        first += 1;
    }
}

/// libc++ std::nth_element (LLVM 17 __nth_element) on v[first..last] with nth
fn nth_element<F: Fn(usize, usize) -> bool>(v: &mut [usize], mut first: usize, nth: usize, mut last: usize, c: &F) {
    let limit = 7usize;
    loop {
        if nth == last {
            return;
        }
        let len = last - first;
        match len {
            0 | 1 => return,
            2 => {
                last -= 1;
                if c(v[last], v[first]) {
                    v.swap(first, last);
                }
                return;
            }
            3 => {
                sort3(v, first, first + 1, last - 1, c);
                return;
            }
            _ => {}
        }
        if len <= limit {
            selection_sort(v, first, last, c);
            return;
        }
        let mut m = first + len / 2;
        let mut lm1 = last;
        lm1 -= 1;
        let mut n_swaps = sort3(v, first, m, lm1, c);
        let mut i = first;
        let mut j = lm1;
        let mut handled_equal_branch = false;
        if !c(v[i], v[m]) {
            // __nth_element_find_guard
            let mut found = false;
            loop {
                j -= 1;
                if i == j {
                    break;
                }
                if c(v[j], v[m]) {
                    found = true;
                    break;
                }
            }
            if found {
                v.swap(i, j);
                n_swaps += 1;
            } else {
                i += 1;
                j = last;
                j -= 1;
                if !c(v[first], v[j]) {
                    loop {
                        if i == j {
                            return;
                        } else if c(v[first], v[i]) {
                            v.swap(i, j);
                            n_swaps += 1;
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                if i == j {
                    return;
                }
                loop {
                    while !c(v[first], v[i]) {
                        i += 1;
                    }
                    loop {
                        j -= 1;
                        if !c(v[first], v[j]) {
                            break;
                        }
                    }
                    if i >= j {
                        break;
                    }
                    v.swap(i, j);
                    n_swaps += 1;
                    i += 1;
                }
                if nth < i {
                    return;
                }
                first = i;
                handled_equal_branch = true;
            }
        }
        if handled_equal_branch {
            continue;
        }
        i += 1;
        if i < j {
            loop {
                while c(v[i], v[m]) {
                    i += 1;
                }
                loop {
                    j -= 1;
                    if c(v[j], v[m]) {
                        break;
                    }
                }
                if i >= j {
                    break;
                }
                v.swap(i, j);
                n_swaps += 1;
                if m == i {
                    m = j;
                }
                i += 1;
            }
        }
        if i != m && c(v[m], v[i]) {
            v.swap(i, m);
            n_swaps += 1;
        }
        if nth == i {
            return;
        }
        if n_swaps == 0 {
            if nth < i {
                let mut jj = first;
                let mut mm = first;
                let mut sorted = false;
                loop {
                    jj += 1;
                    if jj == i {
                        sorted = true;
                        break;
                    }
                    if c(v[jj], v[mm]) {
                        break;
                    }
                    mm = jj;
                }
                if sorted {
                    return;
                }
            } else {
                let mut jj = i;
                let mut mm = i;
                let mut sorted = false;
                loop {
                    jj += 1;
                    if jj == last {
                        sorted = true;
                        break;
                    }
                    if c(v[jj], v[mm]) {
                        break;
                    }
                    mm = jj;
                }
                if sorted {
                    return;
                }
            }
        }
        if nth < i {
            last = i;
        } else {
            i += 1;
            first = i;
        }
    }
}

/// libc++ std::partition (bidirectional version) on v[first..last]; returns the partition point
fn partition<P: Fn(usize) -> bool>(v: &mut [usize], mut first: usize, last0: usize, pred: P) -> usize {
    let mut last = last0;
    loop {
        loop {
            if first == last {
                return first;
            }
            if !pred(v[first]) {
                break;
            }
            first += 1;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if pred(v[last]) {
                break;
            }
        }
        v.swap(first, last);
        first += 1;
    }
}

fn next_up(x: f64) -> f64 {
    // std::nextafter(x, +inf)
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let b = x.to_bits();
    if x > 0.0 {
        f64::from_bits(b + 1)
    } else {
        f64::from_bits(b - 1)
    }
}

/// one node and its subtrees (node 0 of the returned vector); the partitions are scipy's, so building the two
/// halves in parallel gives the same tree
fn build_rec(idx: &mut [usize], base: usize, data: &[V3], leafsize: usize) -> Vec<Node> {
    let n = idx.len();
    let leaf = Node { split_dim: -1, split: 0.0, start: base, end: base + n, less: 0, greater: 0 };
    if n <= leafsize {
        return vec![leaf];
    }
    let mut maxes = data[idx[0]];
    let mut mins = data[idx[0]];
    for j in 1..n {
        let p = data[idx[j]];
        for i in 0..3 {
            let tv = p[i];
            maxes[i] = if maxes[i] > tv { maxes[i] } else { tv };
            mins[i] = if mins[i] < tv { mins[i] } else { tv };
        }
    }
    let mut d = 0usize;
    let mut size = 0.0f64;
    for i in 0..3 {
        if maxes[i] - mins[i] > size {
            d = i;
            size = maxes[i] - mins[i];
        }
    }
    if maxes[d] == mins[d] {
        return vec![leaf];
    }
    let cmp = |a: usize, b: usize| data[a][d] < data[b][d];
    let mid = n / 2;
    nth_element(idx, 0, mid, n, &cmp);
    let mut split = data[idx[mid]][d];
    let mut p = partition(idx, 0, mid, |a| !(data[a][d] >= split));
    if p == 0 {
        let mi = min_element(idx, 0, n, &cmp);
        split = next_up(data[idx[mi]][d]);
        p = partition(idx, 0, n, |a| !(data[a][d] >= split));
    } else if p == n {
        let ma = max_element(idx, 0, n, &cmp);
        split = data[idx[ma]][d];
        p = partition(idx, 0, n, |a| !(data[a][d] >= split));
    }
    if p == 0 || p == n {
        return vec![leaf];
    }
    let (ls, rs) = idx.split_at_mut(p);
    let (l, r) = if n > 50_000 {
        rayon::join(|| build_rec(ls, base, data, leafsize), || build_rec(rs, base + p, data, leafsize))
    } else {
        (build_rec(ls, base, data, leafsize), build_rec(rs, base + p, data, leafsize))
    };
    let nl = l.len();
    let mut out = Vec::with_capacity(1 + nl + r.len());
    out.push(Node { split_dim: d as i32, split, start: base, end: base + n, less: 1, greater: 1 + nl });
    for mut x in l {
        if x.split_dim != -1 {
            x.less += 1;
            x.greater += 1;
        }
        out.push(x);
    }
    for mut x in r {
        if x.split_dim != -1 {
            x.less += 1 + nl;
            x.greater += 1 + nl;
        }
        out.push(x);
    }
    out
}

impl CKDTree {
    pub fn new(data: &[V3]) -> CKDTree {
        let n = data.len();
        let mut mins = [f64::INFINITY; 3];
        let mut maxes = [f64::NEG_INFINITY; 3];
        for p in data {
            for d in 0..3 {
                // numpy amin / amax
                if p[d] < mins[d] {
                    mins[d] = p[d];
                }
                if p[d] > maxes[d] {
                    maxes[d] = p[d];
                }
            }
        }
        let mut t = CKDTree { data: data.to_vec(), idx: (0..n).collect(), nodes: Vec::new(), mins, maxes, leafsize: 16 };
        if n > 0 {
            let (mut mx, mut mn) = (maxes, mins);
            t.build(0, n, &mut mx, &mut mn);
        }
        t
    }

    fn build(&mut self, start: usize, end: usize, _maxes: &mut V3, _mins: &mut V3) -> usize {
        let data = &self.data;
        self.nodes = build_rec(&mut self.idx[start..end], start, data, self.leafsize);
        0
    }

    #[inline]
    fn sqdist(a: &V3, b: &V3) -> f64 {
        // clang contracts `s += d * d` into fused multiply-adds (scipy is built with fp-contract on)
        let mut s = 0.0f64;
        for i in 0..3 {
            let d = a[i] - b[i];
            s = d.dfma(d, s);
        }
        s
    }

    /// query(x, k, distance_upper_bound): squared distances and indices in scipy's order (missing: (inf, n)).
    pub fn query(&self, x: &V3, k: usize, distance_upper_bound: f64) -> Vec<(f64, usize)> {
        let n = self.data.len();
        let mut out = Vec::with_capacity(k);
        if n == 0 {
            for _ in 0..k {
                out.push((f64::INFINITY, n));
            }
            return out;
        }
        // nodeinfo: node, min_distance, side distances
        struct NI {
            node: usize,
            min_distance: f64,
            side: V3,
        }
        let mut pool: Vec<NI> = Vec::with_capacity(64);
        let mut ni1 = {
            let mut ni = NI { node: 0, min_distance: 0.0, side: [0.0; 3] };
            for i in 0..3 {
                let mut s = 0.0f64;
                let t = x[i] - self.maxes[i];
                if t > s {
                    s = t;
                } else {
                    let t2 = self.mins[i] - x[i];
                    if t2 > s {
                        s = t2;
                    }
                }
                let sd = s * s;
                ni.side[i] = 0.0;
                // update_side_distance
                ni.min_distance += sd - ni.side[i];
                ni.side[i] = sd;
            }
            pool.push(ni);
            0usize
        };
        let mut q = Heap::new(12);
        let mut neighbors = Heap::new(k);
        let mut dub = distance_upper_bound * distance_upper_bound;
        if distance_upper_bound.is_infinite() {
            dub = f64::INFINITY;
        }
        let epsfac = 1.0f64; // eps = 0: 1 / (1 * 1)
        loop {
            let nd = self.nodes[pool[ni1].node];
            if nd.split_dim == -1 {
                for i in nd.start..nd.end {
                    let di = self.idx[i];
                    let d = Self::sqdist(&self.data[di], x);
                    if d < dub {
                        if neighbors.n == k {
                            neighbors.remove();
                        }
                        neighbors.push(-d, di);
                        if neighbors.n == k {
                            dub = -neighbors.peek().0;
                        }
                    }
                }
                if q.n == 0 {
                    break;
                } else {
                    let it = q.pop();
                    ni1 = it.1;
                }
            } else {
                let sdim = nd.split_dim as usize;
                let split = nd.split;
                if pool[ni1].min_distance > dub * epsfac {
                    break;
                }
                let side = pool[ni1].side;
                let mind = pool[ni1].min_distance;
                let mut ni2 = NI { node: 0, min_distance: mind, side };
                let sd;
                if x[sdim] < split {
                    pool[ni1].node = nd.less;
                    ni2.node = nd.greater;
                    sd = split - x[sdim];
                } else {
                    pool[ni1].node = nd.greater;
                    ni2.node = nd.less;
                    sd = x[sdim] - split;
                }
                let sd = sd * sd;
                ni2.min_distance += sd - ni2.side[sdim];
                ni2.side[sdim] = sd;
                pool.push(ni2);
                let mut a = ni1;
                let mut b = pool.len() - 1;
                if pool[a].min_distance > pool[b].min_distance {
                    std::mem::swap(&mut a, &mut b);
                }
                ni1 = a;
                if pool[b].min_distance <= dub * epsfac {
                    q.push(pool[b].min_distance, b);
                }
            }
        }
        let nnb = neighbors.n;
        let mut sorted = vec![(0.0f64, 0usize); nnb];
        for i in (0..nnb).rev() {
            sorted[i] = neighbors.pop();
        }
        for i in 0..k {
            if i >= nnb {
                out.push((f64::INFINITY, n));
            } else {
                out.push((-sorted[i].0, sorted[i].1));
            }
        }
        out
    }

    pub fn n(&self) -> usize {
        self.data.len()
    }
}

/// scipy ckdtree query.cxx heap (min-heap on priority, its own sift order)
struct Heap {
    h: Vec<(f64, usize)>,
    n: usize,
}

impl Heap {
    fn new(cap: usize) -> Heap {
        Heap { h: Vec::with_capacity(cap.max(1)), n: 0 }
    }
    fn push(&mut self, pr: f64, c: usize) {
        self.n += 1;
        if self.h.len() < self.n {
            self.h.push((pr, c));
        }
        let mut i = self.n - 1;
        self.h[i] = (pr, c);
        while i > 0 && self.h[i].0 < self.h[(i - 1) / 2].0 {
            self.h.swap(i, (i - 1) / 2);
            i = (i - 1) / 2;
        }
    }
    fn peek(&self) -> (f64, usize) {
        self.h[0]
    }
    fn remove(&mut self) {
        self.h[0] = self.h[self.n - 1];
        self.n -= 1;
        let nn = self.n;
        let (mut i, mut j, mut k) = (0usize, 1usize, 2usize);
        while (j < nn && self.h[i].0 > self.h[j].0) || (k < nn && self.h[i].0 > self.h[k].0) {
            let l = if k < nn && self.h[j].0 > self.h[k].0 { k } else { j };
            self.h.swap(l, i);
            i = l;
            j = 2 * i + 1;
            k = 2 * i + 2;
        }
    }
    fn pop(&mut self) -> (f64, usize) {
        let it = self.h[0];
        self.remove();
        it
    }
}
