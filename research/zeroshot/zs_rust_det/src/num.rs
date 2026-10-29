//! Numerics that reproduce numpy's evaluation order where it is known (pairwise sums, medians, quantiles,
//! arange / linspace fills, histogram bin assignment), plus small dense linear algebra.
#![allow(dead_code)]

pub type V3 = [f64; 3];
use crate::fmath::{fmod, DetF64};

/// sin / cos through the pure-Rust libm (fmath), called separately (the host libm and LLVM's sincos fusion are
/// never involved).
#[inline]
pub fn lsin(x: f64) -> f64 {
    x.dsin()
}
#[inline]
pub fn lcos(x: f64) -> f64 {
    x.dcos()
}
pub type V2 = [f64; 2];

#[inline]
pub fn dot3(a: &V3, b: &V3) -> f64 {
    (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]
}
/// numpy dot of 3-vectors when either operand is a strided view (Accelerate ddot, incx != 1): (p0 + p2) + p1
#[inline]
pub fn dot3_strided(a: &V3, b: &V3) -> f64 {
    (a[0] * b[0] + a[2] * b[2]) + a[1] * b[1]
}
#[inline]
pub fn dot2(a: &V2, b: &V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
#[inline]
pub fn sub3(a: &V3, b: &V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add3(a: &V3, b: &V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale3(a: &V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn neg3(a: &V3) -> V3 {
    [-a[0], -a[1], -a[2]]
}
/// numpy.cross for 3-vectors
#[inline]
pub fn cross3(a: &V3, b: &V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
#[inline]
pub fn norm3(a: &V3) -> f64 {
    ((a[0] * a[0] + a[1] * a[1]) + a[2] * a[2]).sqrt()
}
#[inline]
pub fn norm2(a: &V2) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}
#[inline]
pub fn div3(a: &V3, s: f64) -> V3 {
    [a[0] / s, a[1] / s, a[2] / s]
}
#[inline]
pub fn sq(x: f64) -> f64 {
    x * x
}

/// numpy pairwise summation (umath loops_utils) over a contiguous 1-D reduction.
pub fn pw_sum_by<F: Fn(usize) -> f64>(off: usize, n: usize, f: &F) -> f64 {
    if n < 8 {
        let mut res = 0.0f64;
        // numpy: res = 0.; res += a[i]  -- but -0.0 sums keep sign through "0. + x"
        for i in 0..n {
            res += f(off + i);
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        for j in 0..8 {
            r[j] = f(off + j);
        }
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += f(off + i + j);
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += f(off + i);
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pw_sum_by(off, n2, f) + pw_sum_by(off + n2, n - n2, f)
    }
}

pub fn pw_sum(a: &[f64]) -> f64 {
    pw_sum_by(0, a.len(), &|i| a[i])
}

pub fn mean(a: &[f64]) -> f64 {
    pw_sum(a) / a.len() as f64
}

pub fn mean_bool(a: &[bool]) -> f64 {
    // pairwise sum of 0/1 values is exact
    let c = a.iter().filter(|&&x| x).count();
    c as f64 / a.len() as f64
}

/// numpy std (ddof 0): sqrt(mean(|x - mean|^2)) with pairwise sums
pub fn std(a: &[f64]) -> f64 {
    let m = mean(a);
    let d: Vec<f64> = a.iter().map(|x| {
        let y = (x - m).abs();
        y * y
    }).collect();
    (pw_sum(&d) / a.len() as f64).sqrt()
}

/// Sequential sum over axis 0 of rows (numpy reduction over a non-contiguous axis).
pub fn seq_sum_rows3(rows: impl Iterator<Item = V3>) -> V3 {
    let mut s = [0.0f64; 3];
    for r in rows {
        s[0] += r[0];
        s[1] += r[1];
        s[2] += r[2];
    }
    s
}

pub fn total_cmp(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

/// k-th smallest (0-based) of a slice (copy).
fn kth(v: &mut [f64], k: usize) -> f64 {
    let (_, x, _) = v.select_nth_unstable_by(k, total_cmp);
    *x
}

/// numpy.median (no NaN handling beyond propagation): middle element or mean of the two middle ones.
pub fn median(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    if a.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    let mut v = a.to_vec();
    if n % 2 == 1 {
        kth(&mut v, n / 2)
    } else {
        let hi = kth(&mut v, n / 2);
        // max of the lower part
        let lo = v[..n / 2].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // numpy: mean([lo, hi]) = (0. + lo + hi) / 2
        ((0.0 + lo) + hi) / 2.0
    }
}

/// numpy lerp used by quantile (numpy >= 1.22 `_lerp`).
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// numpy.quantile(a, q) with the default 'linear' method.
pub fn quantile(a: &[f64], q: f64) -> f64 {
    let n = a.len();
    let mut v = a.to_vec();
    let virt = (n as f64 - 1.0) * q;
    let prev = virt.floor();
    let gamma = virt - prev;
    let pi = (prev as i64).clamp(0, n as i64 - 1) as usize;
    let ni = ((prev as i64) + 1).clamp(0, n as i64 - 1) as usize;
    v.sort_by(total_cmp);
    let lo = v[pi];
    let hi = v[ni];
    lerp(lo, hi, gamma)
}

/// several quantiles over one sorted copy
pub fn quantiles_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virt = (n as f64 - 1.0) * q;
    let prev = virt.floor();
    let gamma = virt - prev;
    let pi = (prev as i64).clamp(0, n as i64 - 1) as usize;
    let ni = ((prev as i64) + 1).clamp(0, n as i64 - 1) as usize;
    lerp(sorted[pi], sorted[ni], gamma)
}

/// numpy.arange(start, stop, step) for floats (DOUBLE_fill: x_i = start + i * ((start + step) - start)).
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let len = ((stop - start) / step).ceil();
    if !(len > 0.0) {
        return vec![];
    }
    let n = len as usize;
    let mut out = Vec::with_capacity(n);
    out.push(start);
    if n > 1 {
        let b1 = start + step;
        out.push(b1);
        let delta = b1 - start;
        for i in 2..n {
            out.push(start + (i as f64) * delta);
        }
    }
    out
}

/// numpy searchsorted side='left'
pub fn searchsorted_left(a: &[f64], v: f64) -> usize {
    let (mut lo, mut hi) = (0usize, a.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// numpy.round / Python round on a float: half to even
#[inline]
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// numpy float mod (npy_divmod remainder)
pub fn np_mod(a: f64, b: f64) -> f64 {
    let m = fmod(a, b);
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m + b
        } else {
            m
        }
    } else {
        0.0f64.copysign(b)
    }
}

#[inline]
pub fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    // numpy.clip: minimum(maximum(x, lo), hi)
    let y = if x < lo { lo } else { x };
    if y > hi {
        hi
    } else {
        y
    }
}

/// argmax: first index of the maximum (numpy semantics, NaN aside)
pub fn argmax(a: &[f64]) -> usize {
    let mut k = 0;
    for i in 1..a.len() {
        if a[i] > a[k] {
            k = i;
        }
    }
    k
}
pub fn argmin(a: &[f64]) -> usize {
    let mut k = 0;
    for i in 1..a.len() {
        if a[i] < a[k] {
            k = i;
        }
    }
    k
}

// Small dense linear algebra (eigh, solve, det, lstsq) lives in accel.rs (pure-Rust LAPACK ports).

/// stable argsort ascending by key
pub fn argsort_stable_by<F: Fn(usize) -> f64>(n: usize, key: F) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| total_cmp(&key(a), &key(b)));
    idx
}

#[inline]
fn np_less(a: f64, b: f64) -> bool {
    a < b || (b != b && a == a)
}

/// numpy argsort(kind='quicksort') for float64 on non-x86 hosts: npysort aquicksort_ (introsort, median of 3,
/// insertion sort below 16, heapsort fallback), so the order of equal keys is numpy's.
pub fn np_argsort(v: &[f64]) -> Vec<usize> {
    let num = v.len();
    let mut ts: Vec<usize> = (0..num).collect();
    if num < 2 {
        return ts;
    }
    let small = 15isize;
    let mut stack: Vec<(isize, isize, i32)> = Vec::new();
    let mut pl: isize = 0;
    let mut pr: isize = num as isize - 1;
    let msb = (usize::BITS - 1 - num.leading_zeros()) as i32;
    let mut cdepth: i32 = msb * 2;
    loop {
        let mut do_heap = false;
        if cdepth < 0 {
            do_heap = true;
        } else {
            while pr - pl > small {
                let pm = pl + ((pr - pl) >> 1);
                if np_less(v[ts[pm as usize]], v[ts[pl as usize]]) {
                    ts.swap(pm as usize, pl as usize);
                }
                if np_less(v[ts[pr as usize]], v[ts[pm as usize]]) {
                    ts.swap(pr as usize, pm as usize);
                }
                if np_less(v[ts[pm as usize]], v[ts[pl as usize]]) {
                    ts.swap(pm as usize, pl as usize);
                }
                let vp = v[ts[pm as usize]];
                let mut pi = pl;
                let mut pj = pr - 1;
                ts.swap(pm as usize, pj as usize);
                loop {
                    loop {
                        pi += 1;
                        if !np_less(v[ts[pi as usize]], vp) {
                            break;
                        }
                    }
                    loop {
                        pj -= 1;
                        if !np_less(vp, v[ts[pj as usize]]) {
                            break;
                        }
                    }
                    if pi >= pj {
                        break;
                    }
                    ts.swap(pi as usize, pj as usize);
                }
                let pk = pr - 1;
                ts.swap(pi as usize, pk as usize);
                cdepth -= 1;
                if pi - pl < pr - pi {
                    stack.push((pi + 1, pr, cdepth));
                    pr = pi - 1;
                } else {
                    stack.push((pl, pi - 1, cdepth));
                    pl = pi + 1;
                }
            }
        }
        if do_heap {
            aheapsort(v, &mut ts[pl as usize..=pr as usize]);
        } else {
            let mut pi = pl + 1;
            while pi <= pr {
                let vi = ts[pi as usize];
                let vpv = v[vi];
                let mut pj = pi;
                let mut pk = pi - 1;
                while pj > pl && np_less(vpv, v[ts[pk as usize]]) {
                    ts[pj as usize] = ts[pk as usize];
                    pj -= 1;
                    pk -= 1;
                }
                ts[pj as usize] = vi;
                pi += 1;
            }
        }
        match stack.pop() {
            None => break,
            Some((l, r, d)) => {
                pl = l;
                pr = r;
                cdepth = d;
            }
        }
    }
    ts
}

/// npysort aheapsort_ on an index slice (1-based heap as numpy writes it)
fn aheapsort(v: &[f64], tosort: &mut [usize]) {
    let n = tosort.len();
    if n < 2 {
        return;
    }
    // a = tosort - 1 (1-based)
    let get = |t: &[usize], i: usize| t[i - 1];
    let mut i = n / 2;
    while i > 0 {
        let tmp = get(tosort, i);
        let mut ii = i;
        let mut j = i << 1;
        while j <= n {
            if j < n && np_less(v[get(tosort, j)], v[get(tosort, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(tosort, j)]) {
                tosort[ii - 1] = tosort[j - 1];
                ii = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[ii - 1] = tmp;
        i -= 1;
    }
    let mut nn = n;
    while nn > 1 {
        let tmp = get(tosort, nn);
        tosort[nn - 1] = tosort[0];
        nn -= 1;
        let mut ii = 1;
        let mut j = 2;
        while j <= nn {
            if j < nn && np_less(v[get(tosort, j)], v[get(tosort, j + 1)]) {
                j += 1;
            }
            if np_less(v[tmp], v[get(tosort, j)]) {
                tosort[ii - 1] = tosort[j - 1];
                ii = j;
                j += j;
            } else {
                break;
            }
        }
        tosort[ii - 1] = tmp;
    }
}
