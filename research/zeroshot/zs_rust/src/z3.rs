//! Port of zs_lane_v2/z3s.py (Z3 axis-family slicing), from the family axes on: exact plane sections,
//! chains, corners, tracking, track lines, collinear merging, span refinement, bend-strip evidence and the NMS.
//! Thickness estimation (open3d ray casting) and the family search stay in Python and are inputs here.
//! Every function mirrors its Python counterpart's operation order; comments name the Python function.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::accel;
use crate::kdtree::KdTree;
use crate::lsap::linear_sum_assignment;
use crate::mesh::Mesh;
use crate::morph;
use crate::num::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

#[derive(Deserialize, Clone, Debug)]
pub struct Params {
    pub step_mm: f64,
    pub dp_eps_mm: f64,
    pub merge_deg: f64,
    pub min_flat_mm: f64,
    pub max_arc_mm: f64,
    pub min_bend_deg: f64,
    pub max_bend_deg: f64,
    pub flat_rms_mm: f64,
    pub link_pos_mm: f64,
    pub link_ang_deg: f64,
    pub link_bis_deg: f64,
    pub bridge_mm: f64,
    pub min_track_slices: usize,
    pub max_axis_dev_deg: f64,
    pub dedupe_pos_mm: f64,
}

// ----------------------------------------------------------------------------- slicing

pub struct Chain {
    pub pts: Vec<V2>,
    pub nrm: Vec<V2>,
}

pub fn orthonormal_frame(u: &V3) -> (V3, V3) {
    let a = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let e1 = cross3(u, &a);
    let n = norm3(&e1);
    let e1 = [e1[0] / n, e1[1] / n, e1[2] / n];
    let e2 = cross3(u, &e1);
    (e1, e2)
}

/// slice_family: levels and, per level, the face ids crossing it (ascending face id).
pub fn slice_family(mesh: &Mesh, u: &V3, step: f64) -> (Vec<f64>, Vec<Vec<Chain>>) {
    let h: Vec<f64> = mesh.v.iter().map(|p| dot3(p, u)).collect();
    let hmin = h.iter().cloned().fold(f64::INFINITY, f64::min);
    let hmax = h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let c0 = hmin + 0.5 * step + 1e-4;
    let nlev = ((hmax - c0) / step).floor() as i64 + 1;
    let nlev = nlev.max(0) as usize;
    let levels: Vec<f64> = (0..nlev).map(|k| c0 + step * (k as f64)).collect();
    let mut per: Vec<Vec<u32>> = vec![Vec::new(); nlev];
    for (fi, t) in mesh.f.iter().enumerate() {
        let a = h[t[0] as usize];
        let b = h[t[1] as usize];
        let c = h[t[2] as usize];
        let lo = a.min(b).min(c);
        let hi = a.max(b).max(c);
        let klo = ((lo - c0) / step).ceil() as i64;
        let khi = ((hi - c0) / step).floor() as i64;
        if khi - klo + 1 <= 0 {
            continue;
        }
        for k in klo..=khi {
            if k >= 0 && (k as usize) < nlev {
                per[k as usize].push(fi as u32);
            }
        }
    }
    let (e1, e2) = orthonormal_frame(u);
    let nv = mesh.v.len() as i64;
    let chains: Vec<Vec<Chain>> = per
        .par_iter()
        .enumerate()
        .map(|(k, fi)| {
            if fi.len() < 2 {
                return Vec::new();
            }
            let c = levels[k];
            let m = fi.len();
            let mut pa = Vec::with_capacity(m);
            let mut pb = Vec::with_capacity(m);
            let mut ka = Vec::with_capacity(m);
            let mut kb = Vec::with_capacity(m);
            let mut nrm = Vec::with_capacity(m);
            let p2 = |i: usize| -> V2 { [dot3(&mesh.v[i], &e1), dot3(&mesh.v[i], &e2)] };
            for &f in fi {
                let tri = mesh.f[f as usize];
                let above = [h[tri[0] as usize] > c, h[tri[1] as usize] > c, h[tri[2] as usize] > c];
                let cnt = above.iter().filter(|&&x| x).count();
                let lone_is_above = cnt == 1;
                let mut lone = 0;
                for j in 0..3 {
                    let v = if lone_is_above { above[j] } else { !above[j] };
                    if v {
                        lone = j;
                        break;
                    }
                }
                let a = tri[lone] as usize;
                let b = tri[(lone + 1) % 3] as usize;
                let d = tri[(lone + 2) % 3] as usize;
                let ipt = |p: usize, q: usize| -> V2 {
                    let hp = h[p];
                    let hq = h[q];
                    let t = (c - hp) / (hq - hp);
                    let pp = p2(p);
                    let pq = p2(q);
                    [pp[0] + (pq[0] - pp[0]) * t, pp[1] + (pq[1] - pp[1]) * t]
                };
                pa.push(ipt(a, b));
                pb.push(ipt(a, d));
                let (ai, bi, di) = (a as i64, b as i64, d as i64);
                ka.push(ai.min(bi) * nv + ai.max(bi));
                kb.push(ai.min(di) * nv + ai.max(di));
                let fnv = mesh.fnrm[f as usize];
                nrm.push([dot3(&fnv, &e1), dot3(&fnv, &e2)]);
            }
            chain(&pa, &pb, &ka, &kb, &nrm)
        })
        .collect();
    (levels, chains)
}

/// _chain
fn chain(pa: &[V2], pb: &[V2], ka: &[i64], kb: &[i64], nrm: &[V2]) -> Vec<Chain> {
    let m = pa.len();
    let mut order: Vec<usize> = (0..2 * m).collect();
    let key = |i: usize| if i < m { ka[i] } else { kb[i - m] };
    order.sort_by_key(|&i| key(i)); // stable
    let mut nb = vec![[-1i64; 2]; m];
    for i in 0..(2 * m).saturating_sub(1) {
        if key(order[i + 1]) == key(order[i]) {
            let (o1, o2) = (order[i], order[i + 1]);
            let (s1, e1) = (o1 % m, o1 / m);
            let (s2, e2) = (o2 % m, o2 / m);
            if s1 != s2 {
                nb[s1][e1] = s2 as i64;
                nb[s2][e2] = s1 as i64;
            }
        }
    }
    let mut visited = vec![false; m];
    let mut out = Vec::new();
    let mut starts: Vec<usize> = (0..m).filter(|&s| nb[s][0] < 0 || nb[s][1] < 0).collect();
    starts.extend(0..m);
    for s0 in starts {
        if visited[s0] {
            continue;
        }
        let entry = if nb[s0][0] < 0 {
            0
        } else if nb[s0][1] < 0 {
            1
        } else {
            0
        };
        let mut seq = Vec::new();
        let mut ent = Vec::new();
        let mut s = s0 as i64;
        let mut en = entry;
        while s >= 0 && !visited[s as usize] {
            let su = s as usize;
            visited[su] = true;
            seq.push(su);
            ent.push(en);
            let ex = 1 - en;
            let nxt = nb[su][ex];
            if nxt < 0 {
                break;
            }
            let k = if ex == 0 { ka[su] } else { kb[su] };
            en = if ka[nxt as usize] == k { 0 } else { 1 };
            s = nxt;
        }
        if seq.len() < 3 {
            continue;
        }
        let mut pts = Vec::with_capacity(seq.len() + 1);
        for (j, &sg) in seq.iter().enumerate() {
            pts.push(if ent[j] == 0 { pa[sg] } else { pb[sg] });
        }
        let last = *seq.last().unwrap();
        pts.push(if *ent.last().unwrap() == 0 { pb[last] } else { pa[last] });
        let nr: Vec<V2> = seq.iter().map(|&sg| nrm[sg]).collect();
        out.push(Chain { pts, nrm: nr });
    }
    out
}

// ----------------------------------------------------------------------------- corners

fn dp(q: &[V2], eps: f64) -> Vec<usize> {
    let mut keep = vec![0, q.len() - 1];
    let mut stack = vec![(0usize, q.len() - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let d = [q[b][0] - q[a][0], q[b][1] - q[a][1]];
        let l = norm2(&d);
        let mut best = 0usize;
        let mut bd = f64::NEG_INFINITY;
        for (k, p) in q[a + 1..b].iter().enumerate() {
            let rel = [p[0] - q[a][0], p[1] - q[a][1]];
            let dist = if l < 1e-9 { norm2(&rel) } else { (d[0] * rel[1] - d[1] * rel[0]).abs() / l };
            if k == 0 || dist > bd {
                bd = dist;
                best = k;
            }
        }
        if bd > eps {
            let m = a + 1 + best;
            keep.push(m);
            stack.push((a, m));
            stack.push((m, b));
        }
    }
    keep.sort();
    keep
}

/// _tls: centroid, principal direction, rms of the residual (X.T @ X and eigh as numpy computes them)
fn tls(q: &[V2]) -> (V2, V2, f64) {
    let n = q.len() as f64;
    let mut c = [0.0, 0.0];
    for p in q {
        c[0] += p[0];
        c[1] += p[1];
    }
    let c = [c[0] / n, c[1] / n];
    let mut x = Vec::with_capacity(q.len() * 2);
    for p in q {
        x.push(p[0] - c[0]);
        x.push(p[1] - c[1]);
    }
    let m = accel::xtx(&x, q.len(), 2);
    let (w, v) = accel::eigh(&m, 2);
    let d = [v[1], v[3]];
    let rms = (w[0].max(0.0) / n.max(1.0)).sqrt();
    (c, d, rms)
}

#[derive(Clone)]
struct Flat {
    i0: usize,
    i1: usize,
    c: V2,
    d: V2,
    len: f64,
    rms: f64,
    n: V2,
    wall: bool,
}

#[derive(Clone, Debug)]
pub struct Corner {
    pub apex: V2,
    pub bis: V2,
    pub incl: f64,
    pub la: f64,
    pub lb: f64,
    pub rms: f64,
    pub gap: f64,
    pub skin: u8, // 0 outer, 1 mixed, 2 inner
    pub da: V2,
    pub db: V2,
    pub apex_o: V2,
    pub both_skins: bool,
}

fn chain_corners(ch: &Chain, prm: &Params, t: f64) -> Vec<Corner> {
    let q = &ch.pts;
    if q.len() < 6 {
        return vec![];
    }
    let seglen: Vec<f64> = (0..q.len() - 1).map(|i| norm2(&[q[i + 1][0] - q[i][0], q[i + 1][1] - q[i][1]])).collect();
    let mut arc = Vec::with_capacity(q.len());
    arc.push(0.0);
    let mut acc = 0.0;
    for s in &seglen {
        acc += s;
        arc.push(acc);
    }
    if arc[arc.len() - 1] < 2.0 * prm.min_flat_mm {
        return vec![];
    }
    let eps = prm.dp_eps_mm.max(0.06 * t);
    let idx = dp(q, eps);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let cmerge = prm.merge_deg.to_radians().dcos();
    for w in idx.windows(2) {
        let (a, b) = (w[0], w[1]);
        if let Some(&(i0, i1)) = runs.last() {
            let dpv = [q[i1][0] - q[i0][0], q[i1][1] - q[i0][1]];
            let dcv = [q[b][0] - q[a][0], q[b][1] - q[a][1]];
            let cosang = dot2(&dpv, &dcv) / (norm2(&dpv) * norm2(&dcv) + 1e-12);
            if cosang > cmerge {
                let l = runs.len();
                runs[l - 1] = (i0, b);
                continue;
            }
        }
        runs.push((a, b));
    }
    let mut flats: Vec<Flat> = Vec::new();
    for &(i0, i1) in &runs {
        let l = arc[i1] - arc[i0];
        if l < prm.min_flat_mm {
            continue;
        }
        let trim = 0.3f64.min(0.15 * l);
        let lo = arc[i0] + trim;
        let hi = arc[i1] - trim;
        let mut sel: Vec<usize> = (0..arc.len()).filter(|&k| arc[k] >= lo && arc[k] <= hi).collect();
        if sel.len() < 3 {
            sel = (i0..=i1).collect();
        }
        let qs: Vec<V2> = sel.iter().map(|&k| q[k]).collect();
        let (c, mut d, rms) = tls(&qs);
        if rms > prm.flat_rms_mm {
            continue;
        }
        let ddir = [q[i1][0] - q[i0][0], q[i1][1] - q[i0][1]];
        if dot2(&d, &ddir) < 0.0 {
            d = [-d[0], -d[1]];
        }
        if sel.len() >= 6 {
            let mut a_rows = Vec::with_capacity(qs.len());
            let mut r = Vec::with_capacity(qs.len());
            for p in &qs {
                let x0 = p[0] - c[0];
                let x1 = p[1] - c[1];
                let x = x0 * d[0] + x1 * d[1];
                r.push(d[0] * x1 - d[1] * x0);
                a_rows.push([x * x, x, 1.0]);
            }
            let coef = accel::lstsq3(&a_rows, &r);
            let res: Vec<f64> = a_rows
                .iter()
                .zip(&r)
                .map(|(row, rv)| {
                    let v = (row[0] * coef[0] + row[1] * coef[1]) + row[2] * coef[2] - rv;
                    v * v
                })
                .collect();
            let noise = mean(&res).sqrt();
            let sag = coef[0].abs() * (0.5 * l) * (0.5 * l);
            if coef[0].abs() > 1e-9 {
                let rfit = 1.0 / (2.0 * coef[0].abs());
                if sag > 0.05f64.max(3.0 * noise) && rfit < (8.0 * l).min(40.0) {
                    continue;
                }
            }
        }
        let mut nn = [0.0, 0.0];
        for s in i0..i1 {
            nn[0] += ch.nrm[s][0] * seglen[s];
            nn[1] += ch.nrm[s][1] * seglen[s];
        }
        let nl = norm2(&nn);
        flats.push(Flat {
            i0,
            i1,
            c,
            d,
            len: l,
            rms,
            n: if nl > 1e-9 { [nn[0] / nl, nn[1] / nl] } else { [0.0, 0.0] },
            wall: false,
        });
    }
    // sheet-edge walls
    let wall_max = 1.5 * t + 0.8;
    let nf = flats.len();
    for k in 0..nf {
        let f = flats[k].clone();
        if f.len > wall_max {
            continue;
        }
        let prv = if k > 0 && arc[f.i0] - arc[flats[k - 1].i1] < 2.0 * t + 2.0 { Some(k - 1) } else { None };
        let nxt = if k + 1 < nf && arc[flats[k + 1].i0] - arc[f.i1] < 2.0 * t + 2.0 { Some(k + 1) } else { None };
        if let (Some(p), Some(n)) = (prv, nxt) {
            if dot2(&flats[p].d, &flats[n].d) < -0.85 {
                flats[k].wall = true;
            }
        } else if prv.is_some() != nxt.is_some() && f.len <= 1.3 * t + 0.6 {
            let nbk = prv.or(nxt).unwrap();
            let end_gap = if nxt.is_none() { arc[arc.len() - 1] - arc[f.i1] } else { arc[f.i0] };
            if dot2(&flats[nbk].d, &f.d).abs() < 0.35 && end_gap < 1.5 {
                flats[k].wall = true;
            }
        }
    }
    if flats.iter().any(|f| f.wall) {
        flats.retain(|f| !f.wall);
    }
    let turn = |f1: &Flat, f2: &Flat| -> f64 {
        let cr = f1.d[0] * f2.d[1] - f1.d[1] * f2.d[0];
        cr.datan2(dot2(&f1.d, &f2.d)).to_degrees()
    };
    let short = 2.5 * t + 2.0;
    let mut changed = true;
    while changed && flats.len() >= 3 {
        changed = false;
        for k in 1..flats.len() - 1 {
            let (f0, f1, f2) = (&flats[k - 1], &flats[k], &flats[k + 1]);
            if f1.len >= short {
                continue;
            }
            if arc[f1.i0] - arc[f0.i1] > prm.max_arc_mm || arc[f2.i0] - arc[f1.i1] > prm.max_arc_mm {
                continue;
            }
            let a1 = turn(f0, f1);
            let a2 = turn(f1, f2);
            if a1 * a2 > 0.0 && a1.abs().min(a2.abs()) < 35.0 && a1.abs() + a2.abs() < 150.0 {
                flats.remove(k);
                changed = true;
                break;
            }
        }
    }
    let mut out = Vec::new();
    let c6 = 6.0f64.to_radians().dcos();
    for w in 0..flats.len().saturating_sub(1) {
        let fa = &flats[w];
        let fb = &flats[w + 1];
        let gap = arc[fb.i0] - arc[fa.i1];
        if gap > prm.max_arc_mm {
            continue;
        }
        let dd = dot2(&fa.d, &fb.d);
        if dd.abs() > c6 && dd > 0.0 {
            let rel = [q[fb.i0][0] - q[fa.i1][0], q[fb.i0][1] - q[fa.i1][1]];
            let lat = (fa.d[0] * rel[1] - fa.d[1] * rel[0]).abs();
            if 0.5 * t <= lat && lat <= 4.0 * t + 4.0 && gap <= (4.0 * t + 6.0).max(12.0) {
                let cl = norm2(&rel);
                if cl > 1e-6 {
                    let mut nn = [0.0, 0.0];
                    for s in fa.i1..fb.i0 {
                        nn[0] += ch.nrm[s][0];
                        nn[1] += ch.nrm[s][1];
                    }
                    let nl = norm2(&nn);
                    let web = Flat {
                        i0: fa.i1,
                        i1: fb.i0,
                        c: [0.5 * (q[fa.i1][0] + q[fb.i0][0]), 0.5 * (q[fa.i1][1] + q[fb.i0][1])],
                        d: [rel[0] / cl, rel[1] / cl],
                        len: cl,
                        rms: 0.0,
                        n: if nl > 1e-9 { [nn[0] / nl, nn[1] / nl] } else { [0.0, 0.0] },
                        wall: false,
                    };
                    for (x, y) in [(fa, &web), (&web, fb)] {
                        if let Some(c) = make_corner(x, y, q, &arc, prm, t) {
                            out.push(c);
                        }
                    }
                }
            }
            continue;
        }
        if let Some(c) = make_corner(fa, fb, q, &arc, prm, t) {
            out.push(c);
        }
    }
    out
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else if x == 0.0 {
        0.0
    } else {
        f64::NAN
    }
}

fn make_corner(fa: &Flat, fb: &Flat, q: &[V2], arc: &[f64], prm: &Params, t: f64) -> Option<Corner> {
    let gap = arc[fb.i0] - arc[fa.i1];
    let da = [-fa.d[0], -fa.d[1]];
    let db = fb.d;
    let cosi = clip(dot2(&da, &db), -1.0, 1.0);
    let incl = cosi.dacos().to_degrees();
    let bend = 180.0 - incl;
    if bend < prm.min_bend_deg || bend > prm.max_bend_deg {
        return None;
    }
    let st = accel::solve(&[fa.d[0], -fb.d[0], fa.d[1], -fb.d[1]], &[fb.c[0] - fa.c[0], fb.c[1] - fa.c[1]], 2)?;
    let sharp = [fa.c[0] + st[0] * fa.d[0], fa.c[1] + st[0] * fa.d[1]];
    let jn = [0.5 * (q[fa.i1][0] + q[fb.i0][0]), 0.5 * (q[fa.i1][1] + q[fb.i0][1])];
    if norm2(&[sharp[0] - jn[0], sharp[1] - jn[1]]) > gap + 4.0 * t + 6.0 {
        return None;
    }
    let bis = [da[0] + db[0], da[1] + db[1]];
    let bl = norm2(&bis);
    if bl < 1e-6 {
        return None;
    }
    let bis = [bis[0] / bl, bis[1] / bl];
    let j0 = fa.i0.max(searchsorted_left(arc, arc[fa.i1] - 2.0 * t - 1.0));
    let j1 = fb.i1.min(searchsorted_left(arc, arc[fb.i0] + 2.0 * t + 1.0));
    let w: Vec<V2> = (j0..=j1).map(|k| [q[k][0] - sharp[0], q[k][1] - sharp[1]]).collect();
    let side: Vec<f64> = w.iter().map(|p| bis[0] * p[1] - bis[1] * p[0]).collect();
    let hh: Vec<f64> = w.iter().map(|p| p[0] * bis[0] + p[1] * bis[1]).collect();
    let mut apex: Option<V2> = None;
    let mut best: Option<(f64, V2)> = None;
    for i in 0..side.len().saturating_sub(1) {
        if sign(side[i]) * sign(side[i + 1]) <= 0.0 {
            let (s0, s1) = (side[i], side[i + 1]);
            let tt = if s0 == s1 { 0.0 } else { s0 / (s0 - s1) };
            let hv = hh[i] + tt * (hh[i + 1] - hh[i]);
            if hv < -1.0 {
                return None;
            }
            if best.is_none() || hv < best.unwrap().0 {
                best = Some((hv, [w[i][0] + tt * (w[i + 1][0] - w[i][0]), w[i][1] + tt * (w[i + 1][1] - w[i][1])]));
            }
        }
    }
    if let Some((_, bw)) = best {
        apex = Some([sharp[0] + bw[0], sharp[1] + bw[1]]);
    }
    let apex = match apex {
        Some(a) => a,
        None => {
            if side.is_empty() {
                return None;
            }
            let score: Vec<f64> = side.iter().zip(&hh).map(|(s, h)| s.abs() + 0.1 * h.max(0.0)).collect();
            let k = argmin(&score);
            [sharp[0] + w[k][0], sharp[1] + w[k][1]]
        }
    };
    let sa = dot2(&fa.n, &bis);
    let sb = dot2(&fb.n, &bis);
    let skin = if sa < -0.2 && sb < -0.2 {
        0
    } else if sa > 0.2 && sb > 0.2 {
        2
    } else {
        1
    };
    let (mut da, mut db, mut la, mut lb) = (da, db, fa.len, fb.len);
    if da[0] * db[1] - da[1] * db[0] < 0.0 {
        std::mem::swap(&mut da, &mut db);
        std::mem::swap(&mut la, &mut lb);
    }
    Some(Corner {
        apex,
        bis,
        incl,
        la,
        lb,
        rms: fa.rms.max(fb.rms),
        gap,
        skin,
        da,
        db,
        apex_o: apex,
        both_skins: false,
    })
}

/// slice_corners
fn slice_corners(chains: &[Chain], prm: &Params, t: f64) -> Vec<Corner> {
    let mut cs: Vec<Corner> = Vec::new();
    for ch in chains {
        cs.extend(chain_corners(ch, prm, t));
    }
    for c in cs.iter_mut() {
        c.apex_o = if c.skin == 2 { [c.apex[0] - t * c.bis[0], c.apex[1] - t * c.bis[1]] } else { c.apex };
    }
    cs.sort_by_key(|c| c.skin); // stable
    let mut kept: Vec<Corner> = Vec::new();
    for c in cs {
        let mut dup = None;
        for (ki, k) in kept.iter().enumerate() {
            if norm2(&[k.apex_o[0] - c.apex_o[0], k.apex_o[1] - c.apex_o[1]]) < prm.dedupe_pos_mm + 0.5 * t
                && (k.incl - c.incl).abs() < 8.0
                && dot2(&k.bis, &c.bis) > 0.8
            {
                dup = Some(ki);
                break;
            }
        }
        match dup {
            None => {
                let mut c = c;
                c.both_skins = false;
                kept.push(c);
            }
            Some(ki) => {
                let k = &mut kept[ki];
                k.both_skins = k.both_skins || (k.skin != c.skin);
                k.la = if k.la >= c.la { k.la } else { c.la };
                k.lb = if k.lb >= c.lb { k.lb } else { c.lb };
            }
        }
    }
    kept
}

// ----------------------------------------------------------------------------- tracking

pub type Track = Vec<(usize, Corner)>;

fn track_corners(per_level: &[Vec<Corner>], prm: &Params) -> Vec<Track> {
    let step = prm.step_mm;
    let max_gap = (round_even(prm.bridge_mm / step) as i64).max(1);
    struct Act {
        pts: Track,
        last: usize,
    }
    let mut active: Vec<Act> = Vec::new();
    let mut done: Vec<Track> = Vec::new();
    let cbis = prm.link_bis_deg.to_radians().dcos();
    for (k, cs) in per_level.iter().enumerate() {
        let mut still = Vec::with_capacity(active.len());
        for tr in active.drain(..) {
            if (k as i64) - (tr.last as i64) > max_gap + 1 {
                done.push(tr.pts);
            } else {
                still.push(tr);
            }
        }
        active = still;
        if cs.is_empty() {
            continue;
        }
        let mut used = vec![false; cs.len()];
        if !active.is_empty() {
            let (na, nc) = (active.len(), cs.len());
            let mut cost = vec![f64::INFINITY; na * nc];
            let mut any = false;
            for (i, tr) in active.iter().enumerate() {
                let gapk = k - tr.last;
                let lc = &tr.pts.last().unwrap().1;
                let tol = prm.link_pos_mm + 0.15 * step * (gapk as f64);
                for (j, c) in cs.iter().enumerate() {
                    let dpv = norm2(&[c.apex_o[0] - lc.apex_o[0], c.apex_o[1] - lc.apex_o[1]]);
                    let da = (c.incl - lc.incl).abs();
                    if dpv > tol || da > prm.link_ang_deg || dot2(&c.bis, &lc.bis) < cbis {
                        continue;
                    }
                    cost[i * nc + j] = dpv / tol + da / prm.link_ang_deg + 0.3 * ((gapk as i64 - 1) as f64);
                    any = true;
                }
            }
            if any {
                let cc: Vec<f64> = cost.iter().map(|&x| if x.is_finite() { x } else { 1e6 }).collect();
                let (ri, ci) = linear_sum_assignment(na, nc, &cc);
                for (&i, &j) in ri.iter().zip(&ci) {
                    if cost[i * nc + j].is_finite() {
                        active[i].pts.push((k, cs[j].clone()));
                        active[i].last = k;
                        used[j] = true;
                    }
                }
            }
        }
        for (j, c) in cs.iter().enumerate() {
            if !used[j] {
                active.push(Act { pts: vec![(k, c.clone())], last: k });
            }
        }
    }
    done.extend(active.into_iter().map(|a| a.pts));
    done
}

#[derive(Clone)]
pub struct Seg {
    family: usize,
    p: Vec<V3>,
    lev: Vec<f64>,
    incl: Vec<f64>,
    fmin: Vec<f64>,
    fmax: Vec<f64>,
    fla: Vec<f64>,
    flb: Vec<f64>,
    outer: Vec<bool>,
    inner: Vec<bool>,
    both: Vec<bool>,
    rms: Vec<f64>,
    gap: Vec<f64>,
    bis: Vec<V3>,
    da: Vec<V3>,
    db: Vec<V3>,
    n_parts: usize,
}

fn lift2(v: &V2, e1: &V3, e2: &V3) -> V3 {
    // c[0] * e1 + c[1] * e2
    [v[0] * e1[0] + v[1] * e2[0], v[0] * e1[1] + v[1] * e2[1], v[0] * e1[2] + v[1] * e2[2]]
}

fn tracks_to_segments(tracks: &[Track], levels: &[f64], u: &V3, prm: &Params, fam: usize) -> Vec<Seg> {
    let (e1, e2) = orthonormal_frame(u);
    let mut out = Vec::new();
    for tr in tracks {
        if tr.len() < prm.min_track_slices {
            continue;
        }
        let mut s = Seg {
            family: fam,
            p: vec![],
            lev: vec![],
            incl: vec![],
            fmin: vec![],
            fmax: vec![],
            fla: vec![],
            flb: vec![],
            outer: vec![],
            inner: vec![],
            both: vec![],
            rms: vec![],
            gap: vec![],
            bis: vec![],
            da: vec![],
            db: vec![],
            n_parts: 1,
        };
        for (k, c) in tr {
            let lv = levels[*k];
            // _lift: level * u + p2[0] * e1 + p2[1] * e2
            let p = [
                (lv * u[0] + c.apex_o[0] * e1[0]) + c.apex_o[1] * e2[0],
                (lv * u[1] + c.apex_o[0] * e1[1]) + c.apex_o[1] * e2[1],
                (lv * u[2] + c.apex_o[0] * e1[2]) + c.apex_o[1] * e2[2],
            ];
            s.p.push(p);
            s.lev.push(lv);
            s.incl.push(c.incl);
            s.fmin.push(if c.la <= c.lb { c.la } else { c.lb });
            s.fmax.push(if c.la >= c.lb { c.la } else { c.lb });
            s.fla.push(c.la);
            s.flb.push(c.lb);
            s.outer.push(c.skin == 0);
            s.inner.push(c.skin == 2);
            s.both.push(c.both_skins);
            s.rms.push(c.rms);
            s.gap.push(c.gap);
            s.bis.push(lift2(&c.bis, &e1, &e2));
            s.da.push(lift2(&c.da, &e1, &e2));
            s.db.push(lift2(&c.db, &e1, &e2));
        }
        out.push(s);
    }
    out
}

fn unit3(v: &V3) -> V3 {
    let n = norm3(v);
    if n > 1e-12 {
        div3(v, n)
    } else {
        *v
    }
}

#[derive(Clone, Debug)]
pub struct Line {
    pub family: usize,
    pub axis: V3,
    pub start: V3,
    pub end: V3,
    pub dir: V3,
    pub length: f64,
    pub n_support: usize,
    pub n_slices: usize,
    pub support_frac: f64,
    pub incl: f64,
    pub incl_std: f64,
    pub flank_min: f64,
    pub flank_max: f64,
    pub flank_a: f64,
    pub flank_b: f64,
    pub res_rms: f64,
    pub axis_dev_deg: f64,
    pub frac_outer: f64,
    pub frac_inner: f64,
    pub frac_both: f64,
    pub flat_rms: f64,
    pub gap_med: f64,
    pub bis: V3,
    pub da: V3,
    pub db: V3,
    pub n_parts: usize,
    pub strip_frac: f64,
    pub flank_frac: f64,
    pub quality: f64,
    pub n_dups: usize,
}

fn mean_rows(v: &[V3]) -> V3 {
    let s = seq_sum_rows3(v.iter().cloned());
    let n = v.len() as f64;
    [s[0] / n, s[1] / n, s[2] / n]
}

/// segment_line
fn segment_line(seg: &Seg, u: &V3, prm: &Params) -> Option<Line> {
    let p = &seg.p;
    let n = p.len();
    let c0 = mean_rows(p);
    let mut d = *u;
    // numpy: `d = v[:, 2]` is a strided view; a dot of a strided 3-vector goes through Accelerate's generic ddot
    // loop, (p0 + p2) + p1. A negated copy is contiguous again: (p0 + p1) + p2.
    let mut strided = false;
    if n >= 4 {
        let mut x = Vec::with_capacity(n * 3);
        for q in p {
            let r = sub3(q, &c0);
            x.extend_from_slice(&r);
        }
        let m = accel::xtx(&x, n, 3);
        let (w, v) = accel::eigh(&m, 3);
        let spread = (w[2].max(0.0) / n as f64).sqrt();
        if spread >= 2.5 {
            let dpv = [v[2], v[5], v[8]];
            if dot3_strided(&dpv, u).abs() < prm.max_axis_dev_deg.to_radians().dcos() {
                return None;
            }
            d = dpv;
            strided = true;
        }
    }
    let du = if strided { dot3_strided(&d, u) } else { dot3(&d, u) };
    if du < 0.0 {
        d = neg3(&d);
        strided = false;
    }
    let du = if strided { dot3_strided(&d, u) } else { dot3(&d, u) };
    let s: Vec<f64> = p.iter().map(|q| dot3(&sub3(q, &c0), &d)).collect();
    let res2: Vec<f64> = p
        .iter()
        .zip(&s)
        .map(|(q, &si)| {
            let x = sub3(q, &c0);
            let r = [x[0] - si * d[0], x[1] - si * d[1], x[2] - si * d[2]];
            let nr = norm3(&r);
            nr * nr
        })
        .collect();
    let rr = mean(&res2).sqrt();
    if rr > 1.5 {
        return None;
    }
    let half = 0.5 * prm.step_mm / du.abs().max(0.5);
    let smin = s.iter().cloned().fold(f64::INFINITY, f64::min);
    let smax = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lo = smin - half;
    let hi = smax + half;
    let lmax = seg.lev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lmin = seg.lev.iter().cloned().fold(f64::INFINITY, f64::min);
    let nsl = round_even((lmax - lmin) / prm.step_mm) as i64 + 1;
    let mut keys: Vec<f64> = seg.lev.iter().map(|l| round_even(l / prm.step_mm)).collect();
    keys.sort_by(total_cmp);
    keys.dedup();
    let nsup = keys.len();
    let bis = unit3(&mean_rows(&seg.bis));
    let da = unit3(&mean_rows(&seg.da));
    let db = unit3(&mean_rows(&seg.db));
    let bmean = |v: &[bool]| mean_bool(v);
    Some(Line {
        family: seg.family,
        axis: *u,
        start: [c0[0] + lo * d[0], c0[1] + lo * d[1], c0[2] + lo * d[2]],
        end: [c0[0] + hi * d[0], c0[1] + hi * d[1], c0[2] + hi * d[2]],
        dir: d,
        length: hi - lo,
        n_support: nsup,
        n_slices: nsl as usize,
        support_frac: 1.0f64.min(nsup as f64 / (nsl.max(1) as f64)),
        incl: median(&seg.incl),
        incl_std: std(&seg.incl),
        flank_min: median(&seg.fmin),
        flank_max: median(&seg.fmax),
        flank_a: median(&seg.fla),
        flank_b: median(&seg.flb),
        res_rms: rr,
        axis_dev_deg: 1.0f64.min(du.abs()).dacos().to_degrees(),
        frac_outer: bmean(&seg.outer),
        frac_inner: bmean(&seg.inner),
        frac_both: bmean(&seg.both),
        flat_rms: median(&seg.rms),
        gap_med: median(&seg.gap),
        bis,
        da,
        db,
        n_parts: seg.n_parts,
        strip_frac: 0.0,
        flank_frac: 0.0,
        quality: 0.0,
        n_dups: 0,
    })
}

fn concat(a: &Seg, b: &Seg) -> Seg {
    fn c<T: Clone>(x: &[T], y: &[T]) -> Vec<T> {
        let mut v = x.to_vec();
        v.extend_from_slice(y);
        v
    }
    Seg {
        family: a.family,
        p: c(&a.p, &b.p),
        lev: c(&a.lev, &b.lev),
        incl: c(&a.incl, &b.incl),
        fmin: c(&a.fmin, &b.fmin),
        fmax: c(&a.fmax, &b.fmax),
        fla: c(&a.fla, &b.fla),
        flb: c(&a.flb, &b.flb),
        outer: c(&a.outer, &b.outer),
        inner: c(&a.inner, &b.inner),
        both: c(&a.both, &b.both),
        rms: c(&a.rms, &b.rms),
        gap: c(&a.gap, &b.gap),
        bis: c(&a.bis, &b.bis),
        da: c(&a.da, &b.da),
        db: c(&a.db, &b.db),
        n_parts: a.n_parts + b.n_parts,
    }
}

pub struct Ctx<'a> {
    pub mesh: &'a Mesh,
    pub tree: &'a KdTree,
}

/// same_plates
fn same_plates(
    ctx: &Ctx,
    p0: &V3,
    d: &V3,
    da: &V3,
    db: &V3,
    s0: f64,
    s1: f64,
    t: f64,
    a_max: f64,
    flank_len: (f64, f64),
) -> (bool, bool) {
    let cell = 1.5;
    let pad = 8.0;
    let a0 = 2.0 * t + 2.0;
    let a1 = (a0 + 6.0).max(a_max);
    let hs = 0.5 * (s0 + s1);
    let mid = [p0[0] + hs * d[0], p0[1] + hs * d[1], p0[2] + hs * d[2]];
    let rad = (sq(0.5 * (s1 - s0) + pad) + sq(a1)).sqrt() + 1.0;
    let idx = ctx.tree.ball(&mid, rad);
    if idx.is_empty() {
        return (false, false);
    }
    let mut qv = Vec::new();
    let mut nv = Vec::new();
    let mut sv = Vec::new();
    for &i in &idx {
        let q = sub3(&ctx.mesh.v[i], p0);
        let s = dot3(&q, d);
        if s >= s0 - pad && s <= s1 + pad {
            qv.push(q);
            nv.push(ctx.mesh.vn[i]);
            sv.push(s);
        }
    }
    let ns = ((s1 - s0 + 2.0 * pad) / cell).ceil() as i64 + 1;
    let na = ((a1 - a0) / cell).ceil() as i64 + 1;
    let mut out = [false, false];
    for (fi, f) in [da, db].iter().enumerate() {
        let (mut b0, mut b1) = (a0, a1);
        let fl = if fi == 0 { flank_len.0 } else { flank_len.1 };
        if fl < a0 + 6.0 {
            b0 = a0.min(0.4 * fl);
            b1 = (b0 + 2.0).max(1.1 * fl);
        }
        let nrm = unit3(&cross3(d, f));
        let mut cells = Vec::new();
        for k in 0..qv.len() {
            let a = dot3(&qv[k], f);
            let h = dot3(&qv[k], &nrm).abs();
            if a >= b0 && a <= b1 && h <= t + 1.5 + 0.03 * a && dot3(&nv[k], &nrm).abs() > 0.9 {
                let gi = ((((sv[k] - (s0 - pad)) / cell) as i64).max(0)).min(ns - 1);
                let gj = ((((a - b0) / cell) as i64).max(0)).min(na - 1);
                cells.push((gi as usize, gj as usize));
            }
        }
        if cells.len() < 4 {
            out[fi] = false;
            continue;
        }
        let (ns_u, na_u) = (ns as usize, na as usize);
        let mut grid = vec![false; ns_u * na_u];
        for (i, j) in cells {
            grid[i * na_u + j] = true;
        }
        let grid = morph::closing2x2(&grid, ns_u, na_u);
        let band = (1i64.max((pad / cell) as i64)) as usize;
        out[fi] = morph::rows_connected8(&grid, ns_u, na_u, 0, band.min(ns_u), ns_u.saturating_sub(band), ns_u);
    }
    (out[0], out[1])
}

/// merge_segments (with memoised pair verdicts; the scan order and the restart after each merge are
/// unchanged, so the result equals the Python loop).
fn merge_segments(segs: Vec<Seg>, u: &V3, prm: &Params, ctx: &Ctx, t: f64, max_gap_mm: f64) -> Vec<Line> {
    struct Item {
        seg: Seg,
        ln: Line,
        uid: usize,
    }
    let mut items: Vec<Item> = Vec::new();
    let mut next_uid = 0usize;
    for s in segs {
        if let Some(ln) = segment_line(&s, u, prm) {
            items.push(Item { seg: s, ln, uid: next_uid });
            next_uid += 1;
        }
    }
    let c3 = 3.0f64.to_radians().dcos();
    let try_pair = |a_item: &Item, b_item: &Item| -> Option<(Seg, Line)> {
        let a = &a_item.ln;
        let b = &b_item.ln;
        if a.family != b.family {
            return None;
        }
        let pa0 = a.start;
        let pa1 = a.end;
        let d = unit3(&sub3(&pa1, &pa0));
        let db_ = unit3(&sub3(&b.end, &b.start));
        if dot3(&d, &db_).abs() < c3 {
            return None;
        }
        if (a.incl - b.incl).abs() > 7.0 || dot3(&a.bis, &b.bis) < 0.9 {
            return None;
        }
        let off: Vec<f64> = b_item
            .seg
            .p
            .iter()
            .map(|q| {
                let pb = sub3(q, &pa0);
                let s = dot3(&pb, &d);
                norm3(&[pb[0] - s * d[0], pb[1] - s * d[1], pb[2] - s * d[2]])
            })
            .collect();
        if median(&off) > 1.2 + 0.3 * t {
            return None;
        }
        let sa_ = {
            let x = dot3(&sub3(&pa1, &pa0), &d);
            if x < 0.0 { [x, 0.0] } else { [0.0, x] }
        };
        let sb_ = {
            let x = dot3(&sub3(&b.start, &pa0), &d);
            let y = dot3(&sub3(&b.end, &pa0), &d);
            if x <= y { [x, y] } else { [y, x] }
        };
        let (gap0, gap1) = if sb_[0] >= sa_[0] { (sa_[1], sb_[0]) } else { (sb_[1], sa_[0]) };
        let gap = gap1 - gap0;
        if gap > max_gap_mm {
            return None;
        }
        if gap > 2.5 * prm.step_mm {
            let a_max = 80.0f64.min(1.2 * 3.0f64.max(a.flank_max.min(b.flank_max)));
            let (ok_a, ok_b) = same_plates(
                ctx,
                &pa0,
                &d,
                &a.da,
                &a.db,
                gap0,
                gap1,
                t,
                a_max,
                (a.flank_a.min(b.flank_a), a.flank_b.min(b.flank_b)),
            );
            if !(ok_a && ok_b) {
                return None;
            }
        }
        let merged = concat(&a_item.seg, &b_item.seg);
        let ln = segment_line(&merged, u, prm)?;
        Some((merged, ln))
    };
    // verdict cache: 0 unknown, 1 no merge, 2 merge (result kept in `merged`); speculative batches of unknown
    // pairs are evaluated in parallel in scan order, and the first merging pair in the frozen scan order wins,
    // so the result equals the sequential loop (pair verdicts are pure functions of the two items)
    let cap = 2 * items.len() + 2;
    let dense = cap <= 6000;
    let mut vd: Vec<u8> = if dense { vec![0u8; cap * cap] } else { Vec::new() };
    let mut vh: HashMap<(usize, usize), u8> = HashMap::new();
    let mut merged: HashMap<(usize, usize), (Seg, Line)> = HashMap::new();
    let batch = if rayon::current_num_threads() > 1 { 32usize } else { 1 };
    let mut changed = true;
    while changed {
        changed = false;
        // items.sort(key=-length), stable
        items.sort_by(|x, y| total_cmp(&(-x.ln.length), &(-y.ln.length)));
        let n = items.len();
        let get = |vd: &Vec<u8>, vh: &HashMap<(usize, usize), u8>, k: (usize, usize)| -> u8 {
            if dense { vd[k.0 * cap + k.1] } else { *vh.get(&k).unwrap_or(&0) }
        };
        let mut hit: Option<(usize, usize)> = None;
        let mut i = 0usize;
        let mut j = 1usize;
        'scan: while i < n {
            // collect the next unknown pairs in scan order, stopping at a known merge
            let mut pend: Vec<(usize, usize)> = Vec::new();
            let mut stop_known: Option<(usize, usize)> = None;
            let (mut ci, mut cj) = (i, j);
            while ci < n && pend.len() < batch {
                if cj >= n {
                    ci += 1;
                    cj = ci + 1;
                    continue;
                }
                let key = (items[ci].uid, items[cj].uid);
                match get(&vd, &vh, key) {
                    2 => {
                        stop_known = Some((ci, cj));
                        break;
                    }
                    0 => pend.push((ci, cj)),
                    _ => {}
                }
                cj += 1;
            }
            let res: Vec<Option<(Seg, Line)>> = if pend.len() > 1 {
                pend.par_iter().map(|&(a, b)| try_pair(&items[a], &items[b])).collect()
            } else {
                pend.iter().map(|&(a, b)| try_pair(&items[a], &items[b])).collect()
            };
            let mut first_true: Option<(usize, usize)> = None;
            for (&(a, b), r) in pend.iter().zip(res) {
                let key = (items[a].uid, items[b].uid);
                let v = if r.is_some() { 2u8 } else { 1u8 };
                if dense {
                    vd[key.0 * cap + key.1] = v;
                } else {
                    vh.insert(key, v);
                }
                if let Some(m) = r {
                    merged.insert(key, m);
                    if first_true.is_none() {
                        first_true = Some((a, b));
                    }
                }
            }
            if let Some(h) = first_true.or(stop_known) {
                hit = Some(h);
                break 'scan;
            }
            if ci >= n {
                break;
            }
            i = ci;
            j = cj;
        }
        if let Some((i, j)) = hit {
            let key = (items[i].uid, items[j].uid);
            let (seg, ln) = merged.remove(&key).unwrap();
            items[i] = Item { seg, ln, uid: next_uid };
            next_uid += 1;
            items.remove(j);
            changed = true;
        }
    }
    items.into_iter().map(|it| it.ln).collect()
}

/// strip_evidence
fn strip_evidence(ctx: &Ctx, line: &Line, t: f64) -> (f64, f64) {
    let a = line.start;
    let b = line.end;
    let d0 = sub3(&b, &a);
    let l = norm3(&d0);
    if l < 1e-6 {
        return (0.0, 0.0);
    }
    let d = div3(&d0, l);
    let na0 = cross3(&d, &line.da);
    let nb0 = cross3(&d, &line.db);
    let na = div3(&na0, norm3(&na0).max(1e-9));
    let nb = div3(&nb0, norm3(&nb0).max(1e-9));
    let r = 2.5 * t + 3.0;
    let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
    let idx = ctx.tree.ball(&mid, 0.5 * l + r);
    if idx.is_empty() {
        return (0.0, 0.0);
    }
    let mut ss = Vec::new();
    let mut nn = Vec::new();
    for &i in &idx {
        let p = sub3(&ctx.mesh.v[i], &a);
        let s = dot3(&p, &d);
        let w = [p[0] - s * d[0], p[1] - s * d[1], p[2] - s * d[2]];
        let dist = norm3(&w);
        if s >= 0.0 && s <= l && dist <= r {
            ss.push(s);
            nn.push(ctx.mesh.vn[i]);
        }
    }
    if ss.len() < 5 {
        return (0.0, 0.0);
    }
    let x = cross3(&d, &na);
    let tot = dot3(&nb, &x).datan2(dot3(&nb, &na)).to_degrees().abs();
    let nbn = neg3(&nb);
    let tot2 = dot3(&nbn, &x).datan2(dot3(&nbn, &na)).to_degrees().abs();
    let lo_m = 0.15;
    let nst = 1usize.max(l.ceil() as usize);
    let mut strip = vec![false; nst];
    let mut on_a = vec![false; nst];
    let mut on_b = vec![false; nst];
    for k in 0..ss.len() {
        let n = &nn[k];
        let nd = dot3(n, &d);
        let np = [n[0] - nd * d[0], n[1] - nd * d[1], n[2] - nd * d[2]];
        let okp = norm3(&np) > 0.94;
        let nx = dot3(n, &x);
        let nna = dot3(n, &na);
        let a1 = nx.datan2(nna).to_degrees().abs();
        let a2 = nx.datan2(-nna).to_degrees().abs();
        let mid1 = tot > 8.0 && okp && a1 > lo_m * tot && a1 < (1.0 - lo_m) * tot;
        let mid2 = tot2 > 8.0 && okp && a2 > lo_m * tot2 && a2 < (1.0 - lo_m) * tot2;
        let bin = ((ss[k] as i64).max(0) as usize).min(nst - 1);
        if mid1 || mid2 {
            strip[bin] = true;
        }
        if okp && nna.abs() > 0.985 {
            on_a[bin] = true;
        }
        if okp && dot3(n, &nb).abs() > 0.985 {
            on_b[bin] = true;
        }
    }
    let fl: Vec<bool> = on_a.iter().zip(&on_b).map(|(a, b)| *a && *b).collect();
    (mean_bool(&strip), mean_bool(&fl))
}

/// refine_span
fn refine_span(ctx: &Ctx, line: &mut Line, step: f64, t: f64) {
    let a = line.start;
    let b = line.end;
    let l = norm3(&sub3(&b, &a));
    if l < 1e-6 {
        return;
    }
    let d = div3(&sub3(&b, &a), l);
    let r = 0.5 + 0.15 * t;
    let mut new = Vec::new();
    for (end, sgn) in [(a, -1.0f64), (b, 1.0f64)] {
        let ss = arange(-(step.min(0.4 * l)), 1.5 * step + 1e-9, 0.25);
        let occ: Vec<bool> = ss
            .iter()
            .map(|&s| {
                let f = sgn * s;
                let p = [end[0] + f * d[0], end[1] + f * d[1], end[2] + f * d[2]];
                ctx.tree.any_within(&p, r)
            })
            .collect();
        let k0 = searchsorted_left(&ss, 0.0);
        let mut last: Option<usize> = None;
        let mut miss = 0;
        for k in 0..ss.len() {
            if occ[k] {
                last = Some(k);
                miss = 0;
            } else {
                miss += 1;
                if k > k0 && miss > 3 {
                    break;
                }
            }
        }
        let shift = match last {
            Some(k) => ss[k],
            None => 0.0,
        };
        let f = sgn * shift;
        new.push([end[0] + f * d[0], end[1] + f * d[1], end[2] + f * d[2]]);
    }
    let dv = sub3(&new[1], &new[0]);
    if norm3(&dv) > 1.0 && dot3(&dv, &d) > 0.0 {
        line.start = new[0];
        line.end = new[1];
        line.length = norm3(&dv);
    }
}

/// _pair
pub fn pair(a0: &V3, a1: &V3, b0: &V3, b1: &V3) -> (f64, f64, f64) {
    let da0 = sub3(a1, a0);
    let db0 = sub3(b1, b0);
    let la = norm3(&da0);
    let lb = norm3(&db0);
    let da = div3(&da0, la);
    let db = div3(&db0, lb);
    let ang = 1.0f64.min(dot3(&da, &db).abs()).dacos().to_degrees();
    let ma = [0.5 * (a0[0] + a1[0]), 0.5 * (a0[1] + a1[1]), 0.5 * (a0[2] + a1[2])];
    let mb = [0.5 * (b0[0] + b1[0]), 0.5 * (b0[1] + b1[1]), 0.5 * (b0[2] + b1[2])];
    let dm = sub3(&mb, &ma);
    let pa = dot3(&dm, &da);
    let pb = dot3(&dm, &db);
    let oa = norm3(&[dm[0] - da[0] * pa, dm[1] - da[1] * pa, dm[2] - da[2] * pa]);
    let ob = norm3(&[dm[0] - db[0] * pb, dm[1] - db[1] * pb, dm[2] - db[2] * pb]);
    let off = if oa >= ob { oa } else { ob };
    let srt = |x: f64, y: f64| if x <= y { (x, y) } else { (y, x) };
    let ia = srt(dot3(&sub3(a0, &ma), &da), dot3(&sub3(a1, &ma), &da));
    let ib = srt(dot3(&sub3(b0, &ma), &da), dot3(&sub3(b1, &ma), &da));
    let ov = 0.0f64.max(ia.1.min(ib.1) - ia.0.max(ib.0));
    (ang, off, ov / la.min(lb).max(1e-9))
}

fn merge_candidates(cands: Vec<Line>) -> Vec<Line> {
    let mut c = cands;
    c.sort_by(|x, y| total_cmp(&(-x.quality), &(-y.quality)));
    let mut kept: Vec<Line> = Vec::new();
    for x in c {
        let mut dup = false;
        for k in kept.iter_mut() {
            let (ang, off, ov) = pair(&x.start, &x.end, &k.start, &k.end);
            if ang < 4.0 && off < 2.5 && ov > 0.5 {
                dup = true;
                k.n_dups += 1;
                break;
            }
        }
        if !dup {
            kept.push(x);
        }
    }
    kept
}

pub fn line_json(l: &Line) -> Value {
    let mut m = Map::new();
    m.insert("family".into(), json!(l.family));
    m.insert("axis".into(), json!(l.axis.to_vec()));
    m.insert("start".into(), json!(l.start.to_vec()));
    m.insert("end".into(), json!(l.end.to_vec()));
    m.insert("dir".into(), json!(l.dir.to_vec()));
    m.insert("length".into(), json!(l.length));
    m.insert("n_support".into(), json!(l.n_support));
    m.insert("n_slices".into(), json!(l.n_slices));
    m.insert("support_frac".into(), json!(l.support_frac));
    m.insert("incl".into(), json!(l.incl));
    m.insert("incl_std".into(), json!(l.incl_std));
    m.insert("flank_min".into(), json!(l.flank_min));
    m.insert("flank_max".into(), json!(l.flank_max));
    m.insert("flank_a".into(), json!(l.flank_a));
    m.insert("flank_b".into(), json!(l.flank_b));
    m.insert("res_rms".into(), json!(l.res_rms));
    m.insert("axis_dev_deg".into(), json!(l.axis_dev_deg));
    m.insert("frac_outer".into(), json!(l.frac_outer));
    m.insert("frac_inner".into(), json!(l.frac_inner));
    m.insert("frac_both".into(), json!(l.frac_both));
    m.insert("flat_rms".into(), json!(l.flat_rms));
    m.insert("gap_med".into(), json!(l.gap_med));
    m.insert("bis".into(), json!(l.bis.to_vec()));
    m.insert("da".into(), json!(l.da.to_vec()));
    m.insert("db".into(), json!(l.db.to_vec()));
    m.insert("n_parts".into(), json!(l.n_parts));
    m.insert("strip_frac".into(), json!(l.strip_frac));
    m.insert("flank_frac".into(), json!(l.flank_frac));
    m.insert("quality".into(), json!(l.quality));
    if l.n_dups > 0 {
        m.insert("n_dups".into(), json!(l.n_dups));
    }
    Value::Object(m)
}

/// Debug record per family: counts per stage, the corner list per level and the pre-merge track lines.
pub struct FamDebug {
    pub n_levels: usize,
    pub n_chains: usize,
    pub n_corners: usize,
    pub n_tracks: usize,
    pub n_segments: usize,
    pub n_lines: usize,
    pub corners: Vec<(usize, f64, f64, f64)>, // level, apex_o x, apex_o y, incl
    pub seg_lines: Vec<Line>,
}

/// detect, from the family axes on. Returns (merged candidates, raw count, per-family debug, timings).
pub fn detect(
    mesh: &Mesh,
    tree: &KdTree,
    axes: &[V3],
    prm: &Params,
    t: f64,
    want_debug: bool,
) -> (Vec<Line>, usize, Vec<FamDebug>, Map<String, Value>) {
    let ctx = Ctx { mesh, tree };
    let t0 = std::time::Instant::now();
    let fam_out: Vec<(Vec<Line>, FamDebug)> = axes
        .par_iter()
        .enumerate()
        .map(|(fi, u)| {
            let (levels, chains) = crate::prof::time("z3.detect.slice", || slice_family(mesh, u, prm.step_mm));
            let per_level: Vec<Vec<Corner>> = crate::prof::time("z3.detect.corners", || chains.par_iter().map(|ch| slice_corners(ch, prm, t)).collect());
            let tracks = crate::prof::time("z3.detect.track", || track_corners(&per_level, prm));
            let segs = tracks_to_segments(&tracks, &levels, u, prm, fi);
            let mut dbg = FamDebug {
                n_levels: levels.len(),
                n_chains: chains.iter().map(|c| c.len()).sum(),
                n_corners: per_level.iter().map(|c| c.len()).sum(),
                n_tracks: tracks.len(),
                n_segments: segs.len(),
                n_lines: 0,
                corners: vec![],
                seg_lines: vec![],
            };
            if want_debug {
                for (k, cs) in per_level.iter().enumerate() {
                    for c in cs {
                        dbg.corners.push((k, c.apex_o[0], c.apex_o[1], c.incl));
                    }
                }
                dbg.seg_lines = segs.iter().filter_map(|s| segment_line(s, u, prm)).collect();
            }
            let lines = crate::prof::time("z3.detect.merge", || merge_segments(segs, u, prm, &ctx, t, 80.0));
            dbg.n_lines = lines.len();
            (lines, dbg)
        })
        .collect();
    let t_slice = t0.elapsed().as_secs_f64();
    let mut raw: Vec<Line> = Vec::new();
    let mut dbgs = Vec::new();
    for (l, d) in fam_out {
        raw.extend(l);
        dbgs.push(d);
    }
    let t1 = std::time::Instant::now();
    let _strip_t = crate::prof::Timer::new("z3.detect.refine_strip");
    raw.par_iter_mut().for_each(|c| {
        refine_span(&ctx, c, prm.step_mm, t);
        let (sf, ff) = strip_evidence(&ctx, c, t);
        c.strip_frac = sf;
        c.flank_frac = ff;
        c.quality = (c.n_support as f64) * c.support_frac * (0.5 + c.frac_outer + c.frac_both);
    });
    let t_strip = t1.elapsed().as_secs_f64();
    let n_raw = raw.len();
    let merged = merge_candidates(raw);
    let mut tm = Map::new();
    tm.insert("slicing".into(), json!(t_slice));
    tm.insert("strip".into(), json!(t_strip));
    (merged, n_raw, dbgs, tm)
}
