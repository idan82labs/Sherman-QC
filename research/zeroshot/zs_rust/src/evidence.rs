//! Port of zs_lane_v2/evidence.py (station_profile, sweep_features, plate_connected, collinear_pairs,
//! pair_evidence) and of the per-candidate / per-pair loop of zs_lane_v2.core.evidence_stage (without the
//! split runs, which the frozen v3 configuration does not use).
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::kdtree::KdTree;
use crate::morph;
use crate::num::*;
use rayon::prelude::*;
use serde_json::{json, Map, Value};

pub struct Ctx<'a> {
    pub v: &'a [V3],
    pub vn: &'a [V3],
    pub tree: &'a KdTree,
    pub t: f64,
    pub spacing: f64,
    pub cell: f64,
}

impl<'a> Ctx<'a> {
    pub fn new(v: &'a [V3], vn: &'a [V3], tree: &'a KdTree, t: f64, spacing: f64) -> Ctx<'a> {
        let cell = clip(1.25 * spacing, 0.4, 1.5);
        Ctx { v, vn, tree, t, spacing, cell }
    }
}

fn unit(v: &V3) -> V3 {
    let n = norm3(v);
    if n > 1e-12 {
        div3(v, n)
    } else {
        *v
    }
}

#[derive(Clone)]
pub struct Cand {
    start: V3,
    end: V3,
    da: Option<V3>,
    db: Option<V3>,
    flank_a: f64,
    flank_b: f64,
    incl: Option<f64>,
}

fn opt_v3(v: &Value) -> Option<V3> {
    let a = v.as_array()?;
    Some([a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?])
}

fn parse_cands(v: &Value) -> Vec<Cand> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| Cand {
            start: opt_v3(&c["start"]).unwrap(),
            end: opt_v3(&c["end"]).unwrap(),
            da: opt_v3(&c["da"]),
            db: opt_v3(&c["db"]),
            flank_a: c["flank_a"].as_f64().unwrap_or(f64::NAN),
            flank_b: c["flank_b"].as_f64().unwrap_or(f64::NAN),
            incl: c["incl"].as_f64(),
        })
        .collect()
}

/// _line
fn line(a: &V3, b: &V3) -> (V3, V3, f64) {
    let l = norm3(&sub3(b, a));
    let d = div3(&sub3(b, a), l.max(1e-9));
    (*a, d, l)
}

fn proj_unit(f: &V3, d: &V3) -> V3 {
    let x = unit(f);
    let s = dot3(&x, d);
    unit(&[x[0] - s * d[0], x[1] - s * d[1], x[2] - s * d[2]])
}

pub struct Prof {
    bend: Vec<bool>,
    fa: Vec<bool>,
    fb: Vec<bool>,
    sweep: Vec<bool>,
    sweep_bins: Vec<f64>,
}

/// station_profile(ctx, {start, end, da, db}, s_lo, s_hi)
fn station_profile(ctx: &Ctx, start: &V3, end: &V3, da0: &V3, db0: &V3, s_lo: Option<f64>, s_hi: Option<f64>) -> Prof {
    let t = ctx.t;
    let (p0, d, l) = line(start, end);
    let s_lo = s_lo.unwrap_or(0.0);
    let s_hi = s_hi.unwrap_or(l);
    let nst = 1i64.max((s_hi - s_lo).ceil() as i64) as usize;
    let da = proj_unit(da0, &d);
    let db = proj_unit(db0, &d);
    let a_lo = 2.0 * t + 2.0;
    let a_w = a_lo + 4.0f64.max(2.0 * t);
    let r_zone = 2.5 * t + 3.0;
    let reach = a_w.max(r_zone) + t + 2.0;
    let hm = 0.5 * (s_lo + s_hi);
    let mid = [p0[0] + hm * d[0], p0[1] + hm * d[1], p0[2] + hm * d[2]];
    let rad = (sq(0.5 * (s_hi - s_lo)) + sq(reach)).sqrt() + 1.0;
    let idx = ctx.tree.ball(&mid, rad);
    let mut res = Prof {
        bend: vec![false; nst],
        fa: vec![false; nst],
        fb: vec![false; nst],
        sweep: vec![false; nst],
        sweep_bins: vec![0.0; nst],
    };
    if idx.is_empty() {
        return res;
    }
    let r_touch = 0.6 + 0.2 * t + 1.25 * ctx.spacing;
    let nrm_a = unit(&cross3(&d, &da));
    let nrm_b = unit(&cross3(&d, &db));
    let mut na_ = unit(&cross3(&d, &da));
    let mut nb_ = unit(&cross3(&d, &db));
    let dsum = add3(&da, &db);
    if dot3(&na_, &dsum) > 0.0 {
        na_ = neg3(&na_);
    }
    if dot3(&nb_, &dsum) > 0.0 {
        nb_ = neg3(&nb_);
    }
    let theta = clip(dot3(&na_, &nb_), -1.0, 1.0).dacos().to_degrees();
    let pr = dot3(&nb_, &na_);
    let x = unit(&[nb_[0] - pr * na_[0], nb_[1] - pr * na_[1], nb_[2] - pr * na_[2]]);
    let mut bins4 = vec![[false; 4]; nst];
    let mut any_sel = false;
    for &i in &idx {
        let q = sub3(&ctx.v[i], &p0);
        let s = dot3(&q, &d);
        if !(s >= s_lo && s < s_lo + nst as f64) {
            continue;
        }
        let k = (((s - s_lo) as i64).max(0) as usize).min(nst - 1);
        let w = [q[0] - s * d[0], q[1] - s * d[1], q[2] - s * d[2]];
        let dist = norm3(&w);
        if dist <= r_touch {
            res.bend[k] = true;
        }
        let nv = &ctx.vn[i];
        for (fv, nrm, out) in [(&da, &nrm_a, 0), (&db, &nrm_b, 1)] {
            let a = dot3(&q, fv);
            let h = dot3(&q, nrm).abs();
            if a >= a_lo && a <= a_w && h <= t + 1.5 && dot3(nv, nrm).abs() > 0.9 {
                if out == 0 {
                    res.fa[k] = true;
                } else {
                    res.fb[k] = true;
                }
            }
        }
        let zone = dist <= r_zone;
        let nd = dot3(nv, &d);
        let np = [nv[0] - nd * d[0], nv[1] - nd * d[1], nv[2] - nd * d[2]];
        let okp = norm3(&np) > 0.9;
        if !(zone && okp) {
            continue;
        }
        any_sel = true;
        if theta >= 8.0 {
            let nx = dot3(nv, &x);
            let nna = dot3(nv, &na_);
            let ang = nx.datan2(nna).to_degrees();
            let ang_in = (-nx).datan2(-nna).to_degrees();
            for a in [ang, ang_in] {
                let fr = a / theta;
                if fr > 0.1 && fr < 0.9 {
                    let b = ((((fr - 0.1) / 0.2) as i64).max(0)).min(3) as usize;
                    bins4[k][b] = true;
                }
            }
        }
    }
    if theta >= 8.0 && any_sel {
        for k in 0..nst {
            let c = bins4[k].iter().filter(|&&b| b).count() as f64;
            res.sweep_bins[k] = c;
            res.sweep[k] = c >= 3.0;
        }
    }
    res
}

/// sweep_features -> [sw_full, sw_bins, st_bend, st_fa, st_fb, st_both]
fn sweep_feats(p: &Prof) -> [f64; 6] {
    let n = p.bend.len() as f64;
    let both: Vec<bool> = p.fa.iter().zip(&p.fb).map(|(a, b)| *a && *b).collect();
    [
        mean_bool(&p.sweep),
        (pw_sum(&p.sweep_bins) / n) / 4.0,
        mean_bool(&p.bend),
        mean_bool(&p.fa),
        mean_bool(&p.fb),
        mean_bool(&both),
    ]
}

/// flank_band
fn flank_band(t: f64, flank_len: Option<f64>, a_hi: f64) -> (f64, f64) {
    let a_lo = 2.0 * t + 2.0;
    if let Some(fl) = flank_len {
        if fl == fl && fl < a_lo + 4.0 {
            let lo = 0.5f64.max(0.4 * fl);
            return (lo, (lo + 2.0).max(1.1 * fl));
        }
    }
    (a_lo, (a_lo + 4.0).max(a_hi))
}

struct Plate {
    connected: bool,
    gap_cover: f64,
    left: usize,
    right: usize,
    gap_cover_raw: Option<f64>,
}

/// plate_connected (idx None: the index set is gathered here, after the flank band, as in Python)
fn plate_connected(ctx: &Ctx, p0: &V3, d: &V3, f: &V3, g0: f64, g1: f64, a_hi: f64, pad: f64, idx_in: Option<&[usize]>,
                   flank_len: Option<f64>) -> Plate {
    let t = ctx.t;
    let (a_lo, a_hi) = flank_band(t, flank_len, a_hi);
    let mut out = Plate { connected: false, gap_cover: 0.0, left: 0, right: 0, gap_cover_raw: None };
    let own;
    let idx: &[usize] = match idx_in {
        Some(x) => x,
        None => {
            let hm = 0.5 * (g0 + g1);
            let mid = [p0[0] + hm * d[0], p0[1] + hm * d[1], p0[2] + hm * d[2]];
            let rad = (sq(0.5 * (g1 - g0) + pad) + sq(a_hi)).sqrt() + t + 2.0;
            own = ctx.tree.ball(&mid, rad);
            &own
        }
    };
    if idx.is_empty() {
        return out;
    }
    let nrm = unit(&cross3(d, f));
    let mut on_s = Vec::new();
    let mut on_a = Vec::new();
    for &i in idx {
        let q = sub3(&ctx.v[i], p0);
        let s = dot3(&q, d);
        let a = dot3(&q, f);
        let h = dot3(&q, &nrm).abs();
        if s >= g0 - pad && s <= g1 + pad && a >= a_lo && a <= a_hi && h <= t + 1.5 + 0.03 * a
            && dot3(&ctx.vn[i], &nrm).abs() > 0.9
        {
            on_s.push(s);
            on_a.push(a);
        }
    }
    if on_s.len() < 4 {
        return out;
    }
    let cell = ctx.cell;
    let ns = ((g1 - g0 + 2.0 * pad) / cell).ceil() as i64 + 1;
    let na = ((a_hi - a_lo) / cell).ceil() as i64 + 1;
    let (nsu, nau) = (ns as usize, na as usize);
    let mut grid = vec![false; nsu * nau];
    let mut raw = vec![false; nsu];
    for k in 0..on_s.len() {
        let gi = ((((on_s[k] - (g0 - pad)) / cell) as i64).max(0)).min(ns - 1) as usize;
        let gj = ((((on_a[k] - a_lo) / cell) as i64).max(0)).min(na - 1) as usize;
        grid[gi * nau + gj] = true;
        raw[gi] = true;
    }
    let grid = morph::dilate_cross(&grid, nsu, nau);
    let lab = morph::label8(&grid, nsu, nau);
    let k0 = 1i64.max(round_even(pad.min(4.0) / cell) as i64) as usize;
    let il = ((pad - pad.min(4.0)) / cell) as i64 as usize;
    let ir = (((g1 - g0 + pad) / cell) as i64 + 1) as usize;
    let rows = |a: usize, b: usize| (a.min(nsu), b.min(nsu));
    let (l0, l1) = rows(il, il + k0);
    let (r0, r1) = rows(ir, ir + k0);
    let left = morph::labels_in_rows(&lab, nau, l0, l1.max(l0));
    let right = morph::labels_in_rows(&lab, nau, r0, r1.max(r0));
    let gs = (pad / cell) as i64;
    let ge = ((g1 - g0 + pad) / cell) as i64;
    let cov = if ge >= gs {
        let (a, b) = rows(gs as usize, (ge + 1) as usize);
        let n = b.saturating_sub(a);
        if n == 0 {
            f64::NAN
        } else {
            let c = (a..b).filter(|&i| (0..nau).any(|j| grid[i * nau + j])).count();
            c as f64 / n as f64
        }
    } else {
        1.0
    };
    let gsr = (pad / cell).ceil() as i64;
    let ger = ((g1 - g0 + pad) / cell) as i64 - 1;
    let cov_raw = if ger >= gsr {
        let (a, b) = rows(gsr as usize, (ger + 1) as usize);
        let n = b.saturating_sub(a);
        if n == 0 { f64::NAN } else { (a..b).filter(|&i| raw[i]).count() as f64 / n as f64 }
    } else {
        let k = (gs.max(0) as usize).min(nsu - 1);
        if raw[k] { 1.0 } else { 0.0 }
    };
    out.connected = left.intersection(&right).next().is_some();
    out.gap_cover = cov;
    out.left = left.len();
    out.right = right.len();
    out.gap_cover_raw = Some(cov_raw);
    out
}

/// bend_touch: per `step` station, does scanned surface touch the apex line (radius 0.35 + 0.5 spacing)?
fn bend_touch(ctx: &Ctx, p0: &V3, d: &V3, s_lo: f64, s_hi: f64, step: f64) -> Vec<bool> {
    let nst = 1i64.max(((s_hi - s_lo) / step).ceil() as i64) as usize;
    let r = 0.35 + 0.5 * ctx.spacing;
    (0..nst)
        .map(|k| {
            let s = s_lo + step * (k as f64 + 0.5);
            let p = [p0[0] + s * d[0], p0[1] + s * d[1], p0[2] + s * d[2]];
            ctx.tree.any_within(&p, r)
        })
        .collect()
}

/// split_runs(ctx, c, prof): interior runs where the apex line leaves the scanned surface, with the plate
/// connectivity of both flanks across each run.
fn split_runs(ctx: &Ctx, c: &Cand) -> Vec<Value> {
    let min_run = 1.0;
    let (p0, d, l) = line(&c.start, &c.end);
    let step = 0.5;
    let touch = bend_touch(ctx, &p0, &d, 0.0, l, step);
    let n = touch.len();
    let da = proj_unit(&c.da.unwrap(), &d);
    let db = proj_unit(&c.db.unwrap(), &d);
    let fl = [c.flank_a, c.flank_b];
    let fmax: Vec<f64> = fl.iter().cloned().filter(|x| x == x).collect();
    let a_hi = if fmax.is_empty() {
        clip(60.0, 12.0, 150.0)
    } else {
        let mx = fmax.iter().cloned().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        clip(1.25 * mx + 3.0, 12.0, 150.0)
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if !touch[i] {
            let mut j = i;
            while j < n && !touch[j] {
                j += 1;
            }
            // run (i, j) of ~touch
            if !(i == 0 || j >= n || ((j - i) as f64) * step < min_run) {
                let g0 = i as f64 * step;
                let g1 = j as f64 * step;
                let pad = 2.0f64.max(8.0f64.min(g0).min(l - g1));
                let fla = if fl[0] == fl[0] { Some(fl[0]) } else { None };
                let flb = if fl[1] == fl[1] { Some(fl[1]) } else { None };
                let ca = plate_connected(ctx, &p0, &d, &da, g0, g1, a_hi, pad, None, fla);
                let cb = plate_connected(ctx, &p0, &d, &db, g0, g1, a_hi, pad, None, flb);
                out.push(json!({"g0": g0, "g1": g1, "ca": ca.connected, "cb": cb.connected,
                                "cov_a": fval(ca.gap_cover), "cov_b": fval(cb.gap_cover), "a_hi": a_hi,
                                "raw_a": fval(ca.gap_cover_raw.unwrap_or(1.0)),
                                "raw_b": fval(cb.gap_cover_raw.unwrap_or(1.0)),
                                "sides_a": (ca.left > 0) as i64 + (ca.right > 0) as i64,
                                "sides_b": (cb.left > 0) as i64 + (cb.right > 0) as i64}));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    out
}

/// collinear_pairs(C, t)
fn collinear_pairs(c: &[Cand], t: f64) -> Vec<(usize, usize, f64, f64, f64)> {
    let max_gap = 250.0;
    let ca = 3.0f64.to_radians().dcos();
    let lines: Vec<(V3, V3, f64)> = c.iter().map(|x| line(&x.start, &x.end)).collect();
    let n = c.len();
    let mut best: Vec<Option<(f64, usize)>> = vec![None; n];
    for i in 0..n {
        let (p0, d, l) = lines[i];
        if l < 1.0 {
            continue;
        }
        for j in 0..n {
            if j == i {
                continue;
            }
            let (q0, e, m) = lines[j];
            if m < 1.0 || dot3(&d, &e).abs() < ca {
                continue;
            }
            if let (Some(ia), Some(ib)) = (c[i].incl, c[j].incl) {
                if (ia - ib).abs() > 10.0 {
                    continue;
                }
            }
            let q1 = [q0[0] + m * e[0], q0[1] + m * e[1], q0[2] + m * e[2]];
            let perp = |q: &V3| {
                let r = sub3(q, &p0);
                let s = dot3(&r, &d);
                norm3(&[r[0] - s * d[0], r[1] - s * d[1], r[2] - s * d[2]])
            };
            let o0 = perp(&q0);
            let o1 = perp(&q1);
            let off = if o0 >= o1 { o0 } else { o1 };
            if off > 1.5 + 0.4 * t {
                continue;
            }
            let x0 = dot3(&sub3(&q0, &p0), &d);
            let x1 = dot3(&sub3(&q1, &p0), &d);
            let sb0 = x0.min(x1);
            if sb0 < l - 0.2 * l.min(m) {
                continue;
            }
            let (g0, g1) = (l, sb0);
            if g1 - g0 > max_gap {
                continue;
            }
            match best[i] {
                Some((g, _)) if !(g1 - g0 < g) => {}
                _ => best[i] = Some((g1 - g0, j)),
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for i in 0..n {
        if let Some((_, j)) = best[i] {
            let (p0, d, l) = lines[i];
            let (q0, e, m) = lines[j];
            let q1 = [q0[0] + m * e[0], q0[1] + m * e[1], q0[2] + m * e[2]];
            let x0 = dot3(&sub3(&q0, &p0), &d);
            let x1 = dot3(&sub3(&q1, &p0), &d);
            let sb0 = x0.min(x1);
            let key = (i.min(j), i.max(j));
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);
            out.push((i, j, l, sb0, 0.0f64.max(l - sb0)));
        }
    }
    out
}

fn fval(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

/// pair_evidence(ctx, ci, cj, g0, g1) -> dict (without the i/j/g0/g1/overlap fields)
fn pair_evidence(ctx: &Ctx, ci: &Cand, cj: &Cand, g0: f64, g1: f64) -> Map<String, Value> {
    let (p0, d, l) = line(&ci.start, &ci.end);
    let (_q0, _e, m) = line(&cj.start, &cj.end);
    let t = ctx.t;
    let mut out = Map::new();
    out.insert("gap".into(), json!(g1 - g0));
    out.insert("len_i".into(), json!(l));
    out.insert("len_j".into(), json!(m));
    let fl = [(ci.flank_a, ci.flank_b), (cj.flank_a, cj.flank_b)];
    out.insert("flank_i".into(), json!([fval(fl[0].0), fval(fl[0].1)]));
    out.insert("flank_j".into(), json!([fval(fl[1].0), fval(fl[1].1)]));
    let (da0, db0) = match (ci.da, ci.db) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            out.insert("has_flanks".into(), json!(false));
            return out;
        }
    };
    out.insert("has_flanks".into(), json!(true));
    let da = proj_unit(&da0, &d);
    let db = proj_unit(&db0, &d);
    let fmax: Vec<f64> = [fl[0].0, fl[0].1, fl[1].0, fl[1].1].iter().cloned().filter(|x| x == x).collect();
    let a_hi = if fmax.is_empty() {
        clip(40.0, 12.0, 150.0)
    } else {
        let mx = fmax.iter().cloned().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        clip(1.25 * mx + 3.0, 12.0, 150.0)
    };
    let (g0c, g1c) = if g1 - g0 < 0.5 { (g0 - 0.25, g0 + 0.25) } else { (g0, g1) };
    let pad = 2.0f64.max(8.0f64.min(l).min(m));
    let hm = 0.5 * (g0c + g1c);
    let mid = [p0[0] + hm * d[0], p0[1] + hm * d[1], p0[2] + hm * d[2]];
    let rad = (sq(0.5 * (g1c - g0c) + pad) + sq(a_hi)).sqrt() + t + 2.0;
    let idx = ctx.tree.ball(&mid, rad);
    let maxo = |v: Vec<f64>| -> Option<f64> {
        if v.is_empty() {
            None
        } else {
            Some(v.iter().cloned().fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a }))
        }
    };
    let fla = maxo([fl[0].0, fl[1].0].iter().cloned().filter(|x| x == x).collect());
    let flb = maxo([fl[0].1, fl[1].1].iter().cloned().filter(|x| x == x).collect());
    let ca = plate_connected(ctx, &p0, &d, &da, g0c, g1c, a_hi, pad, Some(&idx), fla);
    let cb = plate_connected(ctx, &p0, &d, &db, g0c, g1c, a_hi, pad, Some(&idx), flb);
    out.insert("ca".into(), json!(ca.connected));
    out.insert("cb".into(), json!(cb.connected));
    out.insert("cov_a".into(), fval(ca.gap_cover));
    out.insert("cov_b".into(), fval(cb.gap_cover));
    out.insert("a_hi".into(), json!(a_hi));
    if g1 - g0 >= 1.0 {
        let end = [p0[0] + l * d[0], p0[1] + l * d[1], p0[2] + l * d[2]];
        let pr = station_profile(ctx, &p0, &end, &da, &db, Some(g0), Some(g1));
        out.insert("gap_bend".into(), json!(mean_bool(&pr.bend)));
        out.insert("gap_fa".into(), json!(mean_bool(&pr.fa)));
        out.insert("gap_fb".into(), json!(mean_bool(&pr.fb)));
    } else {
        out.insert("gap_bend".into(), json!(1.0));
        out.insert("gap_fa".into(), json!(1.0));
        out.insert("gap_fb".into(), json!(1.0));
    }
    out
}

/// evidence_stage (sweep + pairs)
pub fn run(ctx: &Ctx, cands_v: &Value, want_splits: bool) -> Value {
    let c = parse_cands(cands_v);
    let splits: Vec<(usize, Vec<Value>)> = if want_splits {
        (0..c.len())
            .into_par_iter()
            .filter_map(|i| {
                let x = &c[i];
                let l = norm3(&sub3(&x.end, &x.start));
                if x.da.is_none() || x.db.is_none() || l < 1.0 || !(l >= 10.0) {
                    return None;
                }
                let r = split_runs(ctx, x);
                if r.is_empty() { None } else { Some((i, r)) }
            })
            .collect()
    } else {
        vec![]
    };
    let mut sp = Map::new();
    for (i, r) in splits {
        sp.insert(i.to_string(), Value::Array(r));
    }
    let sweep: Vec<Value> = c
        .par_iter()
        .map(|x| {
            let l = norm3(&sub3(&x.end, &x.start));
            match (x.da, x.db) {
                (Some(da), Some(db)) if !(l < 1.0) => {
                    let p = station_profile(ctx, &x.start, &x.end, &da, &db, None, None);
                    json!(sweep_feats(&p).to_vec())
                }
                _ => Value::Null,
            }
        })
        .collect();
    let pl = collinear_pairs(&c, ctx.t);
    let pairs: Vec<Value> = pl
        .par_iter()
        .map(|&(i, j, g0, g1, ov)| {
            let mut ev;
            if c[i].da.is_none() && c[j].da.is_some() {
                let q0 = c[j].start;
                let e0 = sub3(&c[j].end, &q0);
                let lj = norm3(&e0);
                let e = div3(&e0, lj.max(1e-9));
                let x0 = dot3(&sub3(&c[i].start, &q0), &e);
                let x1 = dot3(&sub3(&c[i].end, &q0), &e);
                let (s0, s1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
                let (mut h0, mut h1) = if s1 <= 0.5 * lj { (s1, 0.0) } else { (lj, s0) };
                if h1 < h0 {
                    h0 = h1;
                }
                ev = pair_evidence(ctx, &c[j], &c[i], h0, h1);
                let li = ev["len_i"].clone();
                let lj_ = ev["len_j"].clone();
                ev.insert("len_i".into(), lj_);
                ev.insert("len_j".into(), li);
                let fi = ev["flank_i"].clone();
                let fj = ev["flank_j"].clone();
                ev.insert("flank_i".into(), fj);
                ev.insert("flank_j".into(), fi);
                ev.insert("ref".into(), json!("j"));
            } else {
                ev = pair_evidence(ctx, &c[i], &c[j], g0, g1);
            }
            ev.insert("i".into(), json!(i));
            ev.insert("j".into(), json!(j));
            ev.insert("g0".into(), json!(g0));
            ev.insert("g1".into(), json!(g1));
            ev.insert("overlap".into(), json!(ov));
            Value::Object(ev)
        })
        .collect();
    json!({"sweep": sweep, "pairs": pairs, "splits": Value::Object(sp), "spacing": ctx.spacing, "cell": ctx.cell,
           "t_mm": ctx.t})
}
