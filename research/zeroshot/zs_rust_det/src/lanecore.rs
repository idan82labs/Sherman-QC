//! The zs_lane_v3 (revision 2) lane after detection: pooling (z3_slicing.pooled.pool_scan + selector.feats,
//! zs_lane_v2.core.pool), D1 snap of every pooled candidate (zs_lane_v2.core.snap), v2 evidence, the pooled GBM
//! and the Z3-alone GBM (gbm.rs), v2 span post-processing (zs_lane_v2/postproc.py) and the v3 fixes EXT / Z3R
//! (zs_lane_v3/fixes.py).
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use crate::d1;
use crate::kdtree::KdTree;
use crate::num::*;
use crate::z3::{self, Line};
use serde_json::{json, Map, Value};

#[derive(Clone, Debug)]
pub struct PCand {
    pub start: V3,
    pub end: V3,
    pub src: Vec<String>,
    pub has_z3: bool,
    pub da: Option<V3>,
    pub db: Option<V3>,
    pub flank_a: Option<f64>,
    pub flank_b: Option<f64>,
    pub length: Option<f64>,
    pub family: Option<usize>,
    pub n_support: Option<usize>,
    pub incl: f64,
    pub start_s: V3,
    pub end_s: V3,
    pub snap: Map<String, Value>,
}

pub struct Z2Line {
    pub start: V3,
    pub end: V3,
    pub incl: f64,
}

/// selector.feats(d): the 21 Z3 features per candidate
fn feats(c: &[Line]) -> Vec<[f64; 21]> {
    let n = c.len();
    let negq: Vec<f64> = c.iter().map(|x| -x.quality).collect();
    let order = np_argsort(&negq);
    let mut rank = vec![0usize; n];
    for (r, &i) in order.iter().enumerate() {
        rank[i] = r;
    }
    let lmax = c.iter().map(|x| x.length).fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    c.iter()
        .enumerate()
        .map(|(i, x)| {
            [
                (x.length + 1.0).dln(),
                ((x.n_support + 1) as f64).dln(),
                x.support_frac,
                180.0 - x.incl,
                x.incl_std,
                (x.flank_min + 0.5).dln(),
                (x.flank_max + 0.5).dln(),
                x.res_rms,
                x.axis_dev_deg,
                x.frac_outer,
                x.frac_inner,
                x.frac_both,
                x.flat_rms,
                if x.gap_med <= 30.0 { x.gap_med } else { 30.0 },
                x.strip_frac,
                x.flank_frac,
                (x.n_parts.min(6)) as f64,
                ((x.n_dups + 1) as f64).dln(),
                rank[i] as f64 / n as f64,
                (x.family.min(15)) as f64,
                x.length / lmax,
            ]
        })
        .collect()
}

/// pool_scan + zs_lane_v2.core.pool: pooled candidates and the 26-column feature matrix
pub fn pool(z3c: &[Line], z2: &[Z2Line]) -> (Vec<PCand>, Vec<[f64; 26]>) {
    let f21 = feats(z3c);
    let nz = z3c.len();
    let mut flags: Vec<[f64; 5]> = vec![[1.0, 0.0, 0.0, 0.0, 0.0]; nz];
    let mut src: Vec<Vec<&'static str>> = vec![vec!["z3"]; nz];
    struct Extra {
        start: V3,
        end: V3,
        incl: f64,
        src: Vec<&'static str>,
        row: [f64; 5],
    }
    let mut extra: Vec<Extra> = Vec::new();
    for ln in z2 {
        let mut best: Option<f64> = None;
        let mut bj: i64 = -1;
        for (j, c) in z3c.iter().enumerate() {
            let (ang, off, ov) = z3::pair(&ln.start, &ln.end, &c.start, &c.end);
            if ang < 4.0 && off < 2.5 && ov > 0.5 {
                let sc = ang / 4.0 + off / 2.5 - ov;
                if best.is_none() || sc < best.unwrap() {
                    best = Some(sc);
                    bj = j as i64;
                }
            }
        }
        if bj >= 0 {
            flags[bj as usize][1] = 1.0;
            if !src[bj as usize].contains(&"z2") {
                src[bj as usize].push("z2");
            }
            continue;
        }
        let mut dup = false;
        for e in extra.iter_mut() {
            let (ang, off, ov) = z3::pair(&ln.start, &ln.end, &e.start, &e.end);
            if std::env::var_os("ZS_PG_LOG").is_some() {
                let b = |v: &V3| format!("{:016x},{:016x},{:016x}", v[0].to_bits(), v[1].to_bits(), v[2].to_bits());
                eprintln!("PG {},{},{},{} -> {:016x},{:016x},{:016x}", b(&ln.start), b(&ln.end), b(&e.start), b(&e.end), ang.to_bits(), off.to_bits(), ov.to_bits());
            }
            if ang < 4.0 && off < 2.5 && ov > 0.5 {
                e.row[1] = 1.0;
                if !e.src.contains(&"z2") {
                    e.src.push("z2");
                }
                dup = true;
                break;
            }
        }
        if dup {
            continue;
        }
        let l = norm3(&sub3(&ln.end, &ln.start));
        let mut row = [0.0f64; 5];
        row[1] = 1.0;
        row[4] = (l + 1.0).dln();
        // the pooled incl of an extra candidate: the last Z2 line with the same rounded (start, end)
        extra.push(Extra { start: ln.start, end: ln.end, incl: 0.0, src: vec!["z2"], row });
    }
    for j in 0..nz {
        flags[j][3] = src[j].len() as f64;
        flags[j][4] = (z3c[j].length + 1.0).dln();
    }
    for e in extra.iter_mut() {
        e.row[3] = e.src.len() as f64;
        let key = (round6(&e.start), round6(&e.end));
        let mut incl = 90.0;
        for ln in z2 {
            if (round6(&ln.start), round6(&ln.end)) == key {
                incl = ln.incl;
            }
        }
        e.incl = incl;
    }
    let mut out = Vec::new();
    let mut x = Vec::new();
    for (j, c) in z3c.iter().enumerate() {
        let mut s: Vec<String> = src[j].iter().map(|s| s.to_string()).collect();
        s.sort();
        out.push(PCand {
            start: c.start,
            end: c.end,
            src: s,
            has_z3: true,
            da: Some(c.da),
            db: Some(c.db),
            flank_a: Some(c.flank_a),
            flank_b: Some(c.flank_b),
            length: Some(c.length),
            family: Some(c.family),
            n_support: Some(c.n_support),
            incl: c.incl,
            start_s: c.start,
            end_s: c.end,
            snap: Map::new(),
        });
        let mut row = [0.0f64; 26];
        row[..21].copy_from_slice(&f21[j]);
        row[21..].copy_from_slice(&flags[j]);
        x.push(row);
    }
    for e in &extra {
        let mut s: Vec<String> = e.src.iter().map(|s| s.to_string()).collect();
        s.sort();
        out.push(PCand {
            start: e.start,
            end: e.end,
            src: s,
            has_z3: false,
            da: None,
            db: None,
            flank_a: None,
            flank_b: None,
            length: None,
            family: None,
            n_support: None,
            incl: e.incl,
            start_s: e.start,
            end_s: e.end,
            snap: Map::new(),
        });
        let mut row = [f64::NAN; 26];
        row[21..].copy_from_slice(&e.row);
        x.push(row);
    }
    (out, x)
}

fn round6(v: &V3) -> [u64; 3] {
    // np.round(x, 6): numpy rounds via x * 1e6 -> rint -> / 1e6
    let r = |a: f64| ((a * 1e6).round_ties_even() / 1e6).to_bits();
    [r(v[0]), r(v[1]), r(v[2])]
}

/// zs_lane_v2.core.snap: D1 gated outer-apex snap of every pooled candidate
pub fn snap_all(c: &mut [PCand], p: &[V3], n: &[V3], t: f64, tree: &KdTree) {
    let profs: Vec<d1::Profile> = {
        use rayon::prelude::*;
        c.par_iter().map(|x| d1::bend_profile(p, n, &tree, &x.start, &x.end, t)).collect()
    };
    for (x, prof) in c.iter_mut().zip(profs) {
        x.start_s = x.start;
        x.end_s = x.end;
        let th = if x.incl != 0.0 { x.incl } else { 90.0 };
        let gate = t / (th.to_radians() / 2.0).dsin().max(1e-3) + 1.0;
        let mut info = Map::new();
        info.insert("moved".into(), json!(false));
        if !prof.ok {
            info.insert("reason".into(), json!(prof.reason));
            x.snap = info;
            continue;
        }
        let (target, used) = match prof.outer_apex_2d {
            Some(a) => (a, "outer_apex"),
            None => (prof.outer_sharp_2d.unwrap(), "outer_sharp"),
        };
        let shift = target[0].dhypot(target[1]);
        info.insert("locus".into(), json!(used));
        info.insert("shift_mm".into(), json!(crate::z2graph::py_round(shift, 4)));
        if shift > gate {
            info.insert("reason".into(), json!("SHIFT_TOO_LARGE"));
            x.snap = info;
            continue;
        }
        let d3 = d1::snap_offset(&prof, gate).unwrap();
        x.start_s = add3(&x.start, &d3);
        x.end_s = add3(&x.end, &d3);
        info.insert("moved".into(), json!(true));
        x.snap = info;
    }
}

// ------------------------------------------------------------------ v2 post-processing

pub struct PostParams {
    pub join: bool,
    pub join_low: f64,
    pub sweep_min: f64,
    pub dedupe_join: bool,
}

#[derive(Clone, Debug)]
pub struct OutLine {
    pub start_s: V3,
    pub end_s: V3,
    pub score: f64,
    pub src: Vec<String>,
    pub incl: f64,
    pub parts: Vec<usize>,
    pub op: &'static str,
    pub fix: Option<&'static str>,
    pub p_z3: Option<f64>,
    pub ext_mm: Option<[f64; 2]>,
}

fn sub_seg(c: &PCand, f0: f64, f1: f64) -> (V3, V3) {
    let (a, b) = (c.start_s, c.end_s);
    let d = sub3(&b, &a);
    ([a[0] + f0 * d[0], a[1] + f0 * d[1], a[2] + f0 * d[2]], [a[0] + f1 * d[0], a[1] + f1 * d[1], a[2] + f1 * d[2]])
}

/// postproc.postprocess (split off, join mode 'both')
pub fn postprocess(c: &[PCand], prob: &[f64], thr: f64, ev: Option<&Value>, prm: &PostParams) -> Vec<OutLine> {
    let n = c.len();
    let mut sel: Vec<usize> = (0..n).filter(|&i| prob[i] >= thr).collect();
    let sweep = ev.and_then(|e| e.get("sweep")).and_then(|s| s.as_array());
    if prm.sweep_min > 0.0 {
        if let Some(sw) = sweep {
            sel.retain(|&i| sw[i].is_null() || sw[i][0].as_f64().unwrap() >= prm.sweep_min);
        }
    }
    let low: Vec<usize> = (0..n)
        .filter(|&i| prm.join && prm.join_low < thr && prm.join_low <= prob[i] && prob[i] < thr)
        .collect();
    let is_low = |i: usize| low.binary_search(&i).is_ok();
    // pieces in insertion order: sel, then sorted low
    let mut piece_keys: Vec<usize> = Vec::new();
    for &i in sel.iter().chain(low.iter()) {
        if !piece_keys.contains(&i) {
            piece_keys.push(i);
        }
    }
    let in_pieces = |i: usize, pk: &Vec<usize>| pk.contains(&i);
    let mut parent: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    fn find(p: &mut std::collections::HashMap<usize, usize>, mut x: usize) -> usize {
        while *p.get(&x).unwrap_or(&x) != x {
            let px = p[&x];
            let ppx = *p.get(&px).unwrap_or(&px);
            p.insert(x, ppx);
            x = ppx;
        }
        x
    }
    if prm.join {
        if let Some(pairs) = ev.and_then(|e| e.get("pairs")).and_then(|v| v.as_array()) {
            for p in pairs {
                let i = p["i"].as_u64().unwrap() as usize;
                let j = p["j"].as_u64().unwrap() as usize;
                if !in_pieces(i, &piece_keys) || !in_pieces(j, &piece_keys) {
                    continue;
                }
                if is_low(i) && is_low(j) {
                    continue;
                }
                if !p["has_flanks"].as_bool().unwrap_or(false) {
                    continue;
                }
                let ok = p["ca"].as_bool().unwrap_or(false) && p["cb"].as_bool().unwrap_or(false);
                if !ok {
                    continue;
                }
                // single pieces [0, 1]: i's piece reaches its end; j's nearest end piece
                let ie = c[i].end;
                let dj0 = norm3(&sub3(&c[j].start, &ie));
                let dj1 = norm3(&sub3(&c[j].end, &ie));
                let _ = (dj0, dj1);
                let ra = find(&mut parent, i);
                let rb = find(&mut parent, j);
                if ra != rb {
                    parent.insert(ra, rb);
                }
            }
        }
    }
    // groups keyed by root, in order of first appearance
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for &i in &piece_keys {
        let r = find(&mut parent, i);
        match groups.iter_mut().find(|(k, _)| *k == r) {
            Some((_, v)) => v.push(i),
            None => groups.push((r, vec![i])),
        }
    }
    let mut out: Vec<OutLine> = Vec::new();
    for (_root, mem) in &groups {
        if mem.iter().all(|&m| is_low(m)) {
            continue;
        }
        if mem.len() == 1 {
            let i = mem[0];
            if is_low(i) {
                continue;
            }
            let (a, b) = sub_seg(&c[i], 0.0, 1.0);
            out.push(OutLine { start_s: a, end_s: b, score: prob[i], src: c[i].src.clone(), incl: c[i].incl, parts: vec![i],
                               op: "keep", fix: None, p_z3: None, ext_mm: None });
            continue;
        }
        let segs: Vec<((V3, V3), usize)> = mem.iter().map(|&i| (sub_seg(&c[i], 0.0, 1.0), i)).collect();
        let lens: Vec<f64> = segs.iter().map(|((a, b), _)| norm3(&sub3(b, a))).collect();
        let k = argmax(&lens);
        let ((a0, b0), i0) = segs[k];
        let dl = norm3(&sub3(&b0, &a0)).max(1e-9);
        let d = div3(&sub3(&b0, &a0), dl);
        let mut smin = f64::INFINITY;
        let mut smax = f64::NEG_INFINITY;
        for ((a, b), _) in &segs {
            for q in [a, b] {
                let s = dot3(&sub3(q, &a0), &d);
                smin = smin.min(s);
                smax = smax.max(s);
            }
        }
        let mut score = prob[segs[0].1];
        for &(_, i) in &segs[1..] {
            if prob[i] > score {
                score = prob[i];
            }
        }
        let mut src: Vec<String> = segs.iter().flat_map(|(_, i)| c[*i].src.clone()).collect();
        src.sort();
        src.dedup();
        let mut parts: Vec<usize> = segs.iter().map(|(_, i)| *i).collect();
        parts.sort();
        parts.dedup();
        out.push(OutLine {
            start_s: [a0[0] + smin * d[0], a0[1] + smin * d[1], a0[2] + smin * d[2]],
            end_s: [a0[0] + smax * d[0], a0[1] + smax * d[1], a0[2] + smax * d[2]],
            score,
            src,
            incl: c[i0].incl,
            parts,
            op: "join",
            fix: None,
            p_z3: None,
            ext_mm: None,
        });
    }
    if prm.dedupe_join && out.iter().any(|o| o.op == "join") {
        out = dedupe_joined(out);
    }
    out.sort_by(|x, y| {
        let kx = *x.parts.iter().min().unwrap();
        let ky = *y.parts.iter().min().unwrap();
        kx.cmp(&ky).then_with(|| {
            for q in 0..3 {
                match total_cmp(&x.start_s[q], &y.start_s[q]) {
                    std::cmp::Ordering::Equal => continue,
                    o => return o,
                }
            }
            std::cmp::Ordering::Equal
        })
    });
    out
}

fn olen(o: &OutLine) -> f64 {
    norm3(&sub3(&o.end_s, &o.start_s))
}

/// postproc._pair_geom / fixes.pair_geom (same arithmetic)
pub fn pair_geom(a0: &V3, a1: &V3, b0: &V3, b1: &V3) -> (f64, f64, f64) {
    let da = sub3(a1, a0);
    let db = sub3(b1, b0);
    let la = norm3(&da);
    let lb = norm3(&db);
    let da = div3(&da, la.max(1e-12));
    let db = div3(&db, lb.max(1e-12));
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

fn pair_geom_post(a0: &V3, a1: &V3, b0: &V3, b1: &V3) -> (f64, f64, f64) {
    // postproc._pair_geom normalises by max(l, 1e-9)
    let da = sub3(a1, a0);
    let db = sub3(b1, b0);
    let la = norm3(&da);
    let lb = norm3(&db);
    let da = div3(&da, la.max(1e-9));
    let db = div3(&db, lb.max(1e-9));
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

fn dedupe_joined(out: Vec<OutLine>) -> Vec<OutLine> {
    let mut drop = std::collections::BTreeSet::new();
    let mut joined: Vec<usize> = (0..out.len()).filter(|&k| out[k].op == "join").collect();
    joined.sort_by(|&a, &b| total_cmp(&(-olen(&out[a])), &(-olen(&out[b]))));
    for k in joined {
        if drop.contains(&k) {
            continue;
        }
        let lk = olen(&out[k]);
        for m in 0..out.len() {
            if m == k || drop.contains(&m) {
                continue;
            }
            let (ang, off, ov) = pair_geom_post(&out[k].start_s, &out[k].end_s, &out[m].start_s, &out[m].end_s);
            let lm = olen(&out[m]);
            if ang < 4.0 && off < 2.5 && ov > 0.5 && lm <= lk {
                drop.insert(m);
            }
        }
    }
    out.into_iter().enumerate().filter(|(m, _)| !drop.contains(m)).map(|(_, o)| o).collect()
}

// ------------------------------------------------------------------ v3 fixes

pub struct FixParams {
    pub ext: bool,
    pub ext_ratio: f64,
    pub ext_min_in: f64,
    pub ext_dedupe: bool,
    pub z3r: bool,
    pub z3r_tau: f64,
    pub z3r_sweep_min: f64,
}

fn is_dup(a0: &V3, a1: &V3, b0: &V3, b1: &V3) -> bool {
    let (ang, off, ov) = pair_geom(a0, a1, b0, b1);
    ang < 4.0 && off < 2.5 && ov > 0.5
}

fn extend(lines: Vec<OutLine>, c: &[PCand], z2: &[Z2Line], prm: &FixParams) -> Vec<OutLine> {
    // z2_attachment
    let z3i: Vec<usize> = (0..c.len()).filter(|&j| c[j].has_z3).collect();
    let mut att: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (li, ln) in z2.iter().enumerate() {
        let mut best: Option<f64> = None;
        let mut bj: i64 = -1;
        for &j in &z3i {
            let (ang, off, ov) = pair_geom(&ln.start, &ln.end, &c[j].start, &c[j].end);
            if ang < 4.0 && off < 2.5 && ov > 0.5 {
                let sc = ang / 4.0 + off / 2.5 - ov;
                if best.is_none() || sc < best.unwrap() {
                    best = Some(sc);
                    bj = j as i64;
                }
            }
        }
        if bj >= 0 {
            att.entry(bj as usize).or_default().push(li);
        }
    }
    let mut out: Vec<OutLine> = Vec::new();
    for ln in lines {
        let a = ln.start_s;
        let b = ln.end_s;
        let l = norm3(&sub3(&b, &a));
        let u = div3(&sub3(&b, &a), l.max(1e-9));
        let (mut lo, mut hi) = (0.0f64, l);
        for &i in &ln.parts {
            if let Some(lst) = att.get(&i) {
                for &li in lst {
                    let z = &z2[li];
                    let lz = norm3(&sub3(&z.end, &z.start));
                    let x0 = dot3(&sub3(&z.start, &a), &u);
                    let x1 = dot3(&sub3(&z.end, &a), &u);
                    let (s0, s1) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
                    let inside = 0.0f64.max(hi.min(s1) - lo.max(s0)) / (hi - lo).max(1e-9);
                    if lz >= prm.ext_ratio * (hi - lo) && inside >= prm.ext_min_in {
                        lo = lo.min(s0);
                        hi = hi.max(s1);
                    }
                }
            }
        }
        let mut nl = ln.clone();
        if lo < -1e-9 || hi > l + 1e-9 {
            nl.start_s = [a[0] + lo * u[0], a[1] + lo * u[1], a[2] + lo * u[2]];
            nl.end_s = [a[0] + hi * u[0], a[1] + hi * u[1], a[2] + hi * u[2]];
            nl.fix = Some("EXT");
            nl.ext_mm = Some([crate::z2graph::py_round(-lo, 3), crate::z2graph::py_round(hi - l, 3)]);
        }
        out.push(nl);
    }
    if !prm.ext_dedupe {
        return out;
    }
    let mut drop = std::collections::BTreeSet::new();
    let mut extended: Vec<usize> = (0..out.len()).filter(|&k| out[k].fix == Some("EXT")).collect();
    extended.sort_by(|&a, &b| total_cmp(&(-olen(&out[a])), &(-olen(&out[b]))));
    for k in extended {
        if drop.contains(&k) {
            continue;
        }
        let lk = olen(&out[k]);
        for m in 0..out.len() {
            if m != k && !drop.contains(&m) && olen(&out[m]) <= lk
                && is_dup(&out[k].start_s, &out[k].end_s, &out[m].start_s, &out[m].end_s)
            {
                drop.insert(m);
            }
        }
    }
    out.into_iter().enumerate().filter(|(m, _)| !drop.contains(m)).map(|(_, o)| o).collect()
}

fn rescue(lines: Vec<OutLine>, c: &[PCand], prob: &[f64], p3: &[f64], sweep: Option<&Vec<Value>>, prm: &FixParams) -> Vec<OutLine> {
    let sweep = match sweep {
        Some(s) => s,
        None => return lines,
    };
    let used: std::collections::BTreeSet<usize> = lines.iter().flat_map(|l| l.parts.iter().cloned()).collect();
    let mut out = lines;
    for k in 0..c.len() {
        if c[k].src != vec!["z3".to_string()] || used.contains(&k) {
            continue;
        }
        if !(p3[k] >= prm.z3r_tau) {
            continue;
        }
        let sw = if k < sweep.len() { &sweep[k] } else { &Value::Null };
        if sw.is_null() || !(sw[0].as_f64().unwrap() >= prm.z3r_sweep_min) {
            continue;
        }
        let (a, b) = (c[k].start_s, c[k].end_s);
        if out.iter().any(|o| is_dup(&a, &b, &o.start_s, &o.end_s)) {
            continue;
        }
        out.push(OutLine { start_s: a, end_s: b, score: prob[k], src: c[k].src.clone(), incl: c[k].incl, parts: vec![k],
                           op: "keep", fix: Some("Z3R"), p_z3: Some(p3[k]), ext_mm: None });
    }
    out
}

pub fn apply_fixes(lines: Vec<OutLine>, c: &[PCand], z2: &[Z2Line], prob: &[f64], p3: &[f64], ev: Option<&Value>,
                   prm: &FixParams) -> Vec<OutLine> {
    let mut out = if prm.ext { extend(lines, c, z2, prm) } else { lines };
    if prm.z3r {
        let sweep = ev.and_then(|e| e.get("sweep")).and_then(|s| s.as_array());
        out = rescue(out, c, prob, p3, sweep, prm);
    }
    out
}
