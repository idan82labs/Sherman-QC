//! Scan readers equal to open3d.io.read_triangle_mesh for the formats the lane sees: PLY (ascii /
//! binary little / big endian; vertex x, y, z [+ nx, ny, nz]; triangular faces) and STL (binary / ascii; vertices
//! merged on exact equality in first-occurrence order, as open3d's reader does). Polygons with more than three
//! vertices are refused (no DEV scan has them; open3d's handling is not reproduced).
use crate::num::V3;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

pub struct RawMesh {
    pub v: Vec<V3>,
    pub f: Vec<[u32; 3]>,
    pub vn: Option<Vec<V3>>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

fn ty(s: &str) -> Result<Ty, String> {
    Ok(match s {
        "char" | "int8" => Ty::I8,
        "uchar" | "uint8" => Ty::U8,
        "short" | "int16" => Ty::I16,
        "ushort" | "uint16" => Ty::U16,
        "int" | "int32" => Ty::I32,
        "uint" | "uint32" => Ty::U32,
        "float" | "float32" => Ty::F32,
        "double" | "float64" => Ty::F64,
        _ => return Err(format!("unknown PLY type {}", s)),
    })
}

fn size(t: Ty) -> usize {
    match t {
        Ty::I8 | Ty::U8 => 1,
        Ty::I16 | Ty::U16 => 2,
        Ty::I32 | Ty::U32 | Ty::F32 => 4,
        Ty::F64 => 8,
    }
}

struct Prop {
    name: String,
    t: Ty,
    list: Option<Ty>,
}

struct Elem {
    name: String,
    n: usize,
    props: Vec<Prop>,
}

fn rd(b: &[u8], t: Ty, be: bool) -> f64 {
    macro_rules! get {
        ($ty:ty, $n:expr) => {{
            let mut a = [0u8; $n];
            a.copy_from_slice(&b[..$n]);
            if be { <$ty>::from_be_bytes(a) as f64 } else { <$ty>::from_le_bytes(a) as f64 }
        }};
    }
    match t {
        Ty::I8 => b[0] as i8 as f64,
        Ty::U8 => b[0] as f64,
        Ty::I16 => get!(i16, 2),
        Ty::U16 => get!(u16, 2),
        Ty::I32 => get!(i32, 4),
        Ty::U32 => get!(u32, 4),
        Ty::F32 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(&b[..4]);
            (if be { f32::from_be_bytes(a) } else { f32::from_le_bytes(a) }) as f64
        }
        Ty::F64 => get!(f64, 8),
    }
}

/// Binary little-endian fast path: the header is parsed, then each element is streamed from the file in blocks of
/// about 64 MB, each block decoded in parallel (fixed-stride vertex records, 3-index face records), so the peak
/// memory is the decoded arrays plus one block. Vertex normals are decoded only when asked for (the lane computes
/// its own). None when the layout is not covered (the generic reader then runs).
fn read_ply_fast(path: &Path, want_normals: bool) -> Option<Result<RawMesh, String>> {
    use rayon::prelude::*;
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let flen = file.metadata().ok()?.len() as usize;
    let mut head_buf = vec![0u8; (1 << 20).min(flen)];
    file.read_exact(&mut head_buf).ok()?;
    let hend = head_buf.windows(11).position(|w| w == b"end_header\n")? + 11;
    let head = String::from_utf8_lossy(&head_buf[..hend]).to_string();
    drop(head_buf);
    let mut elems: Vec<Elem> = Vec::new();
    let mut fmt = "";
    for l in head.lines() {
        let tok: Vec<&str> = l.split_whitespace().collect();
        if tok.is_empty() {
            continue;
        }
        match tok[0] {
            "format" => fmt = if tok[1] == "binary_little_endian" { "le" } else { "other" },
            "element" => elems.push(Elem { name: tok[1].to_string(), n: tok[2].parse().ok()?, props: vec![] }),
            "property" => {
                let e = elems.last_mut()?;
                if tok[1] == "list" {
                    e.props.push(Prop { name: tok[4].to_string(), t: ty(tok[3]).ok()?, list: Some(ty(tok[2]).ok()?) });
                } else {
                    e.props.push(Prop { name: tok[2].to_string(), t: ty(tok[1]).ok()?, list: None });
                }
            }
            _ => {}
        }
    }
    if fmt != "le" {
        return None;
    }
    // layout check first (every element must be covered before anything is decoded)
    for e in &elems {
        let scalar = e.props.iter().all(|p| p.list.is_none());
        if !(scalar || (e.name == "face" && e.props.len() == 1)) {
            return None;
        }
    }
    const BLOCK: usize = 64 << 20;
    let mut off = hend;
    let mut v: Vec<V3> = Vec::new();
    let mut vn: Option<Vec<V3>> = None;
    let mut f: Vec<[u32; 3]> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    for e in &elems {
        let scalar = e.props.iter().all(|p| p.list.is_none());
        let (stride, face) = if scalar {
            (e.props.iter().map(|p| size(p.t)).sum::<usize>(), false)
        } else {
            let p = &e.props[0];
            (size(p.list?) + 3 * size(p.t), true)
        };
        let end = off + stride * e.n;
        if end > flen {
            return if face { None } else { Some(Err("PLY truncated".into())) };
        }
        let want = e.name == "vertex" || face;
        if !want || stride == 0 {
            off = end;
            continue;
        }
        let pos = |n: &str| -> Option<(usize, Ty)> {
            let mut o = 0;
            for p in &e.props {
                if p.name == n {
                    return Some((o, p.t));
                }
                o += size(p.t);
            }
            None
        };
        let xyz = if face { None } else { Some((pos("x")?, pos("y")?, pos("z")?)) };
        let nrm = if face || !want_normals { None } else {
            match (pos("nx"), pos("ny"), pos("nz")) {
                (Some(a), Some(b), Some(c)) => Some((a, b, c)),
                _ => None,
            }
        };
        if face {
            f.reserve_exact(e.n);
        } else {
            v.reserve_exact(e.n);
            if nrm.is_some() {
                vn = Some(Vec::with_capacity(e.n));
            }
        }
        let rows = (BLOCK / stride).max(1);
        file.seek(SeekFrom::Start(off as u64)).ok()?;
        let mut done = 0usize;
        while done < e.n {
            let k = rows.min(e.n - done);
            buf.resize(k * stride, 0);
            if let Err(er) = file.read_exact(&mut buf) {
                return Some(Err(er.to_string()));
            }
            if face {
                let p = &e.props[0];
                let (ct, it) = (p.list?, p.t);
                if !buf.par_chunks(stride).all(|r| rd(r, ct, false) == 3.0) {
                    return None;
                }
                let (c, s) = (size(ct), size(it));
                let blk: Vec<[u32; 3]> = buf
                    .par_chunks(stride)
                    .map(|r| [rd(&r[c..], it, false) as u32, rd(&r[c + s..], it, false) as u32, rd(&r[c + 2 * s..], it, false) as u32])
                    .collect();
                f.extend_from_slice(&blk);
            } else {
                let (x, y, z) = xyz.unwrap();
                let blk: Vec<V3> = buf.par_chunks(stride).map(|r| [rd(&r[x.0..], x.1, false), rd(&r[y.0..], y.1, false), rd(&r[z.0..], z.1, false)]).collect();
                v.extend_from_slice(&blk);
                if let (Some((a, b, c)), Some(vv)) = (nrm, vn.as_mut()) {
                    let blk: Vec<V3> = buf.par_chunks(stride).map(|r| [rd(&r[a.0..], a.1, false), rd(&r[b.0..], b.1, false), rd(&r[c.0..], c.1, false)]).collect();
                    vv.extend_from_slice(&blk);
                }
            }
            done += k;
        }
        off = end;
    }
    Some(Ok(RawMesh { v, f, vn }))
}

pub fn read_ply(path: &Path) -> Result<RawMesh, String> {
    read_ply_opt(path, true)
}

pub fn read_ply_opt(path: &Path, want_normals: bool) -> Result<RawMesh, String> {
    if let Some(r) = read_ply_fast(path, want_normals) {
        return r;
    }
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut r = BufReader::with_capacity(1 << 22, file);
    let mut line = String::new();
    let mut elems: Vec<Elem> = Vec::new();
    let mut format = String::new();
    let mut first = true;
    loop {
        line.clear();
        let mut raw = Vec::new();
        if r.read_until(b'\n', &mut raw).map_err(|e| e.to_string())? == 0 {
            return Err("PLY header not terminated".into());
        }
        line.push_str(&String::from_utf8_lossy(&raw));
        let s = line.trim();
        if first {
            if s != "ply" {
                return Err("not a PLY file".into());
            }
            first = false;
            continue;
        }
        let tok: Vec<&str> = s.split_whitespace().collect();
        if tok.is_empty() {
            continue;
        }
        match tok[0] {
            "format" => format = tok[1].to_string(),
            "element" => elems.push(Elem { name: tok[1].to_string(), n: tok[2].parse().map_err(|_| "bad element count")?, props: vec![] }),
            "property" => {
                let e = elems.last_mut().ok_or("property before element")?;
                if tok[1] == "list" {
                    e.props.push(Prop { name: tok[4].to_string(), t: ty(tok[3])?, list: Some(ty(tok[2])?) });
                } else {
                    e.props.push(Prop { name: tok[2].to_string(), t: ty(tok[1])?, list: None });
                }
            }
            "end_header" => break,
            _ => {}
        }
    }
    let ascii = format == "ascii";
    let be = format == "binary_big_endian";
    let mut v: Vec<V3> = Vec::new();
    let mut vn: Vec<V3> = Vec::new();
    let mut f: Vec<[u32; 3]> = Vec::new();
    let mut has_n = false;
    let mut text = String::new();
    let mut toks: std::vec::IntoIter<String> = Vec::new().into_iter();
    if ascii {
        r.read_to_string(&mut text).map_err(|e| e.to_string())?;
        toks = text.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>().into_iter();
    }
    let mut buf = vec![0u8; 8];
    for e in &elems {
        let is_v = e.name == "vertex";
        let is_f = e.name == "face";
        let ix = |n: &str| e.props.iter().position(|p| p.name == n);
        let (px, py, pz) = (ix("x"), ix("y"), ix("z"));
        let (pnx, pny, pnz) = (ix("nx"), ix("ny"), ix("nz"));
        if is_v && pnx.is_some() && pny.is_some() && pnz.is_some() {
            has_n = true;
        }
        let pf = ix("vertex_indices").or(ix("vertex_index"));
        for _ in 0..e.n {
            let mut vals: Vec<f64> = vec![0.0; e.props.len()];
            let mut face: Vec<u32> = Vec::new();
            for (k, p) in e.props.iter().enumerate() {
                match p.list {
                    Some(ct) => {
                        let cnt = if ascii {
                            toks.next().ok_or("truncated")?.parse::<f64>().map_err(|_| "bad count")?
                        } else {
                            r.read_exact(&mut buf[..size(ct)]).map_err(|e| e.to_string())?;
                            rd(&buf, ct, be)
                        } as usize;
                        for _ in 0..cnt {
                            let x = if ascii {
                                toks.next().ok_or("truncated")?.parse::<f64>().map_err(|_| "bad index")?
                            } else {
                                r.read_exact(&mut buf[..size(p.t)]).map_err(|e| e.to_string())?;
                                rd(&buf, p.t, be)
                            };
                            if is_f && Some(k) == pf {
                                face.push(x as u32);
                            }
                        }
                    }
                    None => {
                        vals[k] = if ascii {
                            toks.next().ok_or("truncated")?.parse::<f64>().map_err(|_| "bad value")?
                        } else {
                            r.read_exact(&mut buf[..size(p.t)]).map_err(|e| e.to_string())?;
                            rd(&buf, p.t, be)
                        };
                    }
                }
            }
            if is_v {
                v.push([vals[px.ok_or("no x")?], vals[py.ok_or("no y")?], vals[pz.ok_or("no z")?]]);
                if has_n {
                    vn.push([vals[pnx.unwrap()], vals[pny.unwrap()], vals[pnz.unwrap()]]);
                }
            } else if is_f {
                if face.len() == 3 {
                    f.push([face[0], face[1], face[2]]);
                } else if !face.is_empty() {
                    return Err("PLY face with more than 3 vertices (not supported)".into());
                }
            }
        }
    }
    Ok(RawMesh { v, f, vn: if has_n { Some(vn) } else { None } })
}

/// open3d reads STL through assimp 5.4.2: one vertex per facet corner carrying the facet normal, then
/// aiProcess_JoinIdenticalVertices: a corner joins an earlier vertex with the same position (exact float bits,
/// +0 == -0) whose normal is within 1e-5 (squared float32 distance); first-occurrence order.
pub fn read_stl(path: &Path) -> Result<RawMesh, String> {
    let data = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let mut pos: Vec<[f32; 3]> = Vec::new();
    let mut nrm: Vec<[f32; 3]> = Vec::new();
    let binary = data.len() >= 84 && {
        let n = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        84 + 50 * n == data.len()
    };
    let rf = |q: usize| f32::from_le_bytes([data[q], data[q + 1], data[q + 2], data[q + 3]]);
    if binary {
        let n = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
        for i in 0..n {
            let o = 84 + 50 * i;
            let nn = [rf(o), rf(o + 4), rf(o + 8)];
            for k in 0..3 {
                let q = o + 12 + 12 * k;
                pos.push([rf(q), rf(q + 4), rf(q + 8)]);
                nrm.push(nn);
            }
        }
    } else {
        let text = String::from_utf8_lossy(&data);
        let mut cur_n = [0.0f32; 3];
        for l in text.lines() {
            let tk: Vec<&str> = l.split_whitespace().collect();
            if tk.len() >= 5 && tk[0] == "facet" && tk[1] == "normal" {
                for d in 0..3 {
                    cur_n[d] = tk[2 + d].parse::<f32>().unwrap_or(0.0);
                }
            } else if tk.first() == Some(&"vertex") && tk.len() >= 4 {
                pos.push([tk[1].parse().unwrap_or(f32::NAN), tk[2].parse().unwrap_or(f32::NAN), tk[3].parse().unwrap_or(f32::NAN)]);
                nrm.push(cur_n);
            }
        }
        pos.truncate(pos.len() / 3 * 3);
        nrm.truncate(pos.len());
    }
    let key = |p: &[f32; 3]| -> [u32; 3] {
        let z = |x: f32| if x == 0.0 { 0u32 } else { x.to_bits() };
        [z(p[0]), z(p[1]), z(p[2])]
    };
    let sq_eps = 1e-5f32 * 1e-5f32;
    let mut map: std::collections::HashMap<[u32; 3], Vec<u32>> = std::collections::HashMap::new();
    let mut v: Vec<V3> = Vec::new();
    let mut vnorm: Vec<[f32; 3]> = Vec::new();
    let mut remap = vec![0u32; pos.len()];
    for i in 0..pos.len() {
        let k = key(&pos[i]);
        let mut found: Option<u32> = None;
        if let Some(list) = map.get(&k) {
            for &u in list.iter().rev() {
                let a = vnorm[u as usize];
                let b = nrm[i];
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
                if !((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > sq_eps) {
                    found = Some(u);
                    break;
                }
            }
        }
        remap[i] = match found {
            Some(u) => u,
            None => {
                let u = v.len() as u32;
                v.push([pos[i][0] as f64, pos[i][1] as f64, pos[i][2] as f64]);
                vnorm.push(nrm[i]);
                map.entry(k).or_default().push(u);
                u
            }
        };
    }
    let f: Vec<[u32; 3]> = (0..pos.len() / 3).map(|t| [remap[3 * t], remap[3 * t + 1], remap[3 * t + 2]]).collect();
    let vn: Vec<V3> = vnorm.iter().map(|n| [n[0] as f64, n[1] as f64, n[2] as f64]).collect();
    Ok(RawMesh { v, f, vn: Some(vn) })
}

pub fn read_mesh(path: &Path) -> Result<RawMesh, String> {
    read_mesh_opt(path, true)
}

/// `want_normals` = false skips decoding a PLY's vertex normals in the fast path (the lane never uses them)
pub fn read_mesh_opt(path: &Path, want_normals: bool) -> Result<RawMesh, String> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if ext == "stl" {
        read_stl(path)
    } else {
        read_ply_opt(path, want_normals)
    }
}

/// face count from the header, as budget.face_count
pub fn face_count(path: &Path) -> Option<usize> {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let mut f = std::fs::File::open(path).ok()?;
    if ext == "stl" {
        let mut head = [0u8; 84];
        if f.read_exact(&mut head).is_err() {
            return None;
        }
        let n = u32::from_le_bytes([head[80], head[81], head[82], head[83]]) as u64;
        let len = std::fs::metadata(path).ok()?.len();
        return if 84 + 50 * n == len { Some(n as usize) } else { None };
    }
    let mut r = BufReader::new(f);
    let mut magic = [0u8; 3];
    if r.read_exact(&mut magic).is_err() || &magic != b"ply" {
        return None;
    }
    let mut nf = 0usize;
    let mut raw = Vec::new();
    r.read_until(b'\n', &mut raw).ok()?;
    for _ in 0..199 {
        raw.clear();
        if r.read_until(b'\n', &mut raw).ok()? == 0 {
            break;
        }
        let s: String = raw.iter().map(|&b| b as char).collect();
        let s = s.trim();
        if s.starts_with("element face") {
            nf = s.split_whitespace().nth(2)?.parse().ok()?;
        }
        if s == "end_header" {
            break;
        }
    }
    Some(nf)
}
