//! Minimal .npy (format 1.0 / 2.0) reader and writer for little-endian C-order arrays.
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

pub struct Array {
    pub shape: Vec<usize>,
    pub descr: String,
    pub data: Vec<u8>,
}

fn parse_header(h: &str) -> Result<(String, bool, Vec<usize>), String> {
    let get = |key: &str| -> Result<String, String> {
        let k = format!("'{}':", key);
        let i = h.find(&k).ok_or(format!("npy header lacks {}", key))? + k.len();
        Ok(h[i..].trim_start().to_string())
    };
    let d = get("descr")?;
    let d = d.trim_start_matches('\'');
    let descr = d[..d.find('\'').ok_or("bad descr")?].to_string();
    let f = get("fortran_order")?;
    let fortran = f.starts_with("True");
    let s = get("shape")?;
    let s = &s[1..s.find(')').ok_or("bad shape")?];
    let shape: Vec<usize> = s
        .split(',')
        .map(|x| x.trim())
        .filter(|x| !x.is_empty())
        .map(|x| x.parse::<usize>().map_err(|e| e.to_string()))
        .collect::<Result<_, _>>()?;
    Ok((descr, fortran, shape))
}

pub fn read(path: &Path) -> Result<Array, String> {
    let mut f = BufReader::new(File::open(path).map_err(|e| format!("{}: {}", path.display(), e))?);
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an npy file", path.display()));
    }
    let hlen = if magic[6] == 1 {
        let mut b = [0u8; 2];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        u16::from_le_bytes(b) as usize
    } else {
        let mut b = [0u8; 4];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        u32::from_le_bytes(b) as usize
    };
    let mut hb = vec![0u8; hlen];
    f.read_exact(&mut hb).map_err(|e| e.to_string())?;
    let h = String::from_utf8_lossy(&hb).to_string();
    let (descr, fortran, shape) = parse_header(&h)?;
    if fortran {
        return Err("fortran-order npy not supported".into());
    }
    let mut data = Vec::new();
    f.read_to_end(&mut data).map_err(|e| e.to_string())?;
    Ok(Array { shape, descr, data })
}

impl Array {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn to_f64(&self) -> Result<Vec<f64>, String> {
        let n = self.len();
        match self.descr.as_str() {
            "<f8" => Ok((0..n).map(|i| f64::from_le_bytes(self.data[8 * i..8 * i + 8].try_into().unwrap())).collect()),
            "<f4" => Ok((0..n)
                .map(|i| f32::from_le_bytes(self.data[4 * i..4 * i + 4].try_into().unwrap()) as f64)
                .collect()),
            d => Err(format!("expected float array, got {}", d)),
        }
    }
    pub fn to_i64(&self) -> Result<Vec<i64>, String> {
        let n = self.len();
        match self.descr.as_str() {
            "<i8" => Ok((0..n).map(|i| i64::from_le_bytes(self.data[8 * i..8 * i + 8].try_into().unwrap())).collect()),
            "<i4" => Ok((0..n)
                .map(|i| i32::from_le_bytes(self.data[4 * i..4 * i + 4].try_into().unwrap()) as i64)
                .collect()),
            "<u4" => Ok((0..n)
                .map(|i| u32::from_le_bytes(self.data[4 * i..4 * i + 4].try_into().unwrap()) as i64)
                .collect()),
            "|b1" | "|u1" => Ok(self.data[..n].iter().map(|&b| b as i64).collect()),
            d => Err(format!("expected int array, got {}", d)),
        }
    }
}

pub fn read_f64(path: &Path) -> Result<(Vec<f64>, Vec<usize>), String> {
    let a = read(path)?;
    Ok((a.to_f64()?, a.shape.clone()))
}

pub fn read_i64(path: &Path) -> Result<(Vec<i64>, Vec<usize>), String> {
    let a = read(path)?;
    Ok((a.to_i64()?, a.shape.clone()))
}

fn write_raw(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) -> Result<(), String> {
    let shp = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shp);
    let total = 10 + h.len() + 1;
    let pad = (64 - total % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut f = BufWriter::new(File::create(path).map_err(|e| format!("{}: {}", path.display(), e))?);
    f.write_all(b"\x93NUMPY\x01\x00").map_err(|e| e.to_string())?;
    f.write_all(&(h.len() as u16).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(h.as_bytes()).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn write_f64(path: &Path, shape: &[usize], v: &[f64]) -> Result<(), String> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    write_raw(path, "<f8", shape, &b)
}

pub fn write_i64(path: &Path, shape: &[usize], v: &[i64]) -> Result<(), String> {
    let mut b = Vec::with_capacity(v.len() * 8);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    write_raw(path, "<i8", shape, &b)
}
