//! JSON text as Python's json.dumps writes it (indent = n, separators ", " / ": ", ensure_ascii, float repr),
//! so eval_lines.json and the run record read like the Python lane's.

use serde_json::Value;

/// Python repr(float)
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // shortest round-trip digits and decimal exponent from Rust's {:e}
    let s = format!("{:e}", x);
    let (mant, exp) = s.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1; // position of the decimal point relative to the digit string
    let sign = if neg { "-" } else { "" };
    if decpt > -4 && decpt <= 16 {
        let n = digits.len() as i32;
        if decpt <= 0 {
            format!("{}0.{}{}", sign, "0".repeat((-decpt) as usize), digits)
        } else if decpt >= n {
            format!("{}{}{}.0", sign, digits, "0".repeat((decpt - n) as usize))
        } else {
            format!("{}{}.{}", sign, &digits[..decpt as usize], &digits[decpt as usize..])
        }
    } else {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        let e = decpt - 1;
        format!("{}{}e{}{:02}", sign, m, if e < 0 { "-" } else { "+" }, e.abs())
    }
}

fn esc(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{08}' => o.push_str("\\b"),
            '\u{0c}' => o.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    o.push_str(&format!("\\u{:04x}", u));
                }
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn num(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    float_repr(n.as_f64().unwrap())
}

fn write(v: &Value, indent: usize, level: usize, out: &mut String) {
    let pad = |l: usize| " ".repeat(indent * l);
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&num(n)),
        Value::String(s) => out.push_str(&esc(s)),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push_str("[\n");
            for (i, x) in a.iter().enumerate() {
                out.push_str(&pad(level + 1));
                write(x, indent, level + 1, out);
                if i + 1 < a.len() {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(level));
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push_str("{\n");
            let n = m.len();
            for (i, (k, x)) in m.iter().enumerate() {
                out.push_str(&pad(level + 1));
                out.push_str(&esc(k));
                out.push_str(": ");
                write(x, indent, level + 1, out);
                if i + 1 < n {
                    out.push(',');
                }
                out.push('\n');
            }
            out.push_str(&pad(level));
            out.push('}');
        }
    }
}

pub fn dumps(v: &Value, indent: usize) -> String {
    let mut s = String::new();
    write(v, indent, 0, &mut s);
    s
}
