//! Wall-clock stage timers (perf-style, accumulated by name) reported in the run record ("timers_s").
use std::sync::Mutex;
use std::time::Instant;

static TIMERS: Mutex<Vec<(String, f64, u64)>> = Mutex::new(Vec::new());

pub fn add(name: &str, secs: f64) {
    let mut t = TIMERS.lock().unwrap();
    if let Some(e) = t.iter_mut().find(|e| e.0 == name) {
        e.1 += secs;
        e.2 += 1;
    } else {
        t.push((name.to_string(), secs, 1));
    }
}

pub fn time<T>(name: &str, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    add(name, t0.elapsed().as_secs_f64());
    r
}

pub fn snapshot() -> serde_json::Value {
    let t = TIMERS.lock().unwrap();
    let mut m = serde_json::Map::new();
    for (k, s, n) in t.iter() {
        m.insert(k.clone(), serde_json::json!({"s": (s * 1e4).round() / 1e4, "n": n}));
    }
    serde_json::Value::Object(m)
}

/// adds its lifetime to a named timer when dropped
pub struct Timer {
    name: &'static str,
    t0: Instant,
}
impl Timer {
    pub fn new(name: &'static str) -> Timer {
        Timer { name, t0: Instant::now() }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        add(self.name, self.t0.elapsed().as_secs_f64());
    }
}
