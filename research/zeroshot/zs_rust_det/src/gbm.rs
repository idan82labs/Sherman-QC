//! Inference for the exported sklearn HistGradientBoostingClassifier models (format sherman.zs_gbm.v1, written by
//! research/zeroshot/zs_lane_v3_rs/export_models.py): raw = baseline + sum of tree leaves in tree order (numerical
//! splits: x <= threshold goes left, NaN follows missing_go_to_left), probability = 1 / (1 + exp(-raw)) as
//! scipy.special.expit computes it.
#[allow(unused_imports)]
use crate::fmath::{DetF32, DetF64};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Tree {
    pub feature: Vec<i64>,
    pub threshold: Vec<f64>,
    pub missing_left: Vec<u8>,
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    pub leaf: Vec<u8>,
    pub value: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Model {
    pub format: String,
    pub n_features: usize,
    pub baseline: f64,
    pub trees: Vec<Tree>,
    pub source_sha256: String,
}

impl Model {
    pub fn load(path: &std::path::Path) -> Result<Model, String> {
        let m: Model = serde_json::from_str(&std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?)
            .map_err(|e| e.to_string())?;
        if m.format != "sherman.zs_gbm.v1" {
            return Err(format!("unsupported model format {}", m.format));
        }
        Ok(m)
    }
    pub fn raw(&self, x: &[f64]) -> f64 {
        let mut r = 0.0f64 + self.baseline;
        for t in &self.trees {
            let mut n = 0usize;
            while t.leaf[n] == 0 {
                let v = x[t.feature[n] as usize];
                n = if v.is_nan() {
                    if t.missing_left[n] != 0 { t.left[n] as usize } else { t.right[n] as usize }
                } else if v <= t.threshold[n] {
                    t.left[n] as usize
                } else {
                    t.right[n] as usize
                };
            }
            r += t.value[n];
        }
        r
    }
    pub fn proba(&self, x: &[f64]) -> f64 {
        1.0 / (1.0 + (-self.raw(x)).dexp())
    }
}
