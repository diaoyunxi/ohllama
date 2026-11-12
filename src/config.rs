//! Configuration + data-dir layout.
//!
//! Data lives under `~/.ohllama` (or `$OHLLAMA_HOME`):
//! ```text
//! ~/.ohllama/
//!   config.toml       # user settings
//!   registry.json     # registered models
//!   models/           # where the app downloads model files
//!   engines/          # where the app downloads inference engines
//! ```

use crate::MIRRORS;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Mutex;

static CONFIG: Lazy<Mutex<Config>> = Lazy::new(|| Mutex::new(Config::default()));

/// Root data directory.
pub fn data_dir() -> PathBuf {
    std::env::var("OHLLAMA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".ohllama"))
                .unwrap_or_default()
        })
}

pub fn config_path() -> PathBuf {
    data_dir().join("config.toml")
}

pub fn registry_path() -> PathBuf {
    data_dir().join("registry.json")
}

/// Directory where the app downloads model files.
pub fn models_dir() -> PathBuf {
    data_dir().join("models")
}

/// Directory where the app downloads inference engine binaries.
pub fn engines_dir() -> PathBuf {
    data_dir().join("engines")
}

/// Idempotently create the data dir + a default config file, and load into the global.
/// Call once from `main` (GUI and CLI both call `config::init()`).
pub fn init() {
    let _ = std::fs::create_dir_all(engines_dir());
    let _ = std::fs::create_dir_all(models_dir());
    if !config_path().exists() {
        let c = Config::default();
        if let Ok(s) = toml_string(&c) {
            let _ = std::fs::write(config_path(), s);
        }
    }
    if let Some(c) = load_file() {
        *CONFIG.lock().unwrap() = c;
    }
}

/// Access the global config.
pub fn cfg() -> Config {
    CONFIG.lock().unwrap().clone()
}

/// Persist the current global config.
pub fn save() -> std::io::Result<()> {
    let c = CONFIG.lock().unwrap().clone();
    std::fs::write(config_path(), toml_string(&c)?)
}

/// Update one field and save.
pub fn set_default_engine(name: &str) -> std::io::Result<()> {
    let mut c = CONFIG.lock().unwrap();
    c.default_engine = name.to_string();
    drop(c);
    save()
}

fn load_file() -> Option<Config> {
    let raw = std::fs::read_to_string(config_path()).ok()?;
    let v: serde_json::Value = toml_json(&raw).ok()?;
    serde_json::from_value(v).ok()
}

/// Minimal TOML->JSON bridge (avoids a dep on `toml`). We keep config keys flat.
fn toml_json(src: &str) -> Result<serde_json::Value, String> {
    let mut out = serde_json::Map::new();
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let k: String = k.trim().to_string();
            let v = parse_toml_value(v.trim());
            out.insert(k, v);
        }
    }
    Ok(serde_json::Value::Object(out))
}

fn parse_toml_value(v: &str) -> serde_json::Value {
    if v.starts_with('"') && v.ends_with('"') && v.len() >= 2 {
        return serde_json::Value::String(v[1..v.len() - 1].to_string());
    }
    if v == "true" {
        return serde_json::Value::Bool(true);
    }
    if v == "false" {
        return serde_json::Value::Bool(false);
    }
    if let Ok(i) = v.parse::<i64>() {
        return serde_json::Value::Number(serde_json::Number::from(i));
    }
    if let Ok(f) = v.parse::<f64>() {
        return serde_json::Value::Number(serde_json::Number::from_f64(f).unwrap());
    }
    if v.starts_with('[') && v.ends_with(']') {
        let inner = &v[1..v.len() - 1];
        let items: Vec<serde_json::Value> = inner
            .split(',')
            .filter_map(|s| {
                let s = s.trim();
                if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                    Some(serde_json::Value::String(s[1..s.len() - 1].to_string()))
                } else {
                    s.parse::<i64>().ok().map(serde_json::Value::from)
                }
            })
            .collect();
        return serde_json::Value::Array(items);
    }
    serde_json::Value::String(v.to_string())
}

fn toml_string(c: &Config) -> Result<String, std::io::Error> {
    let mut s = String::new();
    s.push_str("# ohllama config (TOML)\n");
    s.push_str(&format!(
        "# HuggingFace base. Point this at https://hf-mirror.com (or any mirror) to speed up\n# downloads/reads in restricted networks. Current: {}\n",
        c.hf_endpoint
    ));
    s.push_str(&format!("hf_endpoint = \"{}\"\n", c.hf_endpoint));
    s.push_str(&format!("hf_token = \"{}\"\n", c.hf_token));
    s.push_str(&format!(
        "mirror_candidates = [\"{}\"]\n",
        c.mirrors.join("\", \"")
    ));
    s.push_str(&format!("download_threads = {}\n", c.download_threads));
    s.push_str(&format!("default_engine = \"{}\"\n", c.default_engine));
    s.push_str(&format!("calibration_bw_eff = {}\n", c.calibration_bw_eff));
    s.push_str(&format!("eta_prefill = {}\n", c.eta_prefill));
    s.push_str(&format!("eta_weight = {}\n", c.eta_weight));
    s.push_str(&format!("eta_kv = {}\n", c.eta_kv));
    s.push_str(&format!("context_default = {}\n", c.context_default));
    Ok(s)
}

fn default_bw() -> f64 {
    40.0 // ~DDR4 dual channel; calibrate per machine
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// HuggingFace base URL. Defaults to a mirror so it works behind firewalls;
    /// can be switched back to `https://huggingface.co`.
    pub hf_endpoint: String,
    pub hf_token: String,
    /// Mirrors tried, in order of preference, when picking the fastest one.
    pub mirrors: Vec<String>,
    pub download_threads: usize,
    /// Default inference engine name (one of `engine::ENGINES`).
    pub default_engine: String,
    /// Effective bandwidth (GiB/s) used by the perf estimator.
    pub calibration_bw_eff: f64,
    /// Prefill MFU (fraction of peak FLOPs actually achieved). 0..=1.
    pub eta_prefill: f64,
    /// Weight-streaming efficiency (0..=1).
    pub eta_weight: f64,
    /// KV cache access efficiency (0..=1).
    pub eta_kv: f64,
    /// Default context length for perf estimates.
    pub context_default: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hf_endpoint: crate::DEFAULT_HF_ENDPOINT.to_string(),
            hf_token: String::new(),
            mirrors: MIRRORS.iter().map(|s| s.to_string()).collect(),
            download_threads: crate::DOWNLOAD_THREADS,
            default_engine: String::new(),
            calibration_bw_eff: default_bw(),
            eta_prefill: 0.5,
            eta_weight: 0.8,
            eta_kv: 0.7,
            context_default: 4096,
        }
    }
}
