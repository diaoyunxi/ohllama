//! Model registry.
//!
//! `create` registers a *directory of model files* as a model **without moving
//! or modifying the files**. It auto-detects the tokenizer and config files
//! (`tokenizer.json`, `tokenizer_config.json`, `config.json`, `*.gguf`,
//! `*.safetensors`, ...) and records enough metadata to `run` the model later.

use crate::config;
use crate::model::Model;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// A registered model entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub name: String,
    pub backend: String,
    pub model_dir: String,
    /// Detected tokenizer / config files (relative to `model_dir`).
    pub files: Vec<String>,
    /// The chosen quant/weight file that `run` should load.
    pub weight: String,
    /// Optional human notes.
    pub note: String,
}

/// All registered models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    pub models: Vec<RegisteredModel>,
}

/// Auto-detect tokenizer/config/weight files under `dir`.
pub fn create(dir: &str) -> anyhow::Result<RegisteredModel> {
    let path = Path::new(dir);
    anyhow::ensure!(path.is_dir(), "not a directory: {dir}");
    let entry = detect_entry(dir);
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let backend = config::cfg().default_engine.clone();
    let reg = RegisteredModel {
        name,
        backend,
        model_dir: path.to_string_lossy().into_owned(),
        files: entry.files.clone(),
        weight: entry.weight.clone(),
        note: String::new(),
    };
    let mut store = load();
    store.models.retain(|m| m.name != reg.name);
    store.models.push(reg.clone());
    save(&store)?;
    println!(
        "registered `{}`  backend={}  weight={}",
        reg.name,
        reg.backend,
        if reg.weight.is_empty() {
            "<none>"
        } else {
            &reg.weight
        }
    );
    Ok(reg)
}

/// List all registered models.
pub fn list() -> Vec<RegisteredModel> {
    load().models
}

/// Remove a model from the registry only (files stay on disk).
pub fn remove(name: &str) -> anyhow::Result<()> {
    let mut store = load();
    let before = store.models.len();
    store.models.retain(|m| m.name != name);
    if store.models.len() == before {
        anyhow::bail!("no registered model named `{name}`");
    }
    save(&store)?;
    println!("unregistered `{name}` (files left in place)");
    Ok(())
}

fn load() -> Registry {
    std::fs::read_to_string(config::registry_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(store: &Registry) -> anyhow::Result<()> {
    let mut s = store.clone();
    // Drop any empty entries to keep the file tidy.
    s.models.retain(|m| !m.model_dir.is_empty());
    std::fs::write(config::registry_path(), serde_json::to_string_pretty(&s)?)?;
    Ok(())
}

struct Entry {
    files: Vec<String>,
    weight: String,
}

/// Scan `dir` for tokenizer/config/weight files and pick a weight.
fn detect_entry(dir: &str) -> Entry {
    let mut files: Vec<String> = Vec::new();
    let mut weight = String::new();
    let mut weight_score = 0u32;
    let score_of = |f: &str| -> u32 {
        let f = f.to_lowercase();
        if f.contains("safetensors") {
            5
        } else if f.contains("gguf") {
            4
        } else if f.contains(".bin") {
            3
        } else if f.contains("onnx") {
            2
        } else {
            0
        }
    };
    for entry in WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .take(500)
    {
        if entry.file_type().is_file() {
            let fname = entry.file_name().to_string_lossy().into_owned();
            let lower = fname.to_lowercase();
            let is_meta = lower.starts_with("tokenizer")
                || lower.starts_with("config")
                || lower.starts_with("vocab")
                || lower.starts_with("generation")
                || lower.starts_with("model")
                || lower.ends_with(".json")
                || lower.ends_with(".gguf")
                || lower.contains("safetensors")
                || lower.ends_with(".bin")
                || lower.ends_with(".onnx");
            if is_meta {
                let rel = entry.path().to_string_lossy().into_owned();
                files.push(rel);
                let s = score_of(&fname);
                if s > weight_score {
                    weight_score = s;
                    weight = fname.clone();
                }
            }
        }
    }
    files.sort();
    Entry { files, weight }
}

/// A helper the GUI/CLI use to turn a `Model` (HF) into a `RegisteredModel`
/// pointing at a local download.
#[allow(dead_code)]
pub fn register_local(model: &Model, local_dir: &Path) -> anyhow::Result<RegisteredModel> {
    let entry = detect_entry(local_dir.to_string_lossy().as_ref());
    let name = model.display_name();
    let backend = config::cfg().default_engine.clone();
    let reg = RegisteredModel {
        name,
        backend,
        model_dir: local_dir.to_string_lossy().into_owned(),
        files: entry.files,
        weight: entry.weight,
        note: String::new(),
    };
    let mut store = load();
    store.models.retain(|m| m.name != reg.name);
    store.models.push(reg.clone());
    save(&store)?;
    Ok(reg)
}

pub fn _unused() -> PathBuf {
    config::registry_path()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// Build a fake model directory with typical HF file layout.
    fn make_model_dir() -> tempfile::TempDir {
        crate::config::init();
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path();
        let _ = fs::write(base.join("config.json"), "{}");
        let _ = fs::write(base.join("tokenizer.json"), "{}");
        let _ = fs::write(base.join("tokenizer_config.json"), "{}");
        let _ = fs::write(base.join("model-00001-of-00002.safetensors"), "x");
        let _ = fs::write(base.join("model-00002-of-00002.safetensors"), "x");
        let _ = fs::create_dir(base.join("sub"));
        let _ = fs::write(base.join("sub").join("weights.gguf"), "x");
        dir
    }

    #[test]
    fn detect_entry_finds_metadata_and_prefers_safetensors() {
        let dir = make_model_dir();
        let entry = detect_entry(dir.path().to_string_lossy().as_ref());
        // All top-level + nested metadata found.
        assert!(entry.files.len() >= 5, "found: {:?}", entry.files);
        // safetensors beats gguf beats onnx/bin.
        assert!(
            entry.weight.contains("safetensors"),
            "weight: {}",
            entry.weight
        );
        // Files are sorted for stable output.
        let mut sorted = entry.files.clone();
        sorted.sort();
        assert_eq!(entry.files, sorted);
    }

    #[test]
    fn detect_entry_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let entry = detect_entry(dir.path().to_string_lossy().as_ref());
        assert!(entry.files.is_empty());
        assert!(entry.weight.is_empty());
    }

    #[test]
    fn create_rejects_non_directory() {
        crate::config::init();
        assert!(create("/definitely/not/a/real/path/ohllama-test").is_err());
    }
}
