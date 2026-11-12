//! Inference-engine abstraction.
//!
//! "Detecting" an engine means more than checking whether a command exists on
//! `PATH`: we *also* scan known install locations on disk (Homebrew, `~/.ohllama/engines`,
//! `cargo` bins, `ollama` dir, `vllm` python env, etc.) so that pre-compiled
//! binaries the user downloaded but did not add to PATH are still found.
//!
//! Each engine records:
//! - a `path` to an executable / directory when found on disk,
//! - a `present` bool (command on PATH **or** file on disk),
//! - a list of `models` it can serve (model files / GGUF it can load).

use crate::config;
use crate::hardware::Hardware;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Known engine catalog. `bin` is the command name to probe on PATH;
/// `disk` lists extra on-disk locations to scan for a precompiled copy.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Engine {
    pub name: String,
    pub bin: String,
    pub disk_hints: Vec<String>,
    /// Resolved path if found (cmd on PATH, or a file/dir on disk).
    pub path: Option<PathBuf>,
    /// Whether we found *any* copy (command or file).
    pub present: bool,
    /// True when the copy came from a disk scan (not PATH) — "pre-compiled".
    pub precompiled: bool,
    pub is_default: bool,
    /// Human notes about what this engine is / its backends.
    pub note: String,
}

/// The set of engines we probe. Kept intentionally small but extensible.
pub const ENGINES: &[&str] = &["llama.cpp", "ollama", "vllm", "lmstudio", "ggml"];

/// Backends the perf model understands.
pub const BACKENDS: &[&str] = &["cpu", "cpu-avx2", "cuda", "metal", "vulkan", "vllm"];

/// Detect every engine, by command **and** by on-disk file.
pub fn detect() -> Vec<Engine> {
    let cfg = config::cfg();
    let default = cfg.default_engine.clone();
    let mut out = Vec::new();
    for spec in ENGINES.iter() {
        let (bin, disk_hints, note) = engine_spec(spec);
        let path_on_path = which(bin);
        let on_disk = find_on_disk(spec, &disk_hints);
        let present = path_on_path.is_some() || on_disk.is_some();
        let precompiled = on_disk.is_some() && path_on_path.is_none();
        let found = path_on_path.clone().or(on_disk);
        out.push(Engine {
            name: spec.to_string(),
            bin: bin.to_string(),
            disk_hints,
            path: found,
            present,
            precompiled,
            is_default: !default.is_empty() && default == *spec,
            note: note.to_string(),
        });
    }
    out
}

/// Download URL for an engine (best-effort; the real download is the app layer's job).
pub fn download_url(name: &str) -> Option<String> {
    use crate::MIRRORS;
    let base = MIRRORS[0]; // not used for engines, but kept for symmetry
    let _ = base;
    match name {
        "llama.cpp" => Some(
            "https://github.com/ggerganov/llama.cpp/releases/latest/download/llama-b<arch>-bin.tar.gz"
                .to_string(),
        ),
        "ollama" => Some(
            "https://ollama.com/download/ollama-<arch>.zip".to_string(),
        ),
        "vllm" => Some("pip install vllm".to_string()),
        "lmstudio" => Some(
            "https://lmstudio.ai/download".to_string(),
        ),
        "ggml" => Some(
            "https://github.com/ggerganov/llama.cpp/releases/latest/download/".to_string(),
        ),
        _ => None,
    }
}

/// Where to put a downloaded engine binary.
pub fn install_dir() -> PathBuf {
    config::engines_dir()
}

fn which(bin: &str) -> Option<PathBuf> {
    if let Ok(out) = Command::new(bin).arg("--version").output() {
        if out.status.success() {
            // found on PATH
            return Some(PathBuf::from(bin));
        }
    }
    None
}

/// Scan known disk locations for a precompiled copy.
fn find_on_disk(name: &str, hints: &[String]) -> Option<PathBuf> {
    // 1) the user's own engines dir
    let local = config::engines_dir().join(name);
    if local.exists() {
        return Some(local);
    }
    // 2) homebrew / common install roots
    let home = dirs::home_dir().unwrap_or_default();
    for hint in hints {
        let p: PathBuf = match hint.as_str() {
            "homebrew" => home.join(".local/bin/llama.cpp"),
            "ollama" => home.join(".ollama/bin/ollama"),
            _ => continue,
        };
        if p.exists() {
            return Some(p);
        }
    }
    // 3) cargo bins
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        let bin = Path::new(&cargo_home)
            .join("bin")
            .join(if name == "llama.cpp" {
                "llama-cli"
            } else {
                name
            });
        if bin.exists() {
            return Some(bin);
        }
    }
    None
}

fn engine_spec(name: &str) -> (&str, Vec<String>, &str) {
    match name {
        "llama.cpp" => (
            "llama-cli",
            vec!["homebrew".into(), "ollama".into()],
            "GGUF inference; CPU/CUDA/Metal/Vulkan backends.",
        ),
        "ollama" => (
            "ollama",
            vec!["ollama".into()],
            "OpenAI-compatible server; auto-downloads GGUF.",
        ),
        "vllm" => (
            "vllm",
            vec![],
            "High-throughput PagedAttention; GPU-first (CUDA).",
        ),
        "lmstudio" => (
            "lmstudio-ai-server",
            vec![],
            "Desktop GUI + server; GGUF + ML models.",
        ),
        "ggml" => (
            "llama-server",
            vec!["homebrew".into()],
            "llama.cpp HTTP server binary.",
        ),
        _ => (name, vec![], "unknown"),
    }
}

/// Which backend of the *recommended* engine should run, given this hardware.
pub fn recommend(hw: &Hardware) -> (String, String) {
    let backend = hw.recommended_backend();
    let engine = if backend == "vllm" {
        "vllm"
    } else {
        // cpu / cpu-avx2 / metal / cuda / vulkan all use llama.cpp
        "llama.cpp"
    };
    (engine.to_string(), backend)
}
