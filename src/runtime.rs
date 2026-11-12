//! Runtime: start / stop a registered model on its backend.
//!
//! We do NOT reimplement inference. We locate the engine binary (from
//! `engine::detect`) and launch it with the right arguments, then track the
//! child PID so `stop` can terminate it. A small in-process map of running
//! model -> child process is kept under `$OHLLAMA_HOME/running.json`.

use crate::config;
use crate::engine;
use crate::registry;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};

/// A running model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningModel {
    pub name: String,
    pub pid: u32,
    pub engine: String,
    /// Port the engine's HTTP server listens on (if any).
    pub port: u16,
}

/// All running models, persisted so restarts can be reaped.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Running {
    pub models: Vec<RunningModel>,
}

fn running_path() -> std::path::PathBuf {
    config::data_dir().join("running.json")
}

fn load() -> Running {
    std::fs::read_to_string(running_path())
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save(r: &Running) -> std::io::Result<()> {
    std::fs::write(running_path(), serde_json::to_string_pretty(r)?)
}

/// Launch a registered model on its (or the default) backend.
pub fn spawn_model(name: &str) -> anyhow::Result<RunningModel> {
    let reg = registry::list()
        .into_iter()
        .find(|m| m.name == name)
        .ok_or_else(|| anyhow::anyhow!("model `{name}` not registered (run `ohllama create`)"))?;

    let engine_name = if reg.backend.is_empty() {
        let hw = crate::hardware::Hardware::detect();
        engine::recommend(&hw).0
    } else {
        reg.backend.clone()
    };

    let (bin, ok) = resolve_engine(&engine_name)?;
    anyhow::ensure!(ok, "engine `{engine_name}` not found on this machine");

    // Pick a port deterministically from the model name.
    let port = 11_000u16 + (stable_hash(name) % 1_000) as u16;

    let cmd_args = build_args(&engine_name, &reg.model_dir, &reg.weight, port);
    let mut child = Command::new(&bin);
    let spawned = child.args(&cmd_args).stdin(Stdio::null()).spawn()?;
    let pid = spawned.id();
    // Detach: we don't wait on it.
    drop(spawned);

    let entry = RunningModel {
        name: name.to_string(),
        pid,
        engine: engine_name.clone(),
        port,
    };
    let mut r = load();
    r.models.retain(|m| m.name != entry.name);
    r.models.push(entry.clone());
    save(&r)?;
    println!(
        "started `{}`  pid={}  engine={}  port={}",
        entry.name, entry.pid, entry.engine, entry.port
    );
    Ok(entry)
}

/// Stop a running model (terminate its process).
pub fn stop_model(name: &str) -> anyhow::Result<()> {
    let mut r = load();
    let entry = r
        .models
        .iter()
        .find(|m| m.name == name)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("model `{name}` is not running"))?;
    kill(entry.pid);
    r.models.retain(|m| m.name != name);
    save(&r)?;
    println!("stopped `{name}` (pid {})", entry.pid);
    Ok(())
}

/// Reap stale PIDs at startup (best-effort).
pub fn reap() {
    let mut r = load();
    r.models.retain(|m| pid_alive(m.pid));
    let _ = save(&r);
}

fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(target_os = "windows")]
    {
        match std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
        {
            Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&pid.to_string()),
            Err(_) => false,
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        // /proc/<pid> exists only while the process is alive.
        std::path::Path::new(&format!("/proc/{pid}")).exists()
    }
}

/// Send SIGTERM (or taskkill on Windows).
fn kill(pid: u32) {
    #[cfg(target_os = "windows")]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .output();
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .output();
    }
}

/// Resolve the engine to a real executable, preferring on-disk copies.
fn resolve_engine(name: &str) -> anyhow::Result<(String, bool)> {
    for e in engine::detect() {
        if e.name == name {
            if let Some(p) = &e.path {
                return Ok((p.display().to_string(), true));
            }
            return Ok((e.bin.clone(), false));
        }
    }
    anyhow::bail!("unknown engine `{name}`");
}

/// Build engine-specific arguments.
fn build_args(engine: &str, model_dir: &str, weight: &str, port: u16) -> Vec<String> {
    let _ = weight;
    match engine {
        "llama.cpp" | "ggml" => vec![
            "-m".into(),
            format!("{}/{}", model_dir, "model.bin"),
            "--port".into(),
            port.to_string(),
        ],
        "ollama" => {
            vec!["serve".into(), "--port".into(), port.to_string()]
        }
        "vllm" => {
            vec![
                "serve".into(),
                model_dir.into(),
                "--port".into(),
                port.to_string(),
            ]
        }
        "lmstudio" => {
            vec!["--port".into(), port.to_string()]
        }
        _ => Vec::new(),
    }
}

fn stable_hash(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33) ^ b as u32;
    }
    h
}
