//! ohllama — local LLM manager (application layer).
//!
//! Provides:
//! - A Slint-based GUI window (Claude-style) with three pages:
//!   1. **Engines** — detect available inference engines (binary + on-disk check),
//!      estimate the best backend for the current hardware, and set a default.
//!   2. **Models** — pull hot/coding/image/video/tiny models from HuggingFace (with
//!      mirror fallback), compute recommended quantization and a perf estimate
//!      (TTFT / TPOT / total time) per model, and open a detail view with a
//!      quantization table plus a download button that picks the fastest mirror
//!      and uses up to 32/64 threads.
//!   3. **Chat** — a simple chat UI that talks to the running local engine.
//! - A CLI: `serve`, `create`, `run`, `stop`, `list`, `rm`, `help`.
//! - An HTTP API layer started by `serve` (model registry, inference, downloads).
//!
//! Inference engines themselves (llama.cpp, vLLM, ollama, ...) are *not* vendored —
//! the engine layer is abstracted and hidden behind the `engine` module, which only
//! probes for their presence and spawns them.

pub mod api;
pub mod chat;
pub mod config;
pub mod download;
pub mod engine;
pub mod hardware;
pub mod hf;
pub mod model;
pub mod perf;
pub mod registry;
pub mod runtime;

pub mod gui;

use anyhow::Result;

/// Default HuggingFace endpoint. The user can point `mirror_url` at
/// `https://hf-mirror.com` (or any compatible mirror) to swap the base host.
pub const DEFAULT_HF_ENDPOINT: &str = "https://hf-mirror.com";
/// Secondary mirror list used for the "fastest mirror" picker on download.
pub const MIRRORS: &[&str] = &[
    "https://hf-mirror.com",
    "https://huggingface.co",
    "https://hf.co",
];
/// Number of parallel download threads (user requested 32/64).
pub const DOWNLOAD_THREADS: usize = 32;

fn main() -> Result<()> {
    config::init();
    let args: Vec<String> = std::env::args().collect();
    // If a sub-command is given, run CLI; otherwise open the GUI.
    let maybe_cmd = args.get(1).cloned();
    match maybe_cmd.as_deref() {
        Some("gui") | Some("-h") | Some("--help") | None => {
            if matches!(maybe_cmd.as_deref(), Some("-h") | Some("--help")) {
                cli::help();
                return Ok(());
            }
            gui::run()?;
        }
        Some(cmd) => cli::run(cmd, &args[2..])?,
    }
    Ok(())
}

pub mod cli {
    use crate::api;
    use crate::engine;
    use crate::registry;
    use crate::runtime;
    use anyhow::Result;

    /// Top-level command dispatch.
    pub fn run(cmd: &str, rest: &[String]) -> Result<()> {
        match cmd {
            "serve" => {
                let port: u16 = rest
                    .iter()
                    .find(|a| a.starts_with("--port="))
                    .and_then(|a| a.trim_start_matches("--port=").parse().ok())
                    .unwrap_or(8787);
                api::serve(port)
            }
            "create" => {
                let dir = rest.first().cloned().unwrap_or_else(|| {
                    std::env::current_dir()
                        .map(|p| p.to_string_lossy().into_owned())
                        .unwrap_or_else(|_| ".".into())
                });
                let _reg = registry::create(&dir)?;
                Ok(())
            }
            "run" => {
                let name = rest
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("usage: ohllama run <model>"))?;
                let _running = runtime::spawn_model(name)?;
                Ok(())
            }
            "stop" => {
                let name = rest
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("usage: ohllama stop <model>"))?;
                runtime::stop_model(name)?;
                Ok(())
            }
            "list" => {
                let out = registry::list();
                if out.is_empty() {
                    println!("(no registered models)");
                } else {
                    for m in out {
                        println!("{}  [{}]  path={}", m.name, m.backend, m.model_dir);
                    }
                }
                Ok(())
            }
            "rm" => {
                let name = rest
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("usage: ohllama rm <model>"))?;
                registry::remove(name)
            }
            "engines" => {
                for e in engine::detect() {
                    println!(
                        "{}  ok={}  default={}  bin={}",
                        e.name,
                        e.present,
                        e.is_default,
                        e.path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_else(|| "<not found>".into())
                    );
                }
                Ok(())
            }
            "help" | "--help" | "-h" => {
                help();
                Ok(())
            }
            other => anyhow::bail!("unknown command `{}`. Run `ohllama help` for usage.", other),
        }
    }

    pub fn help() {
        let endpoint = crate::DEFAULT_HF_ENDPOINT;
        println!(
            "ohllama — local LLM manager

Usage: ohllama [command] [options]

Commands:
  serve [--port=N]   Start the HTTP API layer (default port 8787).
  create [DIR]       Register a model directory as a new model.
                     Files are not moved; tokenizer/config files are auto-detected.
                     Defaults to the current directory.
  run <MODEL>        Start the registered model on its (or the default) backend.
  stop <MODEL>       Stop a running model.
  list               List all registered models.
  rm <MODEL>         Unregister a model (files on disk are left untouched).
  engines            Print detected inference engines and which one is default.
  gui                Open the Slint GUI window.
  help               Show this help text.

Environment:
  OHLLAMA_HOME   Data directory (defaults to ~/.ohllama).
  HF_ENDPOINT    HuggingFace base URL (defaults to {endpoint}).
"
        );
    }
}
