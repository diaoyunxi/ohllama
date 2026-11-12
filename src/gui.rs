//! Slint GUI driver. Wires the `.slint` UI to the Rust backend.
//!
//! Slint 1.18 API:
//! - `slint::include_modules!()` pulls in generated types (`Main`, `EngineRow`, ...).
//! - `Main::new()?` -> `ComponentHandle`; `show()`/`run()` drive the event loop.
//! - Cross-thread work (network) runs on std threads and marshals UI updates back
//!   with `slint::invoke_from_event_loop` using a `slint::Weak<Main>` handle.
//! - `Model<T>` properties are set with `slint::ModelRc::new(slint::VecModel::from(vec))`.

slint::include_modules!();

use std::future::Future;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use crate::chat;
use crate::config;
use crate::download;
use crate::engine;
use crate::hardware;
use crate::hf;
use crate::model;
use crate::perf;

/// Run the GUI event loop. Blocks until the window is closed.
pub fn run() -> anyhow::Result<()> {
    let hw = hardware::Hardware::detect();
    tracing::info!("hardware: {}", hw.summary());

    let window = Main::new()?;
    let weak = window.as_weak();

    // Default chat model = first registered model, else a placeholder.
    let default_chat_model = crate::registry::list()
        .into_iter()
        .map(|m| m.name)
        .next()
        .unwrap_or_else(|| "(no model registered)".into());
    let chat_model_name: Arc<RwLock<String>> = Arc::new(RwLock::new(default_chat_model.clone()));

    // ---- Initial data (engines) — synchronous on the UI thread. ----
    refresh_engines(&window);
    window.set_chat_model(default_chat_model.into());

    // ---- Initial data (hot models) — async. ----
    fetch_category(weak.clone(), "hot");

    // ================= Callbacks: page 1 (Engines) =================
    let w = weak.clone();
    window.on_change_engine(move |action, name| {
        let action = action.to_string();
        let name = name.to_string();
        let w = w.clone();
        thread::spawn(move || {
            if action == "default" {
                let _ = config::set_default_engine(&name);
            } else {
                download_engine(&name);
            }
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = w.upgrade() {
                    refresh_engines(&w);
                }
            });
        });
    });

    // ================= Callbacks: page 2 (Models) =================
    let w = weak.clone();
    window.on_change_category(move |cat| {
        let cat = cat.to_string();
        let w = w.clone();
        fetch_category(w, &cat);
    });

    let w = weak.clone();
    window.on_do_search(move |q| {
        let q = q.to_string();
        let w = w.clone();
        do_search(w, &q);
    });

    let w = weak.clone();
    let m_name = chat_model_name.clone();
    window.on_open_model(move |repo| {
        let repo = repo.to_string();
        let w = w.clone();
        let m = m_name.clone();
        open_detail(w, &repo, &m);
    });

    let w = weak.clone();
    window.on_close_detail(move || {
        let w = w.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(w) = w.upgrade() {
                w.set_detail_visible(false);
            }
        });
    });

    window.on_download_model(move |repo, quant| {
        let repo = repo.to_string();
        let quant = quant.to_string();
        thread::spawn(move || {
            download_model_file(&repo, &quant);
        });
    });

    // ================= Callbacks: page 3 (Chat) =================
    let chat_hist: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let w = weak.clone();
    let model_name = chat_model_name.clone();
    window.on_send_chat(move |text| {
        let text = text.to_string();
        if text.trim().is_empty() {
            return;
        }
        let w = w.clone();
        let hist = chat_hist.clone();
        let m_name = model_name.clone();
        thread::spawn(move || {
            // 1) append user message + set "thinking…".
            let w1 = w.clone();
            let h1 = hist.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = w1.upgrade() {
                    let mut h = h1.lock().unwrap();
                    h.push(("user".to_string(), text.clone()));
                    w.set_chat(render_chat(&h));
                    w.set_chat_status("thinking…".into());
                }
            });

            // 2) compute reply (blocking network against the local engine).
            let mname = m_name.read().unwrap().clone();
            let hist_snapshot: Vec<(String, String)> = hist.lock().unwrap().clone();
            let reply = chat::send_blocking(&hist_snapshot, &mname).unwrap_or_else(|_| {
                "(engine not running - start it with `ohllama run <model>` first)".to_string()
            });

            // 3) append assistant message.
            let w2 = w.clone();
            let h2 = hist.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(w) = w2.upgrade() {
                    let mut h = h2.lock().unwrap();
                    h.push(("assistant".to_string(), reply));
                    w.set_chat(render_chat(&h));
                    w.set_chat_status(String::new().into());
                }
            });
        });
    });

    window.run()?;
    Ok(())
}

/// Populate page 1 (engines + hardware + recommendation).
fn refresh_engines(window: &Main) {
    let hw = hardware::Hardware::detect();
    let engines = engine::detect();
    let (rec_engine, rec_backend) = engine::recommend(&hw);
    let default = config::cfg().default_engine.clone();

    let rows: Vec<EngineRow> = engines
        .iter()
        .map(|e| EngineRow {
            name: e.name.clone().into(),
            present: e.present,
            precompiled: e.precompiled,
            is_default: e.is_default,
            recommended: e.name == rec_engine,
            path: e
                .path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
                .into(),
            note: e.note.clone().into(),
        })
        .collect();

    window.set_engines(slint::ModelRc::new(slint::VecModel::from(rows)));
    window.set_hw_summary(hw.summary().into());
    window.set_rec_engine(rec_engine.into());
    window.set_rec_backend(rec_backend.into());
    window.set_default_engine(default.into());
}

/// Best-effort engine download (the app layer only records the target; the
/// actual binary fetch is left to the user / a follow-up release pipeline).
fn download_engine(name: &str) {
    let dir = config::engines_dir();
    let _ = std::fs::create_dir_all(&dir);
    if let Some(url) = engine::download_url(name) {
        tracing::info!("engine `{name}` source: {url} -> {}", dir.display());
    } else {
        tracing::warn!("no download source for engine `{name}`");
    }
}

/// Fetch a category and update the model list (async on a worker thread).
fn fetch_category(weak: slint::Weak<Main>, cat: &str) {
    let cat = cat.to_string();
    thread::spawn(move || {
        let w1 = weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(w) = w1.upgrade() {
                w.set_models_loading(true);
            }
        });
        let res = block_on(hf::fetch_category(&cat));
        let _ = slint::invoke_from_event_loop(move || {
            let Some(w) = weak.upgrade() else { return };
            let hw = hardware::Hardware::detect();
            let cfg = config::cfg();
            let models = res.unwrap_or_default().models;
            let rows = build_model_rows(&models, &hw, &cfg);
            w.set_models(slint::ModelRc::new(slint::VecModel::from(rows)));
            w.set_models_loading(false);
        });
    });
}

/// Free-text search (async on a worker thread).
fn do_search(weak: slint::Weak<Main>, query: &str) {
    let q = query.to_string();
    thread::spawn(move || {
        let w1 = weak.clone();
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(w) = w1.upgrade() {
                w.set_models_loading(true);
            }
        });
        let res = block_on(hf::search(&q));
        let _ = slint::invoke_from_event_loop(move || {
            let Some(w) = weak.upgrade() else { return };
            let hw = hardware::Hardware::detect();
            let cfg = config::cfg();
            let models = res.unwrap_or_default();
            let rows = build_model_rows(&models, &hw, &cfg);
            w.set_models(slint::ModelRc::new(slint::VecModel::from(rows)));
            w.set_models_loading(false);
        });
    });
}

/// Convert `Model`s into UI rows with recommended quant + perf estimates.
fn build_model_rows(
    models: &[model::Model],
    hw: &hardware::Hardware,
    cfg: &config::Config,
) -> Vec<ModelRow> {
    let s_in = cfg.context_default;
    let s_out = 256u32;
    models
        .iter()
        .take(60)
        .map(|m| {
            let quants = perf::quants_for_model(m);
            let rec_i = perf::recommend_quant(&quants, hw).unwrap_or(0);
            let quant = &quants[rec_i];
            let est = perf::estimate(m, quant, s_in, s_out, hw, cfg);
            ModelRow {
                name: m.display_name().into(),
                provider: m.provider.clone().into(),
                rec_quant: quant.name.clone().into(),
                rec_quant_size: format!("{:.1} GB", quant.disk_gb).into(),
                ttft: format_s(est.ttft_s).into(),
                tpot: format_s(est.tpot_s).into(),
                total: format_s(est.total_s).into(),
                status_color: hex_color(est.status.color()),
                status_label: est.status_label().into(),
                repo_id: m.repo_id.clone().into(),
            }
        })
        .collect()
}

fn format_s(v: f64) -> String {
    if v < 1.0 {
        format!("{:.2}s", v)
    } else if v < 120.0 {
        format!("{:.0}s", v)
    } else {
        format!("{:.0}m", v / 60.0)
    }
}

/// Parse a `#rrggbb` (or `#rrggbbaa`) hex string into a `slint::Color`.
fn hex_color(hex: &str) -> slint::Color {
    let h = hex.trim_start_matches('#');
    let r = u32::from_str_radix(&h[0..2.min(h.len())], 16).unwrap_or(0) as f32 / 255.0;
    let g = u32::from_str_radix(h.get(2..4).unwrap_or("00"), 16).unwrap_or(0) as f32 / 255.0;
    let b = u32::from_str_radix(h.get(4..6).unwrap_or("00"), 16).unwrap_or(0) as f32 / 255.0;
    slint::Color::from_rgb_f32(r, g, b)
}

/// Open a model's detail page: fetch files + build the quant table (async).
fn open_detail(weak: slint::Weak<Main>, repo: &str, _chat_model_name: &Arc<RwLock<String>>) {
    let repo = repo.to_string();
    thread::spawn(move || {
        let cfg = config::cfg();
        let hw = hardware::Hardware::detect();
        let model = model::Model {
            repo_id: repo.clone(),
            name: repo.clone(),
            provider: repo.split('/').next().unwrap_or("").to_string(),
            n_params_b: 8.0,
            context: cfg.context_default,
            license: String::new(),
            commercial: true,
            uses: vec![],
            description: String::new(),
            arch: String::new(),
            release: String::new(),
            superseded: false,
            successor_id: String::new(),
        };

        let s_in = cfg.context_default;
        let s_out = 256u32;
        let quants = perf::quants_for_model(&model);
        let rec_i = perf::recommend_quant(&quants, &hw).unwrap_or(0);
        let qrows: Vec<QuantRow> = quants
            .iter()
            .map(|q| {
                let est = perf::estimate(&model, q, s_in, s_out, &hw, &cfg);
                QuantRow {
                    quant: q.name.clone().into(),
                    bits: format!("{:.0}", q.bits).into(),
                    vram: format!("{:.1} GB", q.vram_gb).into(),
                    quality: q.quality_str().into(),
                    status: format!("{} · ~{:.0} t/s", est.status_label(), est.tok_s).into(),
                    status_color: hex_color(est.status.color()),
                }
            })
            .collect();

        let files: Vec<FileRow> = block_on(hf::files(&repo))
            .unwrap_or_default()
            .into_iter()
            .map(|f| FileRow {
                name: f.rfilename.clone().into(),
                size: format_bytes(f.size).into(),
            })
            .collect();

        let _ = slint::invoke_from_event_loop(move || {
            let Some(w) = weak.upgrade() else { return };
            w.set_detail_repo(repo.clone().into());
            w.set_detail_provider(model.provider.clone().into());
            w.set_detail_desc(model.description.clone().into());
            w.set_detail_rec_quant(quants[rec_i].name.clone().into());
            w.set_quant_table(slint::ModelRc::new(slint::VecModel::from(qrows)));
            w.set_file_list(slint::ModelRc::new(slint::VecModel::from(files)));
            w.set_detail_visible(true);
        });
    });
}

fn format_bytes(b: f64) -> String {
    if b < 1024.0 {
        format!("{:.0} B", b)
    } else if b < 1024.0 * 1024.0 {
        format!("{:.1} KB", b / 1024.0)
    } else if b < 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} MB", b / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GB", b / (1024.0 * 1024.0 * 1024.0))
    }
}

/// Download a model file through the fastest mirror.
fn download_model_file(repo_id: &str, quant: &str) {
    let cfg = config::cfg();
    let dest_dir = config::models_dir().join(repo_id.replace('/', "-"));
    let _ = std::fs::create_dir_all(&dest_dir);
    let filename = format!("{repo_id}__{quant}.gguf");
    let link = download::user_download_link(repo_id, &filename);
    let job = download::DownloadJob::new(link, dest_dir.to_string_lossy().as_ref(), &filename);
    tracing::info!(
        "downloading `{repo_id}` ({quant}) via {} ({} threads)",
        cfg.hf_endpoint,
        cfg.download_threads
    );
    let _ = block_on(download::download(&job, cfg.download_threads));
}

/// Render chat history into the `chat` model property.
fn render_chat(history: &[(String, String)]) -> slint::ModelRc<ChatMsg> {
    let rows: Vec<ChatMsg> = history
        .iter()
        .map(|(role, content)| ChatMsg {
            role: role.clone().into(),
            content: content.clone().into(),
        })
        .collect();
    slint::ModelRc::new(slint::VecModel::from(rows))
}

/// Run a future on a fresh current-thread tokio runtime and block.
fn block_on<T>(fut: impl Future<Output = T>) -> T {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");
    rt.block_on(fut)
}
