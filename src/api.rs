//! HTTP API layer started by `ohllama serve`.
//!
//! Endpoints:
//! - `GET  /health`
//! - `GET  /engines`             — detected engines + default + recommendation
//! - `GET  /models`             — registered models
//! - `POST /models`             — create (body: `{dir, name?, backend?}`)
//! - `DELETE /models/{name}`    — rm
//! - `GET  /models/{name}/run`  — start
//! - `GET  /models/{name}/stop` — stop
//! - `GET  /hf?cat=coding`      — HuggingFace category fetch
//! - `GET  /perf?repo=...&quant=Q4_K_M` — perf estimate

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{delete, get};
use axum::{Json, Router};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;

use crate::engine;
use crate::hardware;
use crate::hf;
use crate::model;
use crate::perf;
use crate::registry;

#[derive(Clone)]
pub struct AppState {
    pub hw: Arc<hardware::Hardware>,
}

/// Start the API on `port`. Blocks until the process is killed.
pub fn serve(port: u16) -> anyhow::Result<()> {
    let hw = Arc::new(hardware::Hardware::detect());
    let state = AppState { hw };
    let app = Router::new()
        .route("/health", get(health))
        .route("/engines", get(list_engines))
        .route("/models", get(list_models).post(create_model))
        .route("/models/{name}", delete(rm_model))
        .route("/models/{name}/run", get(run_model))
        .route("/models/{name}/stop", get(stop_model))
        .route("/hf", get(hf_category))
        .route("/perf", get(perf_estimate))
        .with_state(state.clone());

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let addr: SocketAddr = format!("127.0.0.1:{port}").parse()?;
        println!(
            "ohllama API listening on http://{addr}  (Ctrl+C to stop; models: `ohllama run <name>`)"
        );
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

async fn health() -> &'static str {
    "ok"
}

async fn list_engines(State(st): State<AppState>) -> Json<serde_json::Value> {
    let engines = engine::detect();
    let default = engines
        .iter()
        .find(|e| e.is_default)
        .map(|e| e.name.clone());
    let rec = engine::recommend(&st.hw);
    let hardware_value = serde_json::to_value(&*st.hw).unwrap_or_default();
    Json(serde_json::json!({
        "engines": engines,
        "default": default,
        "recommended": { "engine": rec.0, "backend": rec.1 },
        "hardware": hardware_value,
    }))
}

async fn list_models() -> Json<Vec<registry::RegisteredModel>> {
    Json(registry::list())
}

#[derive(Deserialize)]
struct CreateBody {
    dir: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    backend: Option<String>,
}

async fn create_model(
    Json(body): Json<CreateBody>,
) -> Result<Json<registry::RegisteredModel>, (StatusCode, String)> {
    let dir = body.dir.unwrap_or_else(|| {
        std::env::current_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| ".".into())
    });
    let mut reg = registry::create(&dir).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    if let Some(n) = body.name {
        reg.name = n;
    }
    if let Some(b) = body.backend {
        reg.backend = b;
    }
    Ok(Json(reg))
}

async fn rm_model(Path(name): Path<String>) -> Result<&'static str, (StatusCode, String)> {
    registry::remove(&name)
        .map(|_| "ok")
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))
}

async fn run_model(
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    crate::runtime::spawn_model(&name)
        .map(|r| Json(serde_json::json!({ "name": name, "pid": r.pid, "engine": r.engine, "port": r.port })))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

async fn stop_model(Path(name): Path<String>) -> Result<&'static str, (StatusCode, String)> {
    crate::runtime::stop_model(&name)
        .map(|_| "ok")
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))
}

#[derive(Deserialize)]
struct HfQuery {
    #[serde(default)]
    cat: Option<String>,
}

async fn hf_category(
    Query(q): Query<HfQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cat = q.cat.unwrap_or_else(|| "hot".into());
    match hf::fetch_category(&cat).await {
        Ok(r) => Ok(Json(serde_json::json!(r.models))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
    }
}

#[derive(Deserialize)]
struct PerfQuery {
    #[serde(default)]
    repo: String,
    #[serde(default)]
    quant: String,
    #[serde(default)]
    s_in: Option<u32>,
    #[serde(default)]
    s_out: Option<u32>,
}

async fn perf_estimate(
    Query(q): Query<PerfQuery>,
    State(st): State<AppState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cfg = crate::config::cfg();
    let m = model::Model {
        repo_id: q.repo.clone(),
        name: q.repo.clone(),
        provider: q.repo.split('/').next().unwrap_or("").to_string(),
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
    let quants = perf::quants_for_model(&m);
    let qi = quants.iter().position(|k| k.name == q.quant).unwrap_or(2);
    let s_in = q.s_in.unwrap_or(cfg.context_default);
    let s_out = q.s_out.unwrap_or(256);
    let est = perf::estimate(&m, &quants[qi], s_in, s_out, &st.hw, &cfg);
    Ok(Json(serde_json::json!({
        "model": q.repo,
        "quant": q.quant,
        "ttft_s": est.ttft_s,
        "tpot_s": est.tpot_s,
        "total_s": est.total_s,
        "tok_s": est.tok_s,
        "status": est.status.to_string(),
        "status_color": est.status.color(),
    })))
}
