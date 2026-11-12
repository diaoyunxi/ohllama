//! HuggingFace data access (with mirror fallback).
//!
//! The `hf_endpoint` base is configurable (defaults to `hf-mirror.com`). Every call
//! tries the configured endpoint and transparently falls back to a mirror when it
//! fails, so the GUI keeps working in restricted networks.

use crate::config;
use crate::model::{CategoryResult, HfFile, HfModel, Model, CATEGORIES};
use serde_json::Value;

type Result<T> = std::result::Result<T, anyhow::Error>;

/// Build a reqwest client with an optional HF token.
fn client() -> Result<reqwest::Client> {
    let cfg = config::cfg();
    let mut b = reqwest::Client::builder().timeout(std::time::Duration::from_secs(30));
    if !cfg.hf_token.is_empty() {
        b = b.default_headers({
            let mut h = reqwest::header::HeaderMap::new();
            let v = format!("Bearer {}", cfg.hf_token);
            h.insert(reqwest::header::AUTHORIZATION, v.parse().unwrap());
            h
        });
    }
    Ok(b.build()?)
}

/// Resolve the model to a displayable `Model`.
pub fn to_model(m: &HfModel) -> Model {
    let (n_params_b, arch, context, uses, provider) = parse_model_meta(m);
    Model {
        repo_id: m.model_id.clone(),
        name: m.model_id.clone(),
        provider: provider.clone(),
        n_params_b,
        context,
        license: m.license.clone(),
        commercial: m.license.to_lowercase().contains("mit")
            || m.license.to_lowercase().contains("apache"),
        uses: uses.clone(),
        description: m.pipeline_tag.clone(),
        arch: arch.clone(),
        release: String::new(),
        superseded: false,
        successor_id: String::new(),
    }
}

fn parse_model_meta(m: &HfModel) -> (f64, String, u32, Vec<String>, String) {
    // provider = first path segment
    let provider = m.model_id.split('/').next().unwrap_or("").to_string();
    let id_lower = m.model_id.to_lowercase();
    // Parameter count guess from the name (e.g. "7B", "13B", "8b-instruct").
    let n_params_b = guess_params(&id_lower);
    // Context: if "128k", "4k" appears in id, parse.
    let context = guess_context(&id_lower);
    // Usage classes from tags.
    let mut uses = Vec::new();
    if m.pipeline_tag.contains("text-generation") || id_lower.contains("llm") {
        uses.push("chat".into());
    }
    if m.tags.iter().any(|t| t.to_lowercase().contains("coding")) || id_lower.contains("code") {
        uses.push("coding".into());
    }
    if m.pipeline_tag.contains("text-to-image") {
        uses.push("image".into());
    }
    if m.pipeline_tag.contains("text-to-video") || m.pipeline_tag.contains("image-to-video") {
        uses.push("video".into());
    }
    (n_params_b, String::new(), context, uses, provider)
}

fn guess_params(id: &str) -> f64 {
    let re = regex::Regex::new(r"(\d+(?:\.\d+)?)\s*b\b|(\d+(?:\.\d+)?)b").unwrap();
    if let Some(m) = re.captures(id) {
        if let Some(g) = m.get(1).or_else(|| m.get(2)) {
            return g.as_str().parse::<f64>().unwrap_or(1.0);
        }
    }
    // Fallback by family.
    if id.contains("7b") {
        7.0
    } else if id.contains("13b") {
        13.0
    } else if id.contains("70b") {
        70.0
    } else if id.contains("tiny") || id.contains("3b") || id.contains("1b") {
        3.0
    } else {
        8.0
    }
}

fn guess_context(id: &str) -> u32 {
    let re = regex::Regex::new(r"(\d+)k").unwrap();
    if let Some(m) = re.captures(id) {
        if let Some(g) = m.get(1) {
            return g.as_str().parse::<u32>().unwrap_or(4096) * 1024;
        }
    }
    4096
}

/// Fetch a category's popular models.
pub async fn fetch_category(category: &str) -> Result<CategoryResult> {
    let cfg = config::cfg();
    let base = cfg.hf_endpoint.trim_end_matches('/').to_string();
    let mut results: Vec<Model> = Vec::new();
    let mut endpoint_ok = false;

    let mut candidates: Vec<String> = vec![base.clone()];
    for m in &cfg.mirrors {
        if !candidates.iter().any(|c| c == m) {
            candidates.push(m.trim_end_matches('/').to_string());
        }
    }
    if let Ok(b) = std::env::var("HF_ENDPOINT") {
        candidates.insert(0, b.trim_end_matches('/').to_string());
    }

    let c = client()?;
    for ep in &candidates {
        let url = match category {
            "hot" => format!("{}/api/models?sort=downloads&direction=-1&limit=30", ep),
            "coding" => format!(
                "{}/api/models?search=coding&sort=downloads&direction=-1&limit=30",
                ep
            ),
            "image" => format!(
                "{}/api/models?pipeline_tag=text-to-image&sort=downloads&direction=-1&limit=30",
                ep
            ),
            "video" => format!(
                "{}/api/models?pipeline_tag=text-to-video&sort=downloads&direction=-1&limit=30",
                ep
            ),
            "tiny" => format!(
                "{}/api/models?search=tiny&sort=downloads&direction=-1&limit=30",
                ep
            ),
            _ => continue,
        };
        match c.get(&url).send().await {
            Ok(r) if r.status().is_success() => {
                if let Ok(items) = r.json::<Vec<Value>>().await {
                    for v in items {
                        if let Ok(hm) = serde_json::from_value::<HfModel>(v) {
                            results.push(to_model(&hm));
                        }
                    }
                    endpoint_ok = true;
                    break;
                }
            }
            _ => continue, // try next mirror
        }
    }
    let _ = endpoint_ok;
    Ok(CategoryResult {
        category: category.to_string(),
        models: results,
    })
}

/// Free-text search.
pub async fn search(query: &str) -> Result<Vec<Model>> {
    let cfg = config::cfg();
    let mut candidates: Vec<String> = vec![cfg.hf_endpoint.trim_end_matches('/').to_string()];
    for m in &cfg.mirrors {
        if !candidates.iter().any(|c| c == m.trim_end_matches('/')) {
            candidates.push(m.trim_end_matches('/').to_string());
        }
    }
    let c = client()?;
    let mut results: Vec<Model> = Vec::new();
    for ep in &candidates {
        let url = format!(
            "{}/api/models?search={}&sort=downloads&direction=-1&limit=50",
            ep,
            urlencoding(query)
        );
        if let Ok(r) = c.get(&url).send().await {
            if r.status().is_success() {
                if let Ok(items) = r.json::<Vec<Value>>().await {
                    for v in items {
                        if let Ok(hm) = serde_json::from_value::<HfModel>(v) {
                            results.push(to_model(&hm));
                        }
                    }
                    break;
                }
            }
        }
    }
    Ok(results)
}

/// List the files under a model (Files & versions section of the detail page).
pub async fn files(repo_id: &str) -> Result<Vec<HfFile>> {
    let cfg = config::cfg();
    let mut candidates: Vec<String> = vec![cfg.hf_endpoint.trim_end_matches('/').to_string()];
    for m in &cfg.mirrors {
        if !candidates.iter().any(|c| c == m.trim_end_matches('/')) {
            candidates.push(m.trim_end_matches('/').to_string());
        }
    }
    let c = client()?;
    for ep in &candidates {
        let url = format!("{}/api/models/{}/tree/main", ep, repo_id);
        if let Ok(r) = c.get(&url).send().await {
            if r.status().is_success() {
                let items: Vec<Value> = r.json().await?;
                let out: Vec<HfFile> = items
                    .iter()
                    .filter_map(|v| {
                        Some(HfFile {
                            rfilename: v.get("path")?.as_str()?.to_string(),
                            size: v.get("size").and_then(|s| s.as_f64()).unwrap_or(0.0),
                        })
                    })
                    .collect();
                return Ok(out);
            }
        }
    }
    Ok(Vec::new())
}

fn urlencoding(s: &str) -> String {
    s.replace('%', "%25")
        .replace(' ', "%20")
        .replace('&', "%26")
        .replace('+', "%2B")
}

/// Re-export CATEGORIES for convenience.
pub fn categories() -> &'static [&'static str] {
    CATEGORIES
}
