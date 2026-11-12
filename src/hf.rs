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
    let repo_id = effective_model_id(m);
    let (n_params_b, arch, context, uses, provider) = parse_model_meta(m, &repo_id);
    Model {
        repo_id: repo_id.clone(),
        name: repo_id,
        provider,
        n_params_b,
        context,
        license: m.license.clone(),
        commercial: m.license.to_lowercase().contains("mit")
            || m.license.to_lowercase().contains("apache"),
        uses,
        description: m.pipeline_tag.clone(),
        arch,
        release: String::new(),
        superseded: false,
        successor_id: String::new(),
    }
}

/// HF Hub payloads carry the repo slug as both `modelId` and a bare `id`; use
/// whichever is present.
fn effective_model_id(m: &HfModel) -> String {
    if !m.model_id.is_empty() {
        m.model_id.clone()
    } else {
        m.hub_id.clone()
    }
}

fn parse_model_meta(m: &HfModel, repo_id: &str) -> (f64, String, u32, Vec<String>, String) {
    // provider = first path segment
    let provider = repo_id.split('/').next().unwrap_or("").to_string();
    let id_lower = repo_id.to_lowercase();
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The HF Hub API returns camelCase keys (`modelId`, `createdAt`). This
    /// guards the deserialization that feeds the GUI: a real-looking item must
    /// yield a non-empty repo/provider/param-count. (Regression test for the
    /// "blank model names" bug where only the snake_case `model_id` was mapped.)
    #[test]
    fn deserializes_camelcase_hub_item() {
        let json = r#"{
            "_id": "621ffdc1",
            "id": "sentence-transformers/all-MiniLM-L6-v2",
            "modelId": "sentence-transformers/all-MiniLM-L6-v2",
            "likes": 6096,
            "downloads": 175899798,
            "pipeline_tag": "text-embedding",
            "createdAt": "2022-03-02T00:00:00Z"
        }"#;
        let v: Value = serde_json::from_str(json).unwrap();
        let hm: HfModel = serde_json::from_value(v).unwrap();
        assert_eq!(hm.model_id, "sentence-transformers/all-MiniLM-L6-v2");

        let m = to_model(&hm);
        assert!(!m.repo_id.is_empty());
        assert_eq!(m.provider, "sentence-transformers");
        assert!(!m.name.is_empty());
    }

    #[test]
    fn deserializes_with_only_id_key() {
        // Some responses only carry `id`. The alias must still fill model_id.
        let json = r#"{"id": "meta-llama/Llama-3-8B-Instruct"}"#;
        let hm: HfModel = serde_json::from_str::<HfModel>(json).expect("parse");
        let m = to_model(&hm);
        assert_eq!(m.repo_id, "meta-llama/Llama-3-8B-Instruct");
        assert_eq!(m.provider, "meta-llama");
        assert!(
            (m.n_params_b - 8.0).abs() < 1e-6,
            "8B guess, got {}",
            m.n_params_b
        );
    }
}
