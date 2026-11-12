//! Model types + HuggingFace-facing data (fetched via `hf.rs`).

use serde::{Deserialize, Serialize};

/// A model as understood by the perf + registry layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// HuggingFace `repo_id`, e.g. "meta-llama/Llama-3-8B-Instruct".
    pub repo_id: String,
    /// Display name (usually the repo_id, or a friendly alias).
    pub name: String,
    /// Publisher / company slug ("Z.ai", "meta", "Qwen", ...).
    pub provider: String,
    /// Parameter count in billions (e.g. 8.0 for an 8B model).
    pub n_params_b: f64,
    /// Context length advertised by the model.
    pub context: u32,
    /// License, if known.
    pub license: String,
    /// Whether the model is open-source / usable locally.
    pub commercial: bool,
    /// Usage class: "chat" | "coding" | "image" | "video" | "tiny" | "hot".
    pub uses: Vec<String>,
    /// A short human description.
    pub description: String,
    /// Model family / arch (optional).
    pub arch: String,
    /// Release month ("2026-01"), optional.
    pub release: String,
    /// Whether the model has a successor (i.e. is superseded).
    pub superseded: bool,
    pub successor_id: String,
}

impl Model {
    pub fn display_name(&self) -> String {
        if self.name.is_empty() {
            self.repo_id.clone()
        } else {
            self.name.clone()
        }
    }
}

/// A HuggingFace model listing (subset of fields the API returns).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HfModel {
    #[serde(default)]
    pub model_id: String,
    #[serde(default, rename = "likes", alias = "likes_count")]
    pub likes: u64,
    #[serde(default, rename = "downloads", alias = "downloads_count")]
    pub downloads: u64,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default, rename = "pipeline_tag")]
    pub pipeline_tag: String,
    #[serde(default)]
    pub license: String,
    /// Created / updated timestamp (RFC-ish).
    #[serde(default)]
    pub created_at: String,
}

/// A file under a model repo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfFile {
    pub rfilename: String,
    pub size: f64,
}

/// Categories shown on the Models page.
pub const CATEGORIES: &[&str] = &["hot", "coding", "image", "video", "tiny"];

/// A "category" result for the models page.
#[derive(Debug, Clone, Default)]
pub struct CategoryResult {
    pub category: String,
    pub models: Vec<Model>,
}
