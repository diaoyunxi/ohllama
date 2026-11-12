//! Simple local chat client (page 3).
//!
//! Talks to a running local engine's OpenAI-compatible endpoint
//! (`http://127.0.0.1:<port>/v1/chat/completions`). The engine is abstracted
//! away; we just send JSON and read the streamed reply.

use crate::config;
use crate::registry;

/// A chat message.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

/// A chat session (in-memory).
pub struct Chat {
    pub messages: Vec<Msg>,
    pub model: String,
}

impl Chat {
    pub fn new(model: &str) -> Self {
        Self {
            messages: Vec::new(),
            model: model.to_string(),
        }
    }

    pub fn push_user(&mut self, text: &str) {
        self.messages.push(Msg {
            role: "user".into(),
            content: text.into(),
        });
    }

    pub fn push_assistant(&mut self, text: &str) {
        self.messages.push(Msg {
            role: "assistant".into(),
            content: text.into(),
        });
    }

    /// Send the conversation to the running engine.
    pub async fn send(&self) -> Result<String, anyhow::Error> {
        let port = self.engine_port()?;
        let body = serde_json::json!({
            "model": self.model,
            "messages": self.messages,
            "stream": false,
        });
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("engine returned {} ", resp.status());
        }
        let v: serde_json::Value = resp.json().await?;
        Ok(v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string())
    }

    /// Which port the engine for this model listens on.
    fn engine_port(&self) -> Result<u16, anyhow::Error> {
        let reg = registry::list()
            .into_iter()
            .find(|m| m.name == self.model)
            .ok_or_else(|| anyhow::anyhow!("model `{}` is not registered", self.model))?;
        // Reuse the deterministic port logic from runtime.
        let port = 11_000u16 + (stable_hash(&reg.name) % 1_000) as u16;
        let _ = config::cfg();
        Ok(port)
    }
}

fn stable_hash(s: &str) -> u32 {
    let mut h: u32 = 5381;
    for b in s.bytes() {
        h = h.wrapping_mul(33) ^ b as u32;
    }
    h
}

/// Blocking chat send for worker threads (used by the GUI). Takes the
/// conversation history as (role, content) pairs plus the model name.
pub fn send_blocking(history: &[(String, String)], model: &str) -> Result<String, anyhow::Error> {
    let port = model_port(model)?;
    let messages: Vec<Msg> = history
        .iter()
        .map(|(role, content)| Msg {
            role: role.clone(),
            content: content.clone(),
        })
        .collect();
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": false,
    });
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        let resp = client
            .post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("engine returned {}", resp.status());
        }
        let v: serde_json::Value = resp.json().await?;
        Ok(v.get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string())
    })
}

/// Deterministic port for a registered model (mirrors `runtime`).
fn model_port(model: &str) -> Result<u16, anyhow::Error> {
    let reg = registry::list()
        .into_iter()
        .find(|m| m.name == model)
        .ok_or_else(|| anyhow::anyhow!("model `{model}` is not registered"))?;
    Ok(11_000u16 + (stable_hash(&reg.name) % 1_000) as u16)
}
