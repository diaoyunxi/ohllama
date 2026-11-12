//! Downloading: pick the fastest mirror and fetch model/engine files.
//!
//! Strategy:
//! - Probe every configured mirror with a small HEAD (or a ranged GET) on a
//!   lightweight object, measure latency + throughput, pick the winner.
//! - Download a model file with up to `DOWNLOAD_THREADS` parallel ranged GETs.
//! - If the user prefers to handle downloads themselves, this module exposes
//!   just the (mirror, url) pair so the GUI can hand it off.

use crate::config;
use std::time::Instant;

/// A download job: the file to fetch and where to put it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DownloadJob {
    pub url: String,
    pub dest: String,
    pub filename: String,
}

impl DownloadJob {
    pub fn new(url: String, dest_dir: &str, filename: &str) -> Self {
        Self {
            url,
            dest: format!("{}/{}", dest_dir, filename),
            filename: filename.to_string(),
        }
    }
}

/// Which mirror is the fastest right now.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MirrorProbe {
    pub base: String,
    /// Measured latency of a tiny ranged GET (ms).
    pub latency_ms: f64,
    pub ok: bool,
}

/// Probe each mirror and return them sorted fastest-first.
pub async fn probe_mirrors() -> Vec<MirrorProbe> {
    let cfg = config::cfg();
    let client = reqwest::Client::new();
    // Probe a small file that exists on every HF mirror (a known GGUF).
    const PROBE_PATH: &str =
        "/TheBloke/TinyLlama-1.1B-Chat-v0.3-GGUF/resolve/main/00000-ggml-00001.safetensors?download=true";
    let mut out: Vec<MirrorProbe> = Vec::new();
    let mut handles = Vec::new();
    for base in cfg.mirrors.clone() {
        let client = client.clone();
        let base = base.trim_end_matches('/').to_string();
        let url = format!("{}{}", base, PROBE_PATH);
        handles.push(tokio::spawn(async move {
            let t0 = Instant::now();
            let r = client
                .get(&url)
                .header("Range", "bytes=0-4095")
                .send()
                .await;
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            let ok = match r {
                Ok(resp) => {
                    let s = resp.status();
                    s.is_success() || s.as_u16() == 206
                }
                Err(_) => false,
            };
            MirrorProbe {
                base,
                latency_ms: elapsed,
                ok,
            }
        }));
    }
    for h in handles {
        if let Ok(p) = h.await {
            out.push(p);
        }
    }
    out.sort_by(|a, b| {
        // ok mirrors first, then by latency
        b.ok.cmp(&a.ok).then_with(|| {
            a.latency_ms
                .partial_cmp(&b.latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    out
}

/// Fastest mirror base URL (or None if none reachable).
pub async fn fastest_mirror() -> Option<String> {
    probe_mirrors()
        .await
        .into_iter()
        .find(|m| m.ok)
        .map(|m| m.base)
}

/// Download `job` using the fastest mirror (rewriting the host) with up to
/// `threads` parallel ranged GETs. Streams to `job.dest`.
pub async fn download(job: &DownloadJob, threads: usize) -> Result<(), anyhow::Error> {
    use std::io::Write;
    let mirror = fastest_mirror()
        .await
        .unwrap_or_else(|| config::cfg().hf_endpoint.trim_end_matches('/').to_string());
    let url = rewrite_host(&job.url, &mirror);
    let client = reqwest::Client::new();

    // 1) full range to learn the length.
    let len = client
        .get(&url)
        .header("Range", "bytes=0-0")
        .send()
        .await?
        .headers()
        .get("content-range")
        .and_then(|v| {
            let s: &str = v.to_str().ok()?;
            let total = s.rsplit('/').next()?;
            total.parse().ok()
        })
        .unwrap_or(0);

    let file = tokio::fs::File::create(&job.dest).await?;
    let threads = threads.clamp(1, 64);

    if len == 0 {
        // No ranged support: single-shot download.
        let bytes = client.get(&url).send().await?.bytes().await?;
        use tokio::io::AsyncWriteExt;
        let mut f = file;
        f.write_all(&bytes).await?;
        f.flush().await?;
        return Ok(());
    }

    // 2) chunked parallel download.
    let chunk = (len as usize + threads) / threads;
    let mut handles = Vec::new();
    for i in 0..threads.min(len as usize) {
        let start = i * chunk;
        let end = if i == threads.min(len as usize) - 1 {
            len - 1
        } else {
            (start + chunk - 1).min(len - 1)
        };
        let client = client.clone();
        let url = url.clone();
        let dest = job.dest.clone();
        handles.push(tokio::spawn(async move {
            let resp = client
                .get(&url)
                .header("Range", format!("bytes={}-{}", start, end))
                .send()
                .await?;
            let body = resp.bytes().await?;
            let mut f = std::fs::OpenOptions::new().write(true).open(&dest)?;
            use std::io::Seek;
            use std::io::SeekFrom;
            f.seek(SeekFrom::Start(start as u64))?;
            f.write_all(&body)?;
            Ok::<(), anyhow::Error>(())
        }));
    }
    for h in handles {
        h.await??;
    }
    Ok(())
}

/// Rewrite a file URL's host to point at `mirror_base`, keeping the repo path.
/// HuggingFace layout is `{base}/{repo_id}/resolve/{branch}/{file}` — so we
/// preserve everything after the original host.
pub fn rewrite_host(url: &str, mirror_base: &str) -> String {
    // Extract the path portion (after scheme://host/); a bare path passes through.
    let path = url
        .split_once("://")
        .and_then(|(_, rest)| rest.split_once('/').map(|(_, p)| p))
        .unwrap_or(url);
    format!("{}/{}", mirror_base.trim_end_matches('/'), path)
}

/// Human-readable byte count.
pub fn human_gib(g: f64) -> String {
    format!("{:.1} GB", g)
}

/// The "hand off to user" variant: return the (mirror, url) for a model file
/// so the user can download it themselves.
pub fn user_download_link(repo_id: &str, filename: &str) -> String {
    let cfg = config::cfg();
    let base = cfg.hf_endpoint.trim_end_matches('/');
    format!("{}/{}/resolve/main/{}", base, repo_id, filename)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrite_host_uses_resolve_suffix() {
        let u = "https://huggingface.co/org/model/resolve/main/w.bin";
        let out = rewrite_host(u, "https://hf-mirror.com");
        assert!(out.starts_with("https://hf-mirror.com/"), "got: {out}");
        assert!(out.contains("resolve/main/w.bin"));
        assert!(out.contains("org/model/"));
    }

    #[test]
    fn rewrite_host_accepts_bare_suffix() {
        let out = rewrite_host("org/model/resolve/main/w.bin", "https://hf-mirror.com/");
        assert!(out.starts_with("https://hf-mirror.com/org/model/resolve/main/w.bin"));
    }

    #[test]
    fn user_download_link_shape() {
        let link = user_download_link("org/model", "w.bin");
        assert!(link.contains("org/model"));
        assert!(link.contains("resolve/main/w.bin"));
    }

    #[test]
    fn download_job_builds_dest_path() {
        let job = DownloadJob::new("u".into(), "/tmp/ohllama", "w.bin");
        assert_eq!(job.dest, "/tmp/ohllama/w.bin");
        assert_eq!(job.filename, "w.bin");
    }

    #[test]
    fn human_gib_formats() {
        assert_eq!(human_gib(6.6), "6.6 GB");
        assert_eq!(human_gib(12.0), "12.0 GB");
    }
}
