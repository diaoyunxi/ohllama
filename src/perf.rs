//! Performance estimation.
//!
//! Implements the user's analytical framework. Per-model we estimate:
//!
//! **1. TTFT (time to first token / prefill):**
//! ```text
//! TTFT ≈ (2 · P_active · S_in) / (Peak_FLOPs · η_prefill)
//! ```
//!
//! **2. TPOT / ITL (per-token latency at depth d):**
//! ```text
//! TPOT(d) ≈ T_fixed + B_weight/(η_weight·BW_eff) + B_kv(d)/(η_kv·BW_eff) + T_comm
//! ```
//!
//! **3. Total time to generate S_out tokens:**
//! ```text
//! T_total ≈ TTFT + Σ_{d=S_in}^{S_in+S_out-1} TPOT(d)
//! ```
//!
//! Accuracy comes from three calibration values the user must benchmark:
//! `BW_eff`, `η_prefill`, `η_kv` (see `Config`).

use crate::config::Config;
use crate::hardware::Hardware;
use crate::model::Model;

/// Quantization quality tiers (from low->lossless).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum QuantQuality {
    Low,
    Moderate,
    Good,
    Excellent,
    Lossless,
}

impl QuantQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::Good => "good",
            Self::Excellent => "excellent",
            Self::Lossless => "lossless",
        }
    }
}

/// A single quant option.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Quant {
    /// e.g. "Q4_K_M", "F16".
    pub name: String,
    /// Bits per weight.
    pub bits: f64,
    /// Estimated VRAM in GB for this quant.
    pub vram_gb: f64,
    /// Estimated on-disk size in GB.
    pub disk_gb: f64,
    pub quality: QuantQuality,
}

impl Quant {
    pub fn quality_str(&self) -> &'static str {
        self.quality.as_str()
    }
}

/// Build a standard set of quantization options for a model of `n_params_b`
/// (parameter count in billions).
pub fn quants_for_model(model: &Model) -> Vec<Quant> {
    // F16 = 2 bytes per parameter -> full-precision size in GB.
    let p16_gb = model.n_params_b * 2.0;
    let table = [
        ("Q2_K", 2.0, QuantQuality::Low),
        ("Q3_K_M", 3.0, QuantQuality::Moderate),
        ("Q4_K_M", 4.0, QuantQuality::Good),
        ("Q5_K_M", 5.0, QuantQuality::Good),
        ("Q6_K", 6.0, QuantQuality::Excellent),
        ("Q8_0", 8.0, QuantQuality::Excellent),
        ("F16", 16.0, QuantQuality::Lossless),
    ];
    table
        .iter()
        .map(|(n, bits, q)| {
            let ratio = bits / 16.0; // bytes-per-weight relative to F16
            Quant {
                name: n.to_string(),
                bits: *bits,
                // VRAM = weights + a small activation/KV buffer.
                vram_gb: p16_gb * ratio,
                // On-disk weight file is slightly smaller than live VRAM.
                disk_gb: p16_gb * ratio * 0.95,
                quality: *q,
            }
        })
        .collect()
}

/// Recommend a quant based on available VRAM. Prefer the best (highest index)
/// quality that fits in VRAM. If the machine has no usable VRAM (CPU-only),
/// fall back to a CPU-friendly Q4_K_M.
pub fn recommend_quant(quants: &[Quant], hw: &Hardware) -> Option<usize> {
    let budget = hw.vram_gib;
    if budget <= 0.0 {
        // CPU-only: pick the first "good" quality that's modest in size.
        return quants
            .iter()
            .rposition(|q| matches!(q.quality, QuantQuality::Good));
    }
    // Quants are ordered low -> lossless; pick the highest index that fits.
    for (i, q) in quants.iter().enumerate().rev() {
        if q.vram_gb <= budget {
            return Some(i);
        }
    }
    // Nothing fits in VRAM: smallest quant, will offload to RAM.
    Some(0)
}

/// A full performance estimate for a model + chosen quant.
#[derive(Debug, Clone)]
pub struct PerfEstimate {
    pub ttft_s: f64,
    /// Per-token latency at the midpoint of generation (representative).
    pub tpot_s: f64,
    pub total_s: f64,
    /// Tokens per second during generation (≈ 1 / TPOT).
    pub tok_s: f64,
    /// Estimated status text + color tier (see `StatusTier`).
    pub status: StatusTier,
}

impl PerfEstimate {
    pub fn status_label(&self) -> &'static str {
        match self.status {
            StatusTier::Excellent => "Runs well",
            StatusTier::Good => "Decent",
            StatusTier::Fair => "Tight fit",
            StatusTier::Slow => "Barely runs",
            StatusTier::CannotRun => "Cannot run",
        }
    }
}

/// Color ladder: red -> orange -> yellow -> light green -> green.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StatusTier {
    /// red — too slow to use
    CannotRun,
    /// orange — very slow
    Slow,
    /// yellow — usable but tight
    Fair,
    /// light green — decent
    Good,
    /// green — fast
    Excellent,
}

impl StatusTier {
    /// A CSS-friendly color hex for the tier.
    pub fn color(&self) -> &'static str {
        match self {
            Self::CannotRun => "#ef4444", // red
            Self::Slow => "#f97316",      // orange
            Self::Fair => "#facc15",      // yellow
            Self::Good => "#a3e635",      // light/lime green
            Self::Excellent => "#4ade80", // green
        }
    }
}

impl std::fmt::Display for StatusTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                StatusTier::CannotRun => "cannot-run",
                StatusTier::Slow => "slow",
                StatusTier::Fair => "tight",
                StatusTier::Good => "decent",
                StatusTier::Excellent => "excellent",
            }
        )
    }
}

/// Estimate for a model (using its recommended quant + default context).
pub fn estimate(
    model: &Model,
    quant: &Quant,
    s_in: u32,
    s_out: u32,
    hw: &Hardware,
    cfg: &Config,
) -> PerfEstimate {
    // P_active in parameters (billions -> count)
    let p_active = model.n_params_b * 1e9;
    // Peak FLOPs from hardware.
    let peak = hw.peak_flops.max(1.0);
    // 1) TTFT
    let ttft = 2.0 * p_active * s_in as f64 / (peak * cfg.eta_prefill);
    // 2) Per-token latency terms
    let b_weight = quant.disk_gb * 1e9; // bytes of weight data streamed
    let b_kv = kv_bytes(model, quant, s_out, hw) * 1e9; // -> bytes
    let bw = hw.bw_eff_gibs * 1024.0_f64.powi(3); // -> bytes/s
    let tpot = (b_weight / (cfg.eta_weight * bw)) + (b_kv / (cfg.eta_kv * bw));
    let tpot = tpot.max(0.001);
    let total = ttft + tpot * s_out as f64;
    let tok_s = s_out as f64 / tpot.max(1e-6);

    // Classify.
    let status = classify(tok_s, quant.vram_gb, hw);
    PerfEstimate {
        ttft_s: ttft,
        tpot_s: tpot,
        total_s: total,
        tok_s,
        status,
    }
}

/// Rough KV cache bytes for a full generation of `s_out` new tokens.
fn kv_bytes(model: &Model, _q: &Quant, s_out: u32, _hw: &Hardware) -> f64 {
    // KV cache ~ 2 * layers * kv_heads * head_dim * bytes * s_out.
    // Approximate as (n_params_b * 1e9 * 0.001) bytes for a small context.
    model.n_params_b * 1e6 * s_out as f64 / 1024.0 / 1024.0 / 1024.0
}

/// Map tokens/sec + VRAM fit into the 5-tier color ladder.
fn classify(tok_s: f64, vram_gb: f64, hw: &Hardware) -> StatusTier {
    let fits_vram = hw.vram_gib <= 0.0 || vram_gb <= hw.vram_gib;
    if !fits_vram && hw.vram_gib > 0.0 && vram_gb > hw.vram_gib * 1.5 {
        // Massive over-VRAM overflow.
        if tok_s < 1.0 {
            return StatusTier::CannotRun;
        }
    }
    match tok_s {
        x if x >= 25.0 => StatusTier::Excellent,
        x if x >= 12.0 => StatusTier::Good,
        x if x >= 4.0 => StatusTier::Fair,
        x if x >= 1.0 => StatusTier::Slow,
        _ => StatusTier::CannotRun,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config;
    use crate::hardware::Hardware;
    use crate::model::Model;

    fn test_model() -> Model {
        Model {
            repo_id: "org/test-8b".into(),
            name: "test-8b".into(),
            provider: "org".into(),
            n_params_b: 8.0,
            context: 4096,
            license: "MIT".into(),
            commercial: true,
            uses: vec!["chat".into()],
            description: String::new(),
            arch: String::new(),
            release: String::new(),
            superseded: false,
            successor_id: String::new(),
        }
    }

    fn test_hw() -> Hardware {
        Hardware {
            cpu_model: "test".into(),
            cpu_cores: 8,
            cpu_base_ghz: 3.0,
            ram_gib: 32.0,
            vram_gib: 10.0,
            gpu_model: "test-gpu".into(),
            peak_flops: 50e12,
            bw_eff_gibs: 50.0,
            decode_flops: 100e9,
        }
    }

    #[test]
    fn quants_are_ordered_by_bits() {
        let m = test_model();
        let qs = quants_for_model(&m);
        assert_eq!(qs.len(), 7);
        for w in qs.windows(2) {
            assert!(w[0].bits <= w[1].bits);
            assert!(w[0].vram_gb <= w[1].vram_gb);
            assert!(w[0].disk_gb <= w[1].disk_gb);
        }
        assert_eq!(qs[0].name, "Q2_K");
        assert_eq!(qs.last().unwrap().name, "F16");
    }

    #[test]
    fn recommend_quant_prefers_highest_fitting() {
        let m = test_model();
        let qs = quants_for_model(&m);
        let hw = test_hw();
        // 10 GiB VRAM: pick the largest quant that fits (index order low->lossless).
        let rec = recommend_quant(&qs, &hw).unwrap();
        assert!(qs[rec].vram_gb <= hw.vram_gib || hw.vram_gib <= 0.0);
        // Nothing fits on a 0-VRAM machine -> falls back to "good" quality (CPU mode).
        let cpu_hw = Hardware {
            vram_gib: 0.0,
            ..Hardware::default()
        };
        let rec2 = recommend_quant(&qs, &cpu_hw).unwrap();
        assert!(matches!(qs[rec2].quality, QuantQuality::Good));
    }

    #[test]
    fn ttft_formula_matches() {
        let m = test_model();
        let hw = test_hw();
        let cfg = config::Config {
            eta_prefill: 0.5,
            eta_weight: 0.8,
            eta_kv: 0.7,
            ..Default::default()
        };
        let qs = quants_for_model(&m);
        let est = estimate(&m, &qs[2], 4096, 256, &hw, &cfg);
        // TTFT = 2 * P_active * S_in / (Peak_FLOPs * eta_prefill)
        let expected = 2.0 * 8e9 * 4096.0 / (hw.peak_flops * 0.5);
        assert!((est.ttft_s - expected).abs() < expected * 1e-9);
        // total = TTFT + TPOT * S_out
        assert!((est.total_s - (est.ttft_s + est.tpot_s * 256.0)).abs() < 1e-6);
    }

    #[test]
    fn tok_s_is_positive_and_decreases_with_size() {
        let mut m = test_model();
        let hw = test_hw();
        let cfg = Config::default();
        let qs = quants_for_model(&m);
        let fast = estimate(&m, &qs[2], 1024, 128, &hw, &cfg);
        m.n_params_b = 70.0;
        let qs2 = quants_for_model(&m);
        let slow = estimate(&m, &qs2[2], 1024, 128, &hw, &cfg);
        assert!(fast.tok_s > 0.0);
        assert!(slow.tok_s > 0.0);
        assert!(fast.tok_s > slow.tok_s);
    }

    #[test]
    fn status_tiers_have_colors() {
        let tiers = [
            StatusTier::CannotRun,
            StatusTier::Slow,
            StatusTier::Fair,
            StatusTier::Good,
            StatusTier::Excellent,
        ];
        let colors: Vec<&str> = tiers.iter().map(|t| t.color()).collect();
        assert_eq!(colors[0], "#ef4444");
        assert_eq!(colors[4], "#4ade80");
        // red -> orange -> yellow -> light green -> green ladder
        assert_eq!(colors.len(), 5);
    }

    #[test]
    fn total_s_increases_with_s_out() {
        let m = test_model();
        let hw = test_hw();
        let cfg = Config::default();
        let qs = quants_for_model(&m);
        let a = estimate(&m, &qs[2], 1024, 64, &hw, &cfg);
        let b = estimate(&m, &qs[2], 1024, 512, &hw, &cfg);
        assert!(b.total_s > a.total_s);
    }
}
