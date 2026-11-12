//! Hardware detection: CPU, GPU, RAM, and peak-FLOP / effective-BW estimates.
//!
//! The perf model (see `perf.rs`) needs two hardware numbers:
//! - `Peak_FLOPs` (floating point ops per second, used for prefill)
//! - `BW_eff` (effective memory bandwidth, used for decode / weight streaming)
//!
//! These are *estimates* until the user calibrates them with a benchmark.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use sysinfo::System;

/// Detected hardware snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Hardware {
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub cpu_base_ghz: f64,
    /// Total system RAM in GiB.
    pub ram_gib: f64,
    /// Estimated GPU VRAM in GiB (0 if no discrete GPU found).
    pub vram_gib: f64,
    pub gpu_model: String,
    /// Estimated peak FLOPs (ops/s) used for prefill math.
    pub peak_flops: f64,
    /// Estimated effective memory bandwidth (GiB/s) used for decode math.
    pub bw_eff_gibs: f64,
    /// Estimated effective FLOPs for decode, in FLOPs/s (weights are read from RAM).
    pub decode_flops: f64,
}

impl Hardware {
    /// Probe the machine. Cheap — only sysinfo + file reads.
    pub fn detect() -> Self {
        let mut s = System::new_all();
        s.refresh_cpu_all();
        let cpu_count = s.cpus().len() as u32;
        let cpu_model = s
            .cpus()
            .first()
            .map(|c| c.name().to_string())
            .unwrap_or_default();
        // base clock: refresh a couple of times to get a sample
        std::thread::sleep(Duration::from_millis(50));
        s.refresh_cpu_all();
        let base_mhz: u64 = s.cpus().iter().map(|c| c.frequency()).max().unwrap_or(0);
        let base_ghz = base_mhz as f64 / 1000.0;

        let total_mem = s.total_memory() as f64;
        let ram_gib = total_mem / (1024.0 * 1024.0 * 1024.0);

        let (gpu_model, vram_gib) = detect_gpu();

        // Rough peak FLOP estimate: assume ~4 FLOPs per core-GHz for scalar-ish
        // integer/FP8-ish inference, plus a 4x multiplier for FMA. We use a
        // conservative "dense matmul" figure: 2 * FLOPs_per_GHz_per_core * GHz * cores.
        let flops_per_ghz_core = 8.0e9; // ~8 GFLOP/GHz/core at FP16-ish
        let peak_flops = if base_ghz > 0.0 {
            2.0 * flops_per_ghz_core * base_ghz * cpu_count as f64
        } else {
            // fallback: assume 3 GHz
            2.0 * flops_per_ghz_core * 3.0 * cpu_count as f64
        };

        // Estimate effective BW from typical DDR4/DDR5. We can't reliably read the
        // actual DIMM bus, so we estimate from RAM size and CPU generation:
        // - 2-channel DDR4 ~ 50 GB/s, DDR5 ~ 90 GB/s
        // Pick a middle value and let the user calibrate.
        let bw_eff_gibs = estimate_bw(cpu_count);

        // Decode FLOPs are bandwidth-bound: effective bandwidth in FLOPs/s,
        // where each byte read ≈ 1 "FLOP-equivalent" of weight traffic.
        let decode_flops = bw_eff_gibs * 1024.0_f64.powi(3) * 2.0;

        Self {
            cpu_model,
            cpu_cores: cpu_count,
            cpu_base_ghz: base_ghz,
            ram_gib,
            vram_gib,
            gpu_model,
            peak_flops,
            bw_eff_gibs,
            decode_flops,
        }
    }

    /// Which backend is recommended for this hardware?
    /// - Discrete GPU with enough VRAM -> `vllm` / `cuda` backends.
    /// - Apple silicon unified memory -> `metal`.
    /// - Otherwise -> `cpu` (llama.cpp) / `avx2`.
    pub fn recommended_backend(&self) -> String {
        if !self.gpu_model.is_empty() && self.vram_gib >= 6.0 {
            // Discrete GPU: prefer vllm if installed, else cuda
            return "vllm".to_string();
        }
        if cfg!(target_os = "macos") {
            return "metal".to_string();
        }
        if self.cpu_cores >= 8 {
            "cpu-avx2".to_string()
        } else {
            "cpu".to_string()
        }
    }

    /// A short human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{} | {} core(s) @ {:.1} GHz | RAM {:.1} GiB | GPU: {}",
            self.cpu_model,
            self.cpu_cores,
            self.cpu_base_ghz,
            self.ram_gib,
            if self.gpu_model.is_empty() {
                "(none / iGPU)"
            } else {
                &self.gpu_model
            }
        )
    }
}

/// Estimate effective BW (GiB/s) heuristically. User can override in config.
fn estimate_bw(_cores: u32) -> f64 {
    40.0 // ~2ch DDR4-3200; conservative default
}

/// Run `nvidia-smi` (first in PATH, then in the standard Windows install
/// location) and return `(gpu_model_name, vram_gib)`, or `(0, 0)` on failure.
fn run_nvidia_smi() -> (String, f64) {
    // Try the standard Windows install path before falling back to PATH.
    let candidates: &[&str] = if cfg!(target_os = "windows") {
        &[
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            "nvidia-smi",
        ]
    } else {
        &["nvidia-smi"]
    };

    for exe in candidates {
        if let Ok(out) = std::process::Command::new(exe)
            .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            if !s.trim().is_empty() {
                let line = s.lines().next().unwrap_or("").to_string();
                if let Some((name, mem)) = line
                    .split_once(',')
                    .map(|(a, b)| (a.trim().to_string(), b.trim().to_string()))
                {
                    // mem.total is in MiB
                    let gib: f64 = mem
                        .trim()
                        .trim_end_matches(" MiB")
                        .parse::<f64>()
                        .unwrap_or(0.0)
                        / 1024.0;
                    return (name, gib);
                }
            }
        }
    }
    (String::new(), 0.0)
}

/// Try to detect a discrete GPU. Best-effort; returns empty model + 0 VRAM if none.
fn detect_gpu() -> (String, f64) {
    #[cfg(target_os = "windows")]
    {
        run_nvidia_smi()
    }
    #[cfg(target_os = "macos")]
    {
        // Apple silicon unified memory: VRAM = 75 % of system RAM (the default
        // Apple GPU allocation).  Query sysinfo directly to avoid recursing
        // into `detect()`.
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        let ram_gib = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);
        (
            String::from("Apple GPU (unified)"),
            (ram_gib * 0.75).max(8.0),
        )
    }
    #[cfg(target_os = "linux")]
    {
        run_nvidia_smi()
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        (String::new(), 0.0)
    }
}

// (Networks/Networks-free helper removed; hardware detection above is self-contained.)
