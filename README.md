# ohllama

A local LLM manager with a **Slint** GUI (Claude-style), a **CLI**, and an **HTTP API** layer.
It detects the inference engines you already have, pulls model catalogs from HuggingFace
(with mirror support), estimates per-model performance on *your* hardware, and abstracts away
the `llama.cpp` / `ollama` / `vLLM` complexity.

> Application layer only. Inference engines themselves are **not** vendored — `ohllama`
> probes for their presence (on `PATH` **and** on disk, so pre-compiled binaries you downloaded
> but didn't add to `PATH` are still found) and spawns them.

## Features

### Page 1 — Engines
- Detects every known inference engine (llama.cpp, ollama, vLLM, LM Studio, ggml) by **command
  and on-disk file**, flagging pre-compiled (not-on-PATH) copies.
- Reads your hardware (CPU cores/clock, RAM, discrete GPU VRAM) and recommends the best
  backend (`cpu` / `cpu-avx2` / `cuda` / `metal` / `vllm`).
- One-click **Download** / **Set default**.

### Page 2 — Models
- Pulls **hot / coding / image / video / tiny** model lists from HuggingFace and free-text
  search, all through a configurable mirror (`hf-mirror.com` by default).
- Each model row shows: name, provider, **recommended quant**, recommended size, and the
  performance triad below.
- **Performance framework** (per model + quant):
  1. **TTFT** ≈ `2·P_active·S_in / (Peak_FLOPs · η_prefill)`
  2. **TPOT(d)** ≈ `T_fixed + B_weight/(η_weight·BW_eff) + B_kv(d)/(η_kv·BW_eff) + T_comm`
  3. **Total** ≈ `TTFT + Σ TPOT(d)`
- Status colors follow a red → orange → yellow → light-green → green ladder tied to the
  estimated tokens/sec and VRAM fit.
- The **detail view** (HuggingFace-style) adds a **quantization table**
  (Quant / Bits / VRAM / Quality / Status) plus **Files & versions**, and a **Download**
  button that picks the **fastest mirror** and fetches with **32/64 threads** — or hands the
  link back to you.

### Page 3 — Chat
- A simple chat page that talks to your running local engine's
  OpenAI-compatible endpoint.

## CLI

```
ohllama serve [--port=N]   # start the HTTP API layer (default 8787)
ohllama create [DIR]       # register a model dir (files are NOT moved;
                           #   tokenizer/config files auto-detected; default = pwd)
ohllama run <MODEL>       # start a registered model on its (or the default) backend
ohllama stop <MODEL>      # stop a running model
ohllama list              # list registered models
ohllama rm <MODEL>        # unregister (files left in place)
ohllama engines           # print detected engines + default + recommendation
ohllama gui               # open the Slint window
ohllama help              # full help
```

`create` never changes model-file location; it auto-finds `tokenizer*`, `config*`,
`vocab*`, and the weight files (`*.safetensors` > `*.gguf` > `*.bin` > `*.onnx`).

## Configuration

Data lives in `~/.ohllama` (override with `OHLLAMA_HOME`):

```
~/.ohllama/
  config.toml     # user settings (see config.example.toml)
  registry.json   # registered models
  models/         # where downloaded model files go
  engines/        # where downloaded engine binaries go
  running.json    # currently-running model PIDs
```

Key settings (all in `config.toml`):

| Key | Default | Meaning |
| --- | --- | --- |
| `hf_endpoint` | `https://hf-mirror.com` | HuggingFace base; switch to `https://huggingface.co` if you don't need a mirror |
| `mirror_candidates` | hf-mirror, huggingface.co, hf.co | mirrors probed to pick the fastest |
| `download_threads` | `32` | parallel ranged-GET threads (32/64) |
| `default_engine` | *(auto)* | force a specific engine backend |
| `calibration_bw_eff` | `40.0` | effective memory bandwidth (GiB/s) — **calibrate** |
| `eta_prefill` / `eta_weight` / `eta_kv` | `0.5` / `0.8` / `0.7` | the three accuracy knobs for the perf model |

`HF_ENDPOINT` and `HF_TOKEN` env vars are honored too.

## Building

```sh
cargo build --release
```

Prerequisites: a Rust toolchain. Slint (via `slint-build`) compiles the UI at build time.

## Running the tests

```sh
cargo test
```

## Repository layout

```
src/
  main.rs        # entry + CLI dispatch
  gui.rs         # Slint GUI driver (3 pages)
  ui/ohllama.slint# the UI (Claude-style)
  config.rs      # config + data-dir layout
  hardware.rs    # CPU/GPU/RAM + peak-FLOP / effective-BW estimates
  engine.rs      # engine detection (command + on-disk) & recommendation
  perf.rs        # TTFT / TPOT / Total + quant tables + status ladder
  model.rs       # Model + HuggingFace data structures
  hf.rs          # HuggingFace fetch with mirror fallback
  download.rs    # fastest-mirror probing + multi-thread download
  registry.rs    # register/list/unregister model directories
  runtime.rs     # spawn/stop engines for a model
  chat.rs        # local OpenAI-compatible chat client
  api.rs         # axum HTTP API layer (serve)
.github/workflows/  # ci.yml + release.yml
```

## License

MIT.
