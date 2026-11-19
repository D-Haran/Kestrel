# Kestrel benchmark audit

Measured on **Apple M2 Pro** (10-core: 6P+4E), **16 GB** RAM, macOS 26.6, Apple clang 17.
Build: `Release`, `-O3 -DNDEBUG`, `KESTREL_METAL=ON`. Model: `tinystories-llama-15m`
(15.19 M params, F32), batch 1. 1 warm-up + 5 timed runs; load/tokenization excluded.

## Throughput: baseline → after optimizations

| Backend | Prompt | Prefill tok/s (base → now) | Decode tok/s (base → now) |
|---|---|---|---|
| CPU   | 32  | 37.9 → **211** | 37.4 → 38.1 |
| CPU   | 128 | 37.6 → **174** | 36.9 → 36.8 |
| Metal | 32  | 22.9 → **244** | 21.8 → **217** |
| Metal | 128 | 28.8 → **226** | 26.2 → **196** |

After the fixes, **Metal is 5.3–6.5× CPU** on decode (was 0.6×), and CPU prefill is ~5× its
old self (and now faster than decode, as batched prefill should be).

## Correctness vs Hugging Face (greedy, 32 new tokens, 50 prompts, fp32 eager)

| Backend | Identical sequences | Max |Δlogit| | Mean |Δlogit| |
|---|---|---|---|
| CPU   | 50 / 50 | 4.6e-05 | 6.1e-06 |
| Metal | 50 / 50 | 4.0e-05 | 6.0e-06 |

Re-verified after every change; zero divergences. Differences are float noise.

## Optimizations applied

1. **Metal resident fused forward (`src/backends/metal.mm`).** The whole step is encoded into
   one command buffer with activations and a GPU-resident KV cache, and a single
   `waitUntilCompleted` at the end — replacing one command buffer + blocking wait per op.
   `waitUntilCompleted` was 91.9% of Metal runtime before; decode went 21.8 → 217 tok/s.
   Used when no trace is requested (tracing keeps the op-by-op path).
2. **Batched prefill (`src/model.cpp` `forward_prefill`, `src/backends/cpu.cpp`).** The prompt
   runs as one matrix-matrix multiply per projection instead of a per-token loop; the CPU
   matmul hoists each weight read across the batch. CPU prefill 37.6 → 211 tok/s.
3. **Llama 3 RoPE scaling (`src/config.cpp`, `src/model.cpp`, both backends).** `rope_scaling`
   with `rope_type: "llama3"` is parsed, and RoPE uses precomputed, wavelength-scaled inverse
   frequencies (HF `_compute_llama3_parameters`). Kestrel now loads Llama 3.2 1B's config;
   scaled logits match HF to ~1e-6 (`tools/verify_llama3_rope.py`).

## Reproduce

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build-run -G Ninja -DCMAKE_BUILD_TYPE=Release -DKESTREL_METAL=ON \
      -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++
cmake --build build-run --target kestrel kestrel_bench kestrel_trace

build-run/kestrel_bench --model models/tinystories-15m --backend metal \
  --prompt-tokens 128 --max-tokens 128 --iterations 5 --warmup 1 --json out.json
.venv/bin/python tools/audit_correctness.py cpu metal     # 50-prompt HF comparison
.venv/bin/python tools/verify_llama3_rope.py              # llama3 RoPE vs HF
```

## Caveats

- **Llama 3.2 1B config loads, but it has not been benchmarked end-to-end** — the 2.5 GB
  weights could not be downloaded (insufficient free disk at audit time). Fix #3 is verified
  by config load + a logit match against HF on a small model with the identical `rope_scaling`.
- For large-context Llama models, pass a small `--context` (e.g. 1024); `max_position_embeddings`
  is 131072 and sizing the KV cache to that would exhaust memory.
- Context cap on TinyStories is 256, so 512-token prompts are not measured.
- llama.cpp is not installed; no external reference point.

Raw data: `benchmarks/results/audit/` — `throughput.csv`, `bench-*.json` (baseline),
`fix1-*.json`, `fix2-*.json` (after each fix), `correctness.json`, `profile-*.json`.
