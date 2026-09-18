# Performance and profiling

## Test environment

The environment and model table below is generated from executed benchmark JSON. GPU core count was separately reported as 16 by `system_profiler SPDisplaysDataType`; it is not inferred from the chip name. Hardware model Mac14,9 is reported as MacBook Pro by `system_profiler`. The checkout has no Git metadata, so `git_commit` is `unknown`; `benchmarks/results/latest.json` records source SHA256 hashes and exact measurement commands.

<!-- BEGIN ENVIRONMENT -->

| Field | Executed value |
|---|---|
| os | macOS 26.6 |
| os_build | 25G5057c |
| machine | Mac14,9 |
| chip | Apple M2 Pro |
| memory_bytes | 17179869184 |
| hw.physicalcpu | 10 |
| hw.logicalcpu | 10 |
| hw.perflevel0.physicalcpu | 6 |
| hw.perflevel1.physicalcpu | 4 |
| timestamp | 2026-09-18T04:31:35Z |
| compiler | AppleClang 17.0.0.17000319 |
| cmake_version | 3.31.6 |
| build_type | Release |
| git_commit | unknown |
| model.attention_heads | 6 |
| model.hidden_size | 288 |
| model.intermediate_size | 768 |
| model.kv_heads | 6 |
| model.layers | 6 |
| model.name | sdobson/tinystories-llama-15m |
| model.path | models/tinystories-15m |
| model.stored_parameter_count | 15191712 |
| model.vocab_size | 32000 |
| model.weight_dtype | F32 |

<!-- END ENVIRONMENT -->

## Methodology

CPU and Metal run the same freshly built Release executable, checkpoint, prompt IDs, greedy sampler, and token counts. The benchmark uses batch 1, 16 repeated BOS IDs (`1`), 32 generated tokens, two excluded warm-ups and five measured runs. EOS stopping is disabled. JSON contains actual generated IDs, prompt IDs, all inter-token intervals, and per-run elapsed times. The evidence runner executes benchmarks sequentially; this is a desktop session without frequency, thermal, or background-process isolation.

All timings use `std::chrono::steady_clock`. Model loading, Metal pipeline compilation, weight preparation, metadata collection, session allocation, tokenization, and printing occur outside measured inference. Each run resets the logical KV lengths but retains allocations and prepared weights. These are warm-model results; no cold-start latency is claimed.

- **Prefill:** validate the prompt, reset cache lengths, then process every prompt token and initialize K/V state. It includes the LM head at every position in this implementation. Throughput is total prompt tokens divided by summed prefill time.
- **TTFT:** elapsed time from entry to prefill through the first greedy selection. It excludes tokenization, model loading, output detokenization, and console/network delivery. This measures time until a token ID is ready, not end-to-end application latency.
- **Decode:** after the first generated token, perform 31 autoregressive forwards and selections. `generated_tokens=32` includes the first token from prefill; the decode throughput numerator is 31 per run, never 32. Total decode time is the sum of inter-token intervals.
- **Inter-token latency:** wall time between consecutive selected IDs, including decode, selection, and small host bookkeeping costs. No output callback is used in the benchmark. Mean is arithmetic mean; p50/p95/p99 use nearest rank `ceil(p*n)-1` after sorting pooled intervals.

The 155 pooled intervals support a descriptive p99, but five correlated sequences are not enough for a stable tail-latency guarantee. Individual JSON samples carry extra digits to permit arithmetic auditing, not to claim nanosecond accuracy. Reported tables round milliseconds to two decimals.

Metal operations already call `waitUntilCompleted` before returning. Timers therefore include completed GPU work, host encoding, waiting, buffer creation, and copies; they do not merely measure command submission. No extra synchronization was introduced for profiling.

## CPU and Metal results

Unprofiled runs supply the headline performance figures. Prefill and decode elapsed times below are means across runs; totals and per-run samples are in the linked JSON.

<!-- BEGIN PERFORMANCE -->

| Metric | [CPU](../benchmarks/results/20260918T043128Z-benchmark-cpu.json) | [Metal](../benchmarks/results/20260918T043128Z-benchmark-metal.json) |
|---|---|---|
| Prefill elapsed mean (ms) | 310.86 | 415.74 |
| Prefill throughput (tok/s) | 51.47 | 38.49 |
| Decode elapsed mean (ms) | 584.34 | 802.83 |
| Decode throughput (tok/s) | 53.05 | 38.61 |
| TTFT mean (ms) | 310.91 | 415.80 |
| Inter-token mean (ms) | 18.85 | 25.90 |
| Inter-token p50 (ms) | 18.80 | 25.89 |
| Inter-token p95 (ms) | 19.37 | 27.53 |
| Inter-token p99 (ms) | 20.01 | 28.64 |
| Inter-token min (ms) | 18.38 | 23.72 |
| Inter-token max (ms) | 21.24 | 30.12 |

<!-- END PERFORMANCE -->

Peak RSS is the process high-water mark from `getrusage`, including loading/warm-up and allocator overhead. It is not GPU VRAM and cannot be added to mapped weights or prepared GPU buffers. KV bytes represent allocated capacity, not just used entries.

<!-- BEGIN MEMORY -->

| Bytes | CPU | Metal |
|---|---|---|
| model_weight_bytes | 60766848 | 60766848 |
| backend_resident_weight_bytes | 0 | 60751872 |
| kv_cache_bytes | 649728 | 649728 |
| peak_process_memory_bytes | 68894720 | 200654848 |

<!-- END MEMORY -->

## Profiler breakdown

The separate `--profile` executions use the same workload and report combined prefill plus decode work. Each has 235 forward calls (5 × (16 + 31)) and 160 sampling calls. CPU operation scopes measure CPU work and host overhead; Metal scopes include complete synchronous operations. The denominator is summed elapsed generation runtime across measured runs, excluding warm-up. Mean is total divided by calls. Unattributed overhead includes vector allocation, validation, loop control, and profiler bookkeeping; it is not a separately timed operation.

<!-- BEGIN PROFILE -->

**[cpu](../benchmarks/results/20260918T043128Z-profile-cpu.json)**

| Operation | Calls | Total ms | Mean ms | Runtime % |
|---|---|---|---|---|
| Activation gating | 1410 | 2.38 | 0.0017 | 0.05 |
| Attention | 1410 | 13.19 | 0.0094 | 0.28 |
| Embedding | 235 | 0.18 | 0.0008 | 0.00 |
| Final norm | 235 | 0.18 | 0.0008 | 0.00 |
| K projection | 1410 | 158.56 | 0.1125 | 3.42 |
| KV append | 1410 | 0.65 | 0.0005 | 0.01 |
| LM head | 235 | 2764.44 | 11.7636 | 59.57 |
| MLP down | 1410 | 399.65 | 0.2834 | 8.61 |
| MLP gate | 1410 | 407.75 | 0.2892 | 8.79 |
| MLP up | 1410 | 421.30 | 0.2988 | 9.08 |
| Output projection | 1410 | 147.75 | 0.1048 | 3.18 |
| Q projection | 1410 | 155.71 | 0.1104 | 3.36 |
| RMSNorm | 2820 | 1.18 | 0.0004 | 0.03 |
| Residual | 2820 | 0.59 | 0.0002 | 0.01 |
| RoPE | 1410 | 4.14 | 0.0029 | 0.09 |
| Sampling | 160 | 8.51 | 0.0532 | 0.18 |
| V projection | 1410 | 151.61 | 0.1075 | 3.27 |

Unattributed elapsed time: 2.73 ms (0.06%).

**[metal](../benchmarks/results/20260918T043128Z-profile-metal.json)**

| Operation | Calls | Total ms | Mean ms | Runtime % |
|---|---|---|---|---|
| Activation gating | 1410 | 575.87 | 0.4084 | 9.38 |
| Attention | 1410 | 901.56 | 0.6394 | 14.69 |
| Embedding | 235 | 0.22 | 0.0009 | 0.00 |
| Final norm | 235 | 55.07 | 0.2343 | 0.90 |
| K projection | 1410 | 347.68 | 0.2466 | 5.66 |
| KV append | 1410 | 0.75 | 0.0005 | 0.01 |
| LM head | 235 | 230.03 | 0.9789 | 3.75 |
| MLP down | 1410 | 478.84 | 0.3396 | 7.80 |
| MLP gate | 1410 | 359.42 | 0.2549 | 5.85 |
| MLP up | 1410 | 362.12 | 0.2568 | 5.90 |
| Output projection | 1410 | 350.28 | 0.2484 | 5.71 |
| Q projection | 1410 | 348.78 | 0.2474 | 5.68 |
| RMSNorm | 2820 | 650.15 | 0.2306 | 10.59 |
| Residual | 2820 | 564.26 | 0.2001 | 9.19 |
| RoPE | 1410 | 548.70 | 0.3891 | 8.94 |
| Sampling | 160 | 8.80 | 0.0550 | 0.14 |
| V projection | 1410 | 353.56 | 0.2508 | 5.76 |

Unattributed elapsed time: 3.19 ms (0.05%).

<!-- END PROFILE -->

Metal's nested diagnostics overlap those operation scopes and **must not be added to them**. Encoding/submission measures host command creation through `commit`; waiting measures the existing `waitUntilCompleted` call. Waiting includes scheduling and GPU execution, so it is not pure synchronization overhead. These are host wall times, not GPU timestamps or occupancy measurements. They exclude buffer preparation/copy costs outside `run()`.

<!-- BEGIN DIAGNOSTICS -->

| Nested host category | Calls | Total ms | Mean ms | Runtime % |
|---|---|---|---|---|
| Encoding and submission | 25850 | 190.57 | 0.0074 | 3.10 |
| Waiting | 25850 | 5535.96 | 0.2142 | 90.17 |

<!-- END DIAGNOSTICS -->

## Interpretation

<!-- BEGIN COMPARISON -->

**MEASURED FACT:** Metal decode was 38.61 tok/s and CPU decode was 53.05 tok/s; CPU throughput was 1.37× Metal in these unprofiled runs. The profiled Metal run spent 90.17% of measured generation wall time in existing command waits.

Profiling changes the amount of host work. The separately executed profiled runs measured cpu 49.82 tok/s, metal 39.21 tok/s. These differences include run-to-run variation and cannot isolate profiler overhead.

<!-- END COMPARISON -->

**Measured/code-inspection facts:** Metal synchronizes each command and allocates/copies non-weight buffers per operation; attention copies host K/V prefixes and uses separate score, softmax, and value commands. The CPU is single-threaded with direct dot-product loops. These observations describe this implementation and this small batch-1 workload.

**Hypotheses:** reducing command count and keeping activations/cache on the GPU may improve Metal latency. Small matvec work and repeated allocations may leave too little useful work per submission. These timings do not establish the fraction caused by launch latency, memory bandwidth, occupancy, unified-memory behavior, or CPU vectorization. No causal claim about the performance gap is supported without controlled changes and new measurements.

## Optimization experiments to run next

1. Reuse Metal activation/scratch buffers and keep K/V resident; compare identical outputs and isolate allocation/copy effects.
2. Batch commands while preserving dependencies; measure submission count, wait time, and total generation time separately.
3. Fuse norm/projection or activation/gating where practical; validate every intermediate stage again.
4. Replace repeated per-element RMSNorm reductions with cooperative reductions, then investigate matrix-vector kernels with GPU timestamps and counters.
5. Implement prompt GEMMs separately from decode; measure prompt-length and model-size sweeps before extrapolating beyond this checkpoint.

These are proposed experiments, not implemented speedups.

## Reproduce

Build and test as shown in the README, then:

```sh
.venv/bin/python tools/run_evidence.py --build build-audit
.venv/bin/python tools/report_results.py
.venv/bin/python tools/report_results.py --check
```

For a single unprofiled CPU run (use `metal` for comparison, `--profile` for diagnostic runs):

```sh
./build-audit/kestrel_bench --model models/tinystories-15m --backend cpu \
  --prompt-tokens 16 --max-tokens 32 --warmup 2 --iterations 5 \
  --json benchmarks/results/manual-cpu.json
```

The evidence runner assigns UTC timestamped filenames. [`latest.json`](../benchmarks/results/latest.json) selects the documented batch. Earlier `docs/results/` files are historical and are not inputs to current tables. No CUDA performance has been measured.
