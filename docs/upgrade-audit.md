# Upgrade audit and execution record

## Initial audit

The starting tree already contained `benchmarks/main.cpp`, scoped profiling in `src/profiler.cpp`, a native CPU backend, synchronous Metal operations, unverified CUDA kernels, a pretrained TinyStories checkpoint, Python reference tracing/primitive/variant comparisons, CTest integration, and explanatory docs. Existing benchmark timings were treated as provisional and were not copied into current results.

Machine discovery reported macOS 26.6 (25G5057c), MacBook Pro Mac14,9, Apple M2 Pro, 10 CPU cores (6 performance/4 efficiency), 16 GPU cores, and 16 GB memory. Apple Clang is 17.0.0 (clang-1700.3.19.1); CMake 3.31.6 and Ninja are installed in `.venv`. The checkpoint header contains 15,191,712 F32 elements: hidden 288, intermediate 768, six layers, six query/KV heads, head width 48, vocabulary 32,000. Its tied embedding/LM head is counted once.

The repository root has no `.git`; a commit identity cannot be recovered. Benchmarks emit `unknown`, and the evidence manifest fingerprints source files as they existed at measurement time. Later Python formatting/report-generation edits do not change the measured native code. No project-facing Kestral misspelling was found. The existing filesystem path `/Users/derrickratnaharan/kestral` remains unchanged for compatibility; absolute paths in logs therefore retain that spelling.

## Files changed

- `CMakeLists.txt`: benchmark compiler/build/CMake/commit metadata.
- `benchmarks/main.cpp`, new `benchmarks/metadata.hpp`: environment/model metadata, explicit prefill/decode totals, raw per-run fields, latency range/sample caveats, nested backend diagnostics.
- `include/kestrel/backend.hpp`, `include/kestrel/model.hpp`, `include/kestrel/profiler.hpp`: diagnostic profiler, session profiler access, elapsed-time recording and runtime denominator.
- `src/model.cpp`, `src/generation.cpp`, `src/profiler.cpp`, `src/backends/metal.mm`: finer operation scopes, sampling, means/runtime percentages, existing Metal submission/wait measurements without new waits.
- `tests/integration.py`: independently recompute latency percentiles and throughput; verify metadata, sampling counts, warm-up exclusion, and profile arithmetic.
- `tools/validate.py`: timestamps, package/dtype metadata and actual checkpoint hashes.
- New `tools/run_evidence.py`: sequential checkpoint validation, reference/tokenizer tests and comparable benchmark/profile runs, timestamped artifacts, exact commands and source fingerprints.
- `tools/report_results.py`: audit evidence and generate README/correctness/performance tables; `--check` detects drift.
- `README.md`: concise project overview, diagram, early correctness/performance, CLI/build/test instructions, explicit limitations.
- New `docs/architecture.md`, `docs/correctness.md`, `docs/performance.md`, this audit; refreshed `docs/backends.md`, `docs/kv-cache.md`, `docs/implementation.md`; old `docs/transformer.md`, `docs/validation.md`, and `docs/benchmarking.md` now direct readers to maintained pages.
- New `benchmarks/results/`: timestamped execution JSON/logs, test/build logs, source/command manifest, hardware and sanitizer evidence. New `docs/results/README.md` labels older evidence historical.

## Exact build and test commands

All commands ran from the repository root. The four `build-audit*` directories were new, so the initial builds were clean. The Metal-enabled Release executable is used for both headline CPU and Metal measurements. A separate CPU-only Release build independently validates the non-Metal configuration.

```sh
pwd
rg --files
sw_vers
system_profiler SPHardwareDataType SPDisplaysDataType
clang --version
.venv/bin/cmake --version
git status --short
```

The last command reported that this is not a Git repository. Bare `cmake --version` initially failed because `.venv/bin` was not on PATH. Hardware JSON is filtered to non-identifying machine/chip/memory/GPU fields in `benchmarks/results/hardware.json`.

```sh
export PATH="$PWD/.venv/bin:$PATH"
cmake -S . -B build-audit -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DKESTREL_METAL=ON -DKESTREL_REFERENCE_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit -j 4
ctest --test-dir build-audit --output-on-failure --timeout 60

cmake -S . -B build-audit-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DKESTREL_REFERENCE_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit-cpu -j 4
ctest --test-dir build-audit-cpu --output-on-failure --timeout 60

cmake -S . -B build-audit-ubsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKESTREL_SANITIZE=ON -DKESTREL_SANITIZERS=undefined \
  -DKESTREL_REFERENCE_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit-ubsan -j 4
ctest --test-dir build-audit-ubsan --output-on-failure --timeout 60
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-audit-ubsan --output-on-failure --timeout 60

cmake -S . -B build-audit-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKESTREL_SANITIZE=ON -DKESTREL_SANITIZERS=address \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit-asan -j 4
```

CTest stdout/stderr was redirected to `benchmarks/results/*tests.log`; CPU/UBSan/ASan build logs are also retained. The Release suite was rerun with GPU access after the sandbox denied the Metal device. CPU and UBSan suites were rerun with host shared-memory access after sandboxed OpenMP initialization failed. These are environment failures, not passing attempts. Accepted logs are `release-device-tests.log`, `cpu-host-tests.log`, `ubsan-host-tests.log`, and `ubsan-strict-tests.log`.

## ASan probe

The address-only build succeeded. To distinguish a project bug from runtime initialization, this minimal source was compiled as `/tmp/kestrel-asan-probe.c`:

```c
#include <stdio.h>
int main(void) { fprintf(stderr, "entered main\n"); return 0; }
```

```sh
clang -fsanitize=address /tmp/kestrel-asan-probe.c -o /tmp/kestrel-asan-probe
```

A Python `subprocess.Popen` wrapper ran each exact argument vector below outside the sandbox, waited eight seconds with `communicate(timeout=8)`, sampled the stalled process with `/usr/bin/sample PID 1 -file benchmarks/results/asan-stack.txt`, killed it, and collected stdout/stderr. `PID` was the actual child PID. The second sample replaces the first, so the retained stack is the minimal program, not Kestrel.

```text
["build-audit-asan/kestrel", "--help"]
["/tmp/kestrel-asan-probe"]
```

Both timed out with empty output. The minimal process stack shows recursive ASan shadow-memory initialization and a sanitizer spin mutex before main. ASan remains unverified; no memory-safety conclusion follows from a successful compile or from UBSan.

## Measurement and documentation commands

```sh
.venv/bin/python tools/run_evidence.py --build build-audit
.venv/bin/python tools/report_results.py
.venv/bin/python tools/report_results.py --check
./build-audit/kestrel --model models/tinystories-15m --backend cpu \
  --prompt "Once upon a time" --max-tokens 32 --temperature 0 \
  --tokenizer-python .venv/bin/python
.venv/bin/python tools/audit_runtime.py
.venv/bin/python -m ruff format tools/report_results.py tools/run_evidence.py tools/validate.py tests/integration.py
.venv/bin/python -m ruff check tools/report_results.py tools/run_evidence.py tools/validate.py tests/integration.py
```

The evidence runner's expanded commands are below, generated from its actual manifest. It verified existing checkpoint files locally; no new download was needed. Measurements were sequential with no competing builds/tests launched by this task. The README's shell commands were executed directly or through the equivalent activated `.venv` paths above. The generation output is retained in `benchmarks/results/readme-generation.txt`.

<!-- BEGIN COMMANDS -->

```sh
/Users/derrickratnaharan/kestral/.venv/bin/python tools/download_model.py
/Users/derrickratnaharan/kestral/.venv/bin/python tools/validate.py --model models/tinystories-15m --executable build-audit/kestrel_trace --backend cpu --tokens 1,9038,2501,263,931,29889 --output benchmarks/results/20260918T043128Z-validation-cpu.json
/Users/derrickratnaharan/kestral/.venv/bin/python tests/tokenizer_integration.py --binary build-audit/kestrel --model models/tinystories-15m --backend cpu
/Users/derrickratnaharan/kestral/.venv/bin/python tools/validate.py --model models/tinystories-15m --executable build-audit/kestrel_trace --backend metal --tokens 1,9038,2501,263,931,29889 --output benchmarks/results/20260918T043128Z-validation-metal.json
/Users/derrickratnaharan/kestral/.venv/bin/python tests/tokenizer_integration.py --binary build-audit/kestrel --model models/tinystories-15m --backend metal
build-audit/kestrel_bench --model models/tinystories-15m --backend cpu --prompt-tokens 16 --max-tokens 32 --warmup 2 --iterations 5 --json benchmarks/results/20260918T043128Z-benchmark-cpu.json
build-audit/kestrel_bench --model models/tinystories-15m --backend metal --prompt-tokens 16 --max-tokens 32 --warmup 2 --iterations 5 --json benchmarks/results/20260918T043128Z-benchmark-metal.json
build-audit/kestrel_bench --model models/tinystories-15m --backend cpu --prompt-tokens 16 --max-tokens 32 --warmup 2 --iterations 5 --json benchmarks/results/20260918T043128Z-profile-cpu.json --profile
build-audit/kestrel_bench --model models/tinystories-15m --backend metal --prompt-tokens 16 --max-tokens 32 --warmup 2 --iterations 5 --json benchmarks/results/20260918T043128Z-profile-metal.json --profile
```

<!-- END COMMANDS -->

## Verified outcomes

See [correctness](correctness.md) for full per-stage metrics and accepted test counts, and [performance](performance.md) for elapsed times, latency distributions, and profiler categories. All headline values are generated from current artifacts. JSON arithmetic and greedy trajectories are checked by `tools/report_results.py --check`.

The workload demonstrates native inference for one pinned small checkpoint and primitive/configuration fixtures. Metal measurements describe completed work, not asynchronous submission. Nested waiting time includes GPU execution and scheduling; it does not isolate overhead or establish causality. No optimization speedup was attempted or claimed.

## Claims that should not appear on a resume

- Validated CUDA/NVIDIA execution, CUDA compilation, or CUDA performance.
- Passing ASan or comprehensive memory-safety certification.
- Metal faster than CPU, a causal diagnosis of the gap, GPU occupancy/bandwidth claims, or an implemented speedup.
- Production-scale LLM serving, continuous batching, paged attention, quantized compute, or broad Llama compatibility.
- End-to-end application TTFT, cold-start latency, or performance on a larger model inferred from this checkpoint.
- A verified CI run or a Git commit identity for this snapshot.

## Evidence-based resume bullets

- Implemented native C++20 Llama inference with memory-mapped Safetensors, KV caching, and CPU/Metal kernels; validated a 15.19M-parameter checkpoint against Hugging Face at every layer and final logits.
- Built auditable prefill/decode benchmarks and operation profiling with hardware metadata, raw latency samples, and synchronized Metal diagnostics; verified CPU, Metal, and UBSan test suites and documented unverified CUDA/ASan scope.
