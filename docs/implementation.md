# Implementation record

This is a historical implementation record. Current re-executed evidence, including sanitizer status, is in [the upgrade audit](upgrade-audit.md), [correctness](correctness.md), and [performance](performance.md).

Local host: Apple M2 Pro, 16 GB, macOS 26.6, Apple Clang 17. The starting workspace was empty. CMake 3.31.6 and Ninja 1.11.1.4 were installed into the project virtual environment. No native CUDA toolchain or NVIDIA GPU was available.

The requested milestone order was followed, with native builds and relevant tests after each implementation step. This record distinguishes working implementations from environment-limited validation.

| Milestone | Implemented | Evidence / limits |
|---|---|---|
| M1 | C++20/CMake structure, owning tensors and non-owning views, shapes/strides, overflow checks | Native build and unit tests passed |
| M2 | Strict config, memory-mapped Safetensors, dtype/shape/offset validation, shards | Build and loader tests passed; later corruption and shard integration tests passed |
| M3 | CPU matmul, RMSNorm, add/multiply, SiLU, softmax, split-half RoPE, GQA attention | Native math tests passed; later independent NumPy comparisons passed |
| M4 | Explicit Llama transformer from embeddings through logits | Fixture forward, finite logits, repeated forward and cached/full equivalence passed |
| M5 | Contiguous cache, explicit prefill/decode, greedy autoregression, EOS, memory accounting | Cache boundaries/isolation/reset and deterministic generation passed |
| M6 | HF eager traces and independent NumPy references | CPU fixture and pretrained 15M checkpoint passed all stage tolerances |
| M7 | CLI, tokenizer helper, interactive mode, temperature/top-k/top-p/seed | Sampling tests and real CLI generation passed; later Unicode and both tokenizer-format checks passed |
| M8 | Scoped profiler, real benchmark executable, raw JSON samples and memory metrics | Benchmark schema integration passed; CPU measurements recorded; profiler output checked |
| M9 | Original Metal kernels, buffers/weight preparation, error handling | Metal compilation, device execution, NumPy/HF comparisons, dtype/shard/tied variants, and pretrained generation passed |
| M10 | Independent CUDA CMake option, RAII buffers and actual CUDA kernels | **Pending validation:** no nvcc/NVIDIA GPU; local configure fails explicitly for missing toolkit. Docker daemon remained unresponsive. Linux compile-only CI workflow supplied but not executed. No CUDA results claimed. |
| M11 | Documentation, formatting, integration tests, reproducible checkpoint lock, runtime dependency audit | CPU 7/7, UBSan 7/7, Metal 11/11 CTest entries passed. Text/interactive integrations passed separately. ASan limitation below. Full multi-backend completion remains contingent on M10 validation. |

## Issues found and resolved

- A local JSON wrapper initially included itself through a quoted basename; corrected to the vendored include path before marking M2 complete.
- The real tied-weight checkpoint stores only `lm_head.weight`; added the tied alias path and strict reference loading, then tested both tied storage layouts.
- Profiler rendering initially iterated a JSON temporary through an items proxy; retained the JSON value and added a table-output regression check.
- Hugging Face's default resolve endpoint returned HTTP 429; the download endpoint succeeded. Downloads now include retry handling and pinned SHA256 verification.
- Metal was unavailable inside the sandbox, then passed with authorized device access. Nested Python/PyTorch reference processes needed normal OpenMP shared-memory access; those tests passed outside the sandbox.

## Remaining environment limitations

ASan builds successfully, but Apple Clang 17's ASan library deadlocks during initialization before `main` on this macOS 26.6 host. Reproduced outside the sandbox with timed tests and a process stack sample. The stack enters `InitializeShadowMemory`, `dyld_shared_cache_iterate_text_swift`, and recursive `AsanInitFromRtl`, then spins in `StaticSpinMutex::LockSlow`. All affected local ASan tests are recorded as unverified, not passed. Stalled processes were stopped. UBSan passed the complete CPU suite. A Linux sanitizer CI job is provided, without claiming that CI has run.

CUDA source implements the current backend contract but compilation and execution remain unverified. Run the CUDA build and full GPU tests on an appropriate machine before advertising it as supported production execution.
