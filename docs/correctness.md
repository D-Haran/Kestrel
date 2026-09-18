# Numerical correctness

## What is compared

`tools/validate.py` loads the exact same Safetensors checkpoint into Hugging Face `LlamaForCausalLM`, strictly loads the state dictionary, restores the checkpoint's tied embedding alias, and uses eager CPU FP32 reference inference. Native inference is independently implemented. Reference package versions, tensor dtype, checkpoint hashes/revision, tolerances, backend, timestamp, and input IDs are recorded in every new validation JSON.

The pinned checkpoint is `sdobson/tinystories-llama-15m`, revision `6cc2bab8c4a55cce6f7532317af84a94bb340d5e`. The input is the exact token sequence `[1, 9038, 2501, 263, 931, 29889]`; validation does not depend on a tokenizer's rendering of a natural-language prompt. Stored weights and reference/native activations are FP32.

The reference processes all six tokens with a causal mask. Native tracing prefills the first token and decodes the other five using its KV cache. Hooks compare embeddings, each of six decoder-layer outputs, final normalized hidden states, and final logits at **all six positions**. Shapes are `[6,288]` for hidden stages and `[6,32000]` for logits. Metrics are calculated in NumPy FP64 from the FP32 outputs.

Exact floating-point equality is not expected: reduction order, contraction, and transcendental implementations differ. Every stage must satisfy max absolute error ≤ `2e-4`, mean absolute error ≤ `2e-5`, and cosine similarity ≥ `0.99999`. These are gates for the tested fixture/checkpoint, not a universal error bound for all models. Nonfinite metrics fail the gates.

<!-- BEGIN REFERENCE -->

Executed reference: **transformers 4.51.3, PyTorch 2.6.0**, dtype `float32`. Checkpoint SHA256: `8e291ca975614c7d76c01610e753eaa64e4563e9796c0b0791310333893e0ebb`.

<!-- END REFERENCE -->

<!-- BEGIN STAGES -->

| Backend | Stage | Max abs error | Mean abs error | Cosine similarity |
|---|---|---|---|---|
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | embedding | 0.000000e+00 | 0.000000e+00 | 1.000000000000000 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_0 | 9.536743e-07 | 6.101045e-08 | 0.999999999999854 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_1 | 1.907349e-06 | 1.471968e-07 | 0.999999999999950 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_2 | 2.861023e-06 | 2.387942e-07 | 0.999999999999908 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_3 | 3.814697e-06 | 3.667746e-07 | 0.999999999999881 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_4 | 3.814697e-06 | 4.907426e-07 | 0.999999999999848 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | layer_5 | 8.106232e-06 | 7.519309e-07 | 0.999999999999787 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | final_hidden | 4.100800e-05 | 3.134433e-06 | 0.999999999999633 |
| [cpu](../benchmarks/results/20260918T043128Z-validation-cpu.json) | logits | 3.695488e-05 | 4.123539e-06 | 0.999999999999775 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | embedding | 0.000000e+00 | 0.000000e+00 | 1.000000000000000 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_0 | 5.364418e-07 | 6.009867e-08 | 0.999999999999861 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_1 | 1.907349e-06 | 1.517936e-07 | 0.999999999999945 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_2 | 1.907349e-06 | 2.567734e-07 | 0.999999999999898 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_3 | 4.291534e-06 | 4.228657e-07 | 0.999999999999844 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_4 | 3.337860e-06 | 5.255214e-07 | 0.999999999999827 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | layer_5 | 9.536743e-06 | 7.991093e-07 | 0.999999999999771 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | final_hidden | 2.861023e-05 | 3.204216e-06 | 0.999999999999652 |
| [metal](../benchmarks/results/20260918T043128Z-validation-metal.json) | logits | 4.124641e-05 | 4.932517e-06 | 0.999999999999685 |

<!-- END STAGES -->

## Tests and primitive comparisons

`validate_ops.py` independently computes NumPy matmul, add, RMSNorm, SiLU, SwiGLU, softmax, split-half RoPE, and nonuniform grouped-query attention. It gates outputs with absolute tolerance `2e-6` and relative tolerance `2e-5`. These run in CTest for both CPU and Metal. The miniature GQA fixture uses four query heads and two KV heads; the pretrained checkpoint has equal query/KV head counts.

Native tests additionally cover dtype decoding, shape errors, cache boundaries/reset/isolation, cached/full equivalence, greedy generation, EOS and sampling. Executable integration rejects malformed checkpoints and CLI inputs, and recomputes benchmark arithmetic from raw samples. Variant tests cover F16/BF16 storage expanded to FP32 compute, indexed shards, and both tied-weight layouts. Text/interactive and Unicode tokenizer tests execute separately in the evidence runner.

<!-- BEGIN TESTS -->

| Configuration | Result | Evidence |
|---|---|---|
| CPU Release | 7/7 passed | [log](../benchmarks/results/cpu-host-tests.log) |
| CPU + Metal Release | 11/11 passed | [log](../benchmarks/results/release-device-tests.log) |
| CPU Debug UBSan | 7/7 passed | [log](../benchmarks/results/ubsan-strict-tests.log) |
| ASan Debug | Compiled; execution timed out before main | [probe](../benchmarks/results/asan-probe.json) |

<!-- END TESTS -->

CTest counts are test entries, not counts of individual numeric assertions. CPU-only Release and Debug UBSan use separate clean build directories; the 11-entry Release suite includes CPU tests plus four Metal entries. CUDA has not been compiled or executed in a working NVIDIA environment during this task.

## Sanitizer and environment findings

The address-only Debug build completed. Outside the sandbox, both `build-audit-asan/kestrel --help` and a minimal C program that prints on entry to `main` timed out after eight seconds without output. A one-second stack sample of the minimal program shows `InitializeShadowMemory`, `dyld_shared_cache_iterate_text_swift`, recursive `AsanInitFromRtl`, and `StaticSpinMutex::LockSlow`. This reproduces the issue independently of Kestrel and is evidence of a local runtime/toolchain initialization problem; it is not proof that Kestrel is memory-safe. Processes were killed after sampling. **ASan is unverified, not passing.**

See [probe results](../benchmarks/results/asan-probe.json), [stack](../benchmarks/results/asan-stack.txt), and [build log](../benchmarks/results/asan-build.log). The probe source and commands are retained in [the audit](upgrade-audit.md).

Sandboxed Metal tests fail with `no Metal device available`; nested Python variant comparisons can fail with OpenMP `Can't open SHM`. The documented accepted suites ran with host device/shared-memory access. Failed attempts remain in the results directory. No CI execution is claimed.

## Reproduce

From the repository root, with `.venv/bin` on PATH:

```sh
cmake -S . -B build-audit-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DKESTREL_REFERENCE_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit-cpu -j 4
ctest --test-dir build-audit-cpu --output-on-failure --timeout 60

cmake -S . -B build-audit-ubsan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DKESTREL_SANITIZE=ON -DKESTREL_SANITIZERS=undefined \
  -DKESTREL_REFERENCE_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-audit-ubsan -j 4
ctest --test-dir build-audit-ubsan --output-on-failure --timeout 60

python tools/validate.py --model models/tinystories-15m \
  --executable build-audit/kestrel_trace --backend cpu \
  --tokens 1,9038,2501,263,931,29889 --output benchmarks/results/manual-validation-cpu.json
```

Use `--backend metal` with the same executable for the GPU comparison. `tools/run_evidence.py` executes both and preserves commands. For ASan, use a separate directory with `-DKESTREL_SANITIZE=ON -DKESTREL_SANITIZERS=address -DCMAKE_BUILD_TYPE=Debug`; run with a timeout on this host. A passing build does not establish a passing sanitizer run.
