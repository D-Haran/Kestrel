# Backend architecture

The public synchronous `Backend` interface separates transformer structure from arithmetic. Tensor views retain dtype, shape, byte extent, and contiguous strides; the checkpoint owner outlives all weight views. Activations use explicit float spans and output spans. Every operation checks dimensions before touching memory.

`matmul` computes `X[batch,in] * W[out,in]^T`. Elementwise methods support input/output aliasing. Matmul inputs and outputs must not overlap. RMSNorm operates on one vector; RoPE accepts flattened head-major vectors; attention consumes token-major K/V prefixes. These contracts make the same model code run on each backend without an inference framework.

## CPU

The CPU backend is a single-threaded reference implementation. It decodes F32/F16/BF16 weights directly from the memory map and accumulates scalar dot products. Softmax subtracts the maximum. No BLAS, external graph executor, tensor engine, or device fallback is hidden behind this backend. SIMD/BLAS optimization is future work; the reference path should remain available.

## Metal

`metal.mm` owns the device, command queue, pipeline objects, shared buffers, and prepared weight buffers through Objective-C ARC. Original Metal Shading Language kernels live in `kernels.metal`, are embedded by CMake, and compile at runtime with safe math. Execution errors propagate as exceptions. The backend fails if there is no device; it never silently selects CPU.

Matrix multiplication, RMSNorm, RoPE, elementwise math, and all three attention stages (scores, softmax, value reduction) execute on the GPU. Each operation waits for its command buffer before returning, so host wall-clock profiling includes completed GPU work. Projection weights are converted to FP32 once when the model loads. Other buffers are currently allocated and copied per operation. KV prefixes live on the host and are copied for each attention call.

These choices make the implementation readable and testable, but incur synchronization, allocation, transfer, and redundant reduction costs. RMSNorm currently recomputes its sum per output element. The recorded batch-1 workload is faster on the CPU; see [executed performance evidence](performance.md). Future work: threadgroup reductions, persistent device activations/cache, reusable scratch buffers, fused kernels, and command batching.

## CUDA

`cuda.cu` and `cuda_kernels.cuh` implement independent CUDA device kernels for the same operations. Device allocations are RAII-owned; CUDA API and kernel launch errors are checked; synchronous completion makes errors observable. No Metal headers or frameworks enter the CUDA build. CUDA 12+ with a C++20 host compiler is required. `--fmad=false` keeps contraction differences small for reference comparisons.

**Local status:** source implemented; no local CUDA compiler or NVIDIA GPU. Docker was installed but its daemon did not respond, including after requesting app startup. Neither CUDA compilation nor execution is claimed verified. `.github/workflows/build.yml` includes a Linux CUDA compile job; that workflow has not been run in this workspace. On an NVIDIA machine, run all `cuda` CTest entries and reference comparisons before treating this backend as validated. There are no CUDA benchmark results.

## Ownership and concurrency

Create the backend before the model and destroy it afterward. Model validation completes before GPU weight preparation; failed preparation releases partially prepared buffers. Model destruction releases backend weight buffers before its memory maps disappear. Session owns its KV cache. Instances are not thread-safe: use an independent backend/model/session per concurrent worker. Backend cache keys are valid only during model lifetime; ephemeral standalone matmul weights are uploaded without caching.

The mapped format and tokenizer process implementation currently target POSIX macOS/Linux. Windows is not supported. The owning float tensor currently requires a little-endian host. Tensor views are contiguous; arbitrary-strided compute is not implemented.
