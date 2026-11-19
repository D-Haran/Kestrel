#pragma once
#include "kestrel/tensor.hpp"
#include "kestrel/profiler.hpp"
#include <memory>
#include <string>
namespace kestrel {
using Vec = std::vector<float>;
using Input = std::span<const float>;
using Output = std::span<float>;
// Synchronous contract: inputs/outputs are host FP32, weights retain checkpoint dtype.
// matmul computes X[batch,in] * W[out,in]^T -> Y[batch,out].
// attention reads only the supplied prefix [tokens,kv_heads,dim].
// Scalar dimensions a resident forward pass needs, passed without pulling in config.hpp.
struct ForwardDims {
    size_t hidden, intermediate, heads, kv_heads, head_dim, vocab, capacity;
    float epsilon, theta;
};
// One transformer layer's weights for the resident path. Projection weights are device
// tensors already uploaded by prepare_weight; the two norms are host vectors.
struct ResidentLayer {
    const TensorView *q, *k, *v, *o, *gate, *up, *down;
    Input input_norm, post_norm;
};

class Backend {
  public:
    Profiler diagnostics; // Nested host timings; never add these to operation totals.
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual void matmul(const TensorView &w, Input x, Output y, size_t batch = 1) = 0;
    virtual void rmsnorm(Input x, Input w, Output y, float epsilon) = 0;
    virtual void add(Input a, Input b, Output y) = 0;
    virtual void multiply(Input a, Input b, Output y) = 0;
    virtual void silu(Input x, Output y) = 0;
    virtual void softmax(Input x, Output y) = 0;
    // inv_freq holds the precomputed inverse frequencies (dim/2), so any RoPE variant
    // (plain or llama3-scaled) is selected by the caller, not recomputed here.
    virtual void rope(Output x, size_t heads, size_t dim, size_t position, Input inv_freq) = 0;
    virtual void attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads,
                           size_t dim, size_t tokens) = 0;
    virtual void prepare_weight(const TensorView &) {}
    virtual void release_weights() {}
    virtual size_t resident_weight_bytes() const {
        return 0;
    }
    // Resident forward path: when supported, the whole step is encoded into one command
    // buffer with activations and the KV cache kept on the device and a single sync at the
    // end. Backends that do not support it keep the op-by-op host path (used for tracing).
    virtual bool supports_resident_forward() const {
        return false;
    }
    // Reset the device-resident KV cache state (call at the start of a sequence).
    virtual void reset_resident_state() {}
    // Run one step. embedding is the gathered host embedding row for `token`; head/final_norm
    // are the output projection and final RMSNorm weights; logits receives the host result.
    virtual void resident_forward(Input embedding, std::span<const ResidentLayer> layers,
                                  Input final_norm, const TensorView &head, const ForwardDims &dims,
                                  Input inv_freq, size_t position, Output logits) {
        (void)embedding, (void)layers, (void)final_norm, (void)head, (void)dims, (void)inv_freq,
            (void)position, (void)logits;
        throw std::logic_error("resident_forward unsupported");
    }
};
std::unique_ptr<Backend> make_backend(const std::string &name);
std::unique_ptr<Backend> make_cpu_backend();
#ifdef KESTREL_HAS_METAL
std::unique_ptr<Backend> make_metal_backend();
#endif
#ifdef KESTREL_HAS_CUDA
std::unique_ptr<Backend> make_cuda_backend();
#endif
void require_size(Input x, size_t expected, const char *operation);
void validate_attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads,
                        size_t dim, size_t tokens);
} // namespace kestrel
