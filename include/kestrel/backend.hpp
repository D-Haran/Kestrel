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
    virtual void rope(Output x, size_t heads, size_t dim, size_t position, float theta) = 0;
    virtual void attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads,
                           size_t dim, size_t tokens) = 0;
    virtual void prepare_weight(const TensorView &) {}
    virtual void release_weights() {}
    virtual size_t resident_weight_bytes() const {
        return 0;
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
