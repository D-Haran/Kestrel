#include "cuda_kernels.cuh"
#include "kestrel/backend.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
namespace kestrel {
namespace {
void checked(cudaError_t result) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(result));
}
unsigned u32(size_t n) {
    if (n > UINT32_MAX)
        throw std::overflow_error("CUDA dimension exceeds uint32");
    return unsigned(n);
}
unsigned blocks(size_t n) {
    return unsigned((n + 127) / 128);
}
void finish() {
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
}
struct Buffer {
    float *data = nullptr;
    size_t size = 0;
    explicit Buffer(size_t n) : size(n) {
        checked(
            cudaMalloc(reinterpret_cast<void **>(&data), std::max(size_t(4), checked_mul(n, 4))));
    }
    explicit Buffer(Input x) : Buffer(x.size()) {
        if (!x.empty())
            checked(cudaMemcpy(data, x.data(), x.size_bytes(), cudaMemcpyHostToDevice));
    }
    Buffer(Buffer &&other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
    }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer() {
        if (data)
            cudaFree(data);
    }
    void copy(Output y) {
        require_size(y, size, "CUDA output");
        if (!y.empty())
            checked(cudaMemcpy(y.data(), data, y.size_bytes(), cudaMemcpyDeviceToHost));
    }
};
} // namespace
class CudaBackend final : public Backend {
    std::map<const std::byte *, Buffer> weights_;
    size_t resident_ = 0;
    Buffer weight_buffer(const TensorView &w) {
        Vec values(w.size());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = w.at(i);
        return Buffer(values);
    }

  public:
    CudaBackend() {
        int count = 0;
        checked(cudaGetDeviceCount(&count));
        if (!count)
            throw std::runtime_error("no CUDA device available");
        checked(cudaSetDevice(0));
    }
    std::string name() const override {
        return "cuda";
    }
    void prepare_weight(const TensorView &w) override {
        if (!weights_.contains(w.data())) {
            weights_.emplace(w.data(), weight_buffer(w));
            resident_ += w.size() * 4;
        }
    }
    void release_weights() override {
        weights_.clear();
        resident_ = 0;
    }
    size_t resident_weight_bytes() const override {
        return resident_;
    }
    void matmul(const TensorView &w, Input x, Output y, size_t batch) override {
        if (w.shape().size() != 2)
            throw std::invalid_argument("matmul weight rank must be 2");
        cuda_kernels::Params p{};
        p.rows = u32(w.shape()[0]);
        p.cols = u32(w.shape()[1]);
        p.batch = u32(batch);
        require_size(x, checked_mul(batch, p.cols), "matmul input");
        require_size(y, checked_mul(batch, p.rows), "matmul output");
        u32(y.size());
        u32(w.size());
        if (y.empty())
            return;
        std::unique_ptr<Buffer> temporary;
        const float *weight;
        if (weights_.contains(w.data()))
            weight = weights_.at(w.data()).data;
        else {
            temporary = std::make_unique<Buffer>(weight_buffer(w));
            weight = temporary->data;
        }
        Buffer xb(x), yb(y.size());
        cuda_kernels::matmul<<<blocks(y.size()), 128>>>(weight, xb.data, yb.data, p);
        finish();
        yb.copy(y);
    }
    void rmsnorm(Input x, Input w, Output y, float epsilon) override {
        require_size(w, x.size(), "rmsnorm");
        require_size(y, x.size(), "rmsnorm");
        if (x.empty() || !(epsilon > 0) || !std::isfinite(epsilon))
            throw std::invalid_argument("invalid RMSNorm");
        cuda_kernels::Params p{};
        p.n = u32(x.size());
        p.epsilon = epsilon;
        Buffer xb(x), wb(w), yb(y.size());
        cuda_kernels::rmsnorm<<<blocks(y.size()), 128>>>(xb.data, wb.data, yb.data, p);
        finish();
        yb.copy(y);
    }
    void elementwise(Input a, Input b, Output y, unsigned op) {
        require_size(b, a.size(), "elementwise");
        require_size(y, a.size(), "elementwise");
        cuda_kernels::Params p{};
        p.n = u32(a.size());
        p.operation = op;
        if (y.empty())
            return;
        Buffer ab(a), bb(b), yb(y.size());
        cuda_kernels::elementwise<<<blocks(y.size()), 128>>>(ab.data, bb.data, yb.data, p);
        finish();
        yb.copy(y);
    }
    void add(Input a, Input b, Output y) override {
        elementwise(a, b, y, 0);
    }
    void multiply(Input a, Input b, Output y) override {
        elementwise(a, b, y, 1);
    }
    void silu(Input x, Output y) override {
        elementwise(x, x, y, 2);
    }
    void softmax(Input x, Output y) override {
        require_size(y, x.size(), "softmax");
        if (x.empty() || !std::isfinite(*std::max_element(x.begin(), x.end())))
            throw std::invalid_argument("invalid softmax");
        for (float f : x)
            if (std::isnan(f))
                throw std::invalid_argument("NaN softmax");
        cuda_kernels::Params p{};
        p.n = u32(x.size());
        p.rows = 1;
        Buffer xb(x), yb(y.size());
        cuda_kernels::softmax<<<1, 128>>>(xb.data, yb.data, p);
        finish();
        yb.copy(y);
    }
    void rope(Output x, size_t heads, size_t dim, size_t position, float theta) override {
        require_size(x, checked_mul(heads, dim), "rope");
        if (!dim || dim % 2 || !(theta > 0) || !std::isfinite(theta))
            throw std::invalid_argument("invalid RoPE");
        cuda_kernels::Params p{};
        p.heads = u32(heads);
        p.dim = u32(dim);
        p.position = u32(position);
        p.theta = theta;
        u32(x.size());
        if (x.empty())
            return;
        Buffer xb(x), yb(x.size());
        cuda_kernels::rope<<<blocks(x.size() / 2), 128>>>(xb.data, yb.data, p);
        finish();
        yb.copy(x);
    }
    void attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads, size_t dim,
                   size_t tokens) override {
        validate_attention(q, k, v, y, heads, kv_heads, dim, tokens);
        cuda_kernels::Params p{};
        p.heads = u32(heads);
        p.kv_heads = u32(kv_heads);
        p.dim = u32(dim);
        p.tokens = u32(tokens);
        p.rows = p.heads;
        p.n = p.tokens;
        u32(k.size());
        u32(q.size());
        u32(checked_mul(heads, tokens));
        Buffer qb(q), kb(k), vb(v), scores(heads * tokens), probs(heads * tokens), yb(y.size());
        cuda_kernels::attention_scores<<<blocks(heads * tokens), 128>>>(qb.data, kb.data,
                                                                        scores.data, p);
        finish();
        cuda_kernels::softmax<<<blocks(heads), 128>>>(scores.data, probs.data, p);
        finish();
        cuda_kernels::attention_values<<<blocks(y.size()), 128>>>(probs.data, vb.data, yb.data, p);
        finish();
        yb.copy(y);
    }
};
std::unique_ptr<Backend> make_cuda_backend() {
    return std::make_unique<CudaBackend>();
}
} // namespace kestrel
