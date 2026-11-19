#include "kestrel/backend.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace kestrel {
void require_size(Input x, size_t expected, const char *op) {
    if (x.size() != expected)
        throw std::invalid_argument(std::string(op) + ": shape mismatch");
}
void validate_attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads,
                        size_t dim, size_t tokens) {
    if (!heads || !kv_heads || !dim || !tokens || heads % kv_heads)
        throw std::invalid_argument("invalid attention dimensions");
    require_size(q, checked_mul(heads, dim), "attention Q");
    require_size(y, q.size(), "attention output");
    require_size(k, checked_mul(tokens, checked_mul(kv_heads, dim)), "attention K");
    require_size(v, k.size(), "attention V");
}
class CpuBackend final : public Backend {
  public:
    std::string name() const override {
        return "cpu";
    }
    void matmul(const TensorView &w, Input x, Output y, size_t batch) override {
        if (w.shape().size() != 2)
            throw std::invalid_argument("matmul weight must have rank 2");
        size_t rows = w.shape()[0], cols = w.shape()[1];
        require_size(x, checked_mul(batch, cols), "matmul input");
        require_size(y, checked_mul(batch, rows), "matmul output");
        const auto stride = dtype_size(w.dtype());
        if (batch == 1) {
            for (size_t r = 0; r < rows; ++r) {
                float sum = 0;
                const auto *p = w.data() + r * cols * stride;
                for (size_t c = 0; c < cols; ++c)
                    sum += decode_scalar(p + c * stride, w.dtype()) * x[c];
                y[r] = sum;
            }
            return;
        }
        // Batched (prefill): decode each weight element once and reuse it across all batch
        // rows, so the weight matrix is streamed from memory once instead of `batch` times.
        for (size_t b = 0; b < batch * rows; ++b)
            y[b] = 0;
        for (size_t r = 0; r < rows; ++r) {
            const auto *p = w.data() + r * cols * stride;
            for (size_t c = 0; c < cols; ++c) {
                float wv = decode_scalar(p + c * stride, w.dtype());
                for (size_t b = 0; b < batch; ++b)
                    y[b * rows + r] += wv * x[b * cols + c];
            }
        }
    }
    void rmsnorm(Input x, Input w, Output y, float eps) override {
        require_size(w, x.size(), "rmsnorm weight");
        require_size(y, x.size(), "rmsnorm output");
        if (x.empty() || !(eps > 0) || !std::isfinite(eps))
            throw std::invalid_argument("invalid RMSNorm input");
        float sum = 0;
        for (float v : x)
            sum += v * v;
        float scale = 1 / std::sqrt(sum / float(x.size()) + eps);
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = x[i] * scale * w[i];
    }
    void add(Input a, Input b, Output y) override {
        require_size(b, a.size(), "add");
        require_size(y, a.size(), "add output");
        for (size_t i = 0; i < a.size(); ++i)
            y[i] = a[i] + b[i];
    }
    void multiply(Input a, Input b, Output y) override {
        require_size(b, a.size(), "multiply");
        require_size(y, a.size(), "multiply output");
        for (size_t i = 0; i < a.size(); ++i)
            y[i] = a[i] * b[i];
    }
    void silu(Input x, Output y) override {
        require_size(y, x.size(), "silu");
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = x[i] / (1 + std::exp(-x[i]));
    }
    void softmax(Input x, Output y) override {
        require_size(y, x.size(), "softmax");
        if (x.empty())
            throw std::invalid_argument("empty softmax");
        float max = *std::max_element(x.begin(), x.end()), sum = 0;
        if (!std::isfinite(max))
            throw std::invalid_argument("softmax requires finite maximum");
        for (size_t i = 0; i < x.size(); ++i) {
            y[i] = std::exp(x[i] - max);
            sum += y[i];
        }
        if (!std::isfinite(sum))
            throw std::invalid_argument("softmax contains NaN");
        for (auto &v : y)
            v /= sum;
    }
    void rope(Output x, size_t heads, size_t dim, size_t position, Input inv_freq) override {
        require_size(x, checked_mul(heads, dim), "rope");
        if (!dim || dim % 2)
            throw std::invalid_argument("invalid RoPE parameters");
        require_size(inv_freq, dim / 2, "rope inv_freq");
        // Hugging Face Llama uses split-half rotate_half, not interleaved pairs.
        for (size_t h = 0; h < heads; ++h)
            for (size_t i = 0; i < dim / 2; ++i) {
                float angle = float(position) * inv_freq[i];
                float a = x[h * dim + i], b = x[h * dim + i + dim / 2], c = std::cos(angle),
                      s = std::sin(angle);
                x[h * dim + i] = a * c - b * s;
                x[h * dim + i + dim / 2] = b * c + a * s;
            }
    }
    void attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads, size_t dim,
                   size_t tokens) override {
        validate_attention(q, k, v, y, heads, kv_heads, dim, tokens);
        Vec scores(tokens), probs(tokens);
        for (size_t h = 0; h < heads; ++h) {
            size_t kh = h / (heads / kv_heads);
            for (size_t t = 0; t < tokens; ++t) {
                float dot = 0;
                for (size_t d = 0; d < dim; ++d)
                    dot += q[h * dim + d] * k[(t * kv_heads + kh) * dim + d];
                scores[t] = dot / std::sqrt(float(dim));
            }
            softmax(scores, probs);
            for (size_t d = 0; d < dim; ++d) {
                float sum = 0;
                for (size_t t = 0; t < tokens; ++t)
                    sum += probs[t] * v[(t * kv_heads + kh) * dim + d];
                y[h * dim + d] = sum;
            }
        }
    }
};
std::unique_ptr<Backend> make_cpu_backend() {
    return std::make_unique<CpuBackend>();
}
std::unique_ptr<Backend> make_backend(const std::string &name) {
    if (name == "cpu")
        return make_cpu_backend();
#ifdef KESTREL_HAS_METAL
    if (name == "metal")
        return make_metal_backend();
#endif
#ifdef KESTREL_HAS_CUDA
    if (name == "cuda")
        return make_cuda_backend();
#endif
    throw std::invalid_argument("backend unavailable in this build: " + name);
}
} // namespace kestrel
