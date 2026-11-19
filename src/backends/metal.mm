#include "kestrel/backend.hpp"
#include "metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
namespace kestrel {
namespace {
struct Params {
    uint32_t n = 0, rows = 0, cols = 0, batch = 0, heads = 0, kv_heads = 0, dim = 0, tokens = 0,
             position = 0, operation = 0;
    float epsilon = 0, theta = 0;
};
uint32_t u32(size_t n) {
    if (n > UINT32_MAX)
        throw std::overflow_error("Metal dimension exceeds uint32");
    return uint32_t(n);
}
std::string error_text(NSError *error) {
    return error ? std::string([[error localizedDescription] UTF8String]) : "unknown Metal error";
}
} // namespace
class MetalBackend final : public Backend {
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    std::map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::map<const std::byte *, id<MTLBuffer>> weights_;
    std::map<const float *, id<MTLBuffer>> norms_;
    size_t resident_ = 0;
    // Device-resident scratch for the fused forward path, allocated once per model.
    ForwardDims rdims_{};
    bool rbuffers_ = false;
    id<MTLBuffer> x_, norm_, q_, att_, proj_, gate_, up_, scores_, probs_, logits_;
    std::vector<id<MTLBuffer>> key_cache_, val_cache_;
    id<MTLBuffer> buffer(Input input) {
        auto b = input.empty()
                     ? [device_ newBufferWithLength:4 options:MTLResourceStorageModeShared]
                     : [device_ newBufferWithBytes:input.data()
                                            length:input.size_bytes()
                                           options:MTLResourceStorageModeShared];
        if (!b)
            throw std::runtime_error("Metal buffer allocation failed");
        return b;
    }
    id<MTLBuffer> output(size_t n) {
        auto b = [device_ newBufferWithLength:std::max(size_t(4), checked_mul(n, 4))
                                      options:MTLResourceStorageModeShared];
        if (!b)
            throw std::runtime_error("Metal output allocation failed");
        return b;
    }
    void run(const char *name, std::initializer_list<id<MTLBuffer>> buffers, const Params &p,
             size_t count) {
        if (!count)
            return;
        @autoreleasepool {
            auto begin = diagnostics.enabled ? Profiler::Clock::now() : Profiler::Clock::time_point{};
            auto pipeline = pipelines_.at(name);
            auto command = [queue_ commandBuffer];
            auto encoder = [command computeCommandEncoder];
            if (!command || !encoder)
                throw std::runtime_error("Metal command allocation failed");
            [encoder setComputePipelineState:pipeline];
            NSUInteger index = 0;
            for (auto b : buffers)
                [encoder setBuffer:b offset:0 atIndex:index++];
            [encoder setBytes:&p length:sizeof(p) atIndex:index];
            NSUInteger width = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
            [encoder endEncoding];
            [command commit];
            auto submitted = diagnostics.enabled ? Profiler::Clock::now() : Profiler::Clock::time_point{};
            [command waitUntilCompleted];
            if (diagnostics.enabled) {
                auto completed = Profiler::Clock::now();
                diagnostics.record("Encoding and submission", std::chrono::duration<double, std::milli>(submitted - begin).count());
                diagnostics.record("Waiting", std::chrono::duration<double, std::milli>(completed - submitted).count());
            }
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error("Metal execution failed: " + error_text(command.error));
        }
    }
    void copy(id<MTLBuffer> b, Output y) {
        if (!y.empty())
            std::memcpy(y.data(), [b contents], y.size_bytes());
    }
    id<MTLBuffer> weight_buffer(const TensorView &w) {
        Vec values(w.size());
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = w.at(i);
        return buffer(values);
    }

  public:
    MetalBackend() {
        @autoreleasepool {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_)
                throw std::runtime_error("no Metal device available");
            queue_ = [device_ newCommandQueue];
            if (!queue_)
                throw std::runtime_error("cannot create Metal command queue");
            NSError *error = nil;
            MTLCompileOptions *options = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                options.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                options.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            auto library =
                [device_ newLibraryWithSource:[NSString stringWithUTF8String:kMetalSource]
                                      options:options
                                        error:&error];
            if (!library)
                throw std::runtime_error("Metal kernel compilation failed: " + error_text(error));
            for (auto name : {"matmul", "elementwise", "rmsnorm", "rope", "softmax",
                              "attention_scores", "attention_values"}) {
                auto function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                auto pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
                if (!pipeline)
                    throw std::runtime_error("Metal pipeline failed: " + error_text(error));
                pipelines_[name] = pipeline;
            }
        }
    }
    std::string name() const override {
        return "metal";
    }
    void prepare_weight(const TensorView &w) override {
        if (!weights_.contains(w.data())) {
            weights_[w.data()] = weight_buffer(w);
            resident_ += w.size() * 4;
        }
    }
    void release_weights() override {
        weights_.clear();
        resident_ = 0;
        rbuffers_ = false;
        key_cache_.clear();
        val_cache_.clear();
        norms_.clear();
    }
    size_t resident_weight_bytes() const override {
        return resident_;
    }
    void matmul(const TensorView &w, Input x, Output y, size_t batch) override {
        if (w.shape().size() != 2)
            throw std::invalid_argument("matmul weight rank must be 2");
        Params p;
        p.rows = u32(w.shape()[0]);
        p.cols = u32(w.shape()[1]);
        p.batch = u32(batch);
        require_size(x, checked_mul(batch, p.cols), "matmul input");
        require_size(y, checked_mul(batch, p.rows), "matmul output");
        u32(y.size());
        u32(w.size());
        auto wb = weights_.contains(w.data()) ? weights_.at(w.data()) : weight_buffer(w);
        auto xb = buffer(x), yb = output(y.size());
        run("matmul", {wb, xb, yb}, p, y.size());
        copy(yb, y);
    }
    void rmsnorm(Input x, Input w, Output y, float epsilon) override {
        require_size(w, x.size(), "rmsnorm");
        require_size(y, x.size(), "rmsnorm");
        if (x.empty() || !(epsilon > 0) || !std::isfinite(epsilon))
            throw std::invalid_argument("invalid RMSNorm");
        Params p;
        p.n = u32(x.size());
        p.epsilon = epsilon;
        auto xb = buffer(x), wb = buffer(w), yb = output(y.size());
        run("rmsnorm", {xb, wb, yb}, p, y.size());
        copy(yb, y);
    }
    void elementwise(Input a, Input b, Output y, uint32_t op) {
        require_size(b, a.size(), "elementwise");
        require_size(y, a.size(), "elementwise");
        Params p;
        p.n = u32(a.size());
        p.operation = op;
        auto ab = buffer(a), bb = buffer(b), yb = output(y.size());
        run("elementwise", {ab, bb, yb}, p, y.size());
        copy(yb, y);
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
        Params p;
        p.n = u32(x.size());
        p.rows = 1;
        auto xb = buffer(x), yb = output(y.size());
        run("softmax", {xb, yb}, p, 1);
        copy(yb, y);
    }
    void rope(Output x, size_t heads, size_t dim, size_t position, Input inv_freq) override {
        require_size(x, checked_mul(heads, dim), "rope");
        if (!dim || dim % 2)
            throw std::invalid_argument("invalid RoPE");
        require_size(inv_freq, dim / 2, "rope inv_freq");
        Params p;
        p.heads = u32(heads);
        p.dim = u32(dim);
        p.position = u32(position);
        u32(x.size());
        auto xb = buffer(x), yb = output(x.size()), fb = buffer(inv_freq);
        run("rope", {xb, yb, fb}, p, x.size() / 2);
        copy(yb, x);
    }
    void attention(Input q, Input k, Input v, Output y, size_t heads, size_t kv_heads, size_t dim,
                   size_t tokens) override {
        validate_attention(q, k, v, y, heads, kv_heads, dim, tokens);
        Params p;
        p.heads = u32(heads);
        p.kv_heads = u32(kv_heads);
        p.dim = u32(dim);
        p.tokens = u32(tokens);
        p.rows = p.heads;
        p.n = p.tokens;
        u32(k.size());
        u32(q.size());
        u32(checked_mul(heads, tokens));
        auto qb = buffer(q), kb = buffer(k), vb = buffer(v), scores = output(heads * tokens),
             probs = output(heads * tokens), yb = output(y.size());
        run("attention_scores", {qb, kb, scores}, p, heads * tokens);
        run("softmax", {scores, probs}, p, heads);
        run("attention_values", {probs, vb, yb}, p, y.size());
        copy(yb, y);
    }

    // ---- Resident fused-forward path (Fix #1) ----
    bool supports_resident_forward() const override {
        return true;
    }
    id<MTLBuffer> device_weight(const TensorView &w) {
        return weights_.contains(w.data()) ? weights_.at(w.data()) : weight_buffer(w);
    }
    id<MTLBuffer> norm_buffer(Input w) {
        if (!norms_.contains(w.data()))
            norms_[w.data()] = buffer(w);
        return norms_.at(w.data());
    }
    // Encode one kernel into an already-open encoder; no commit, no wait. Each entry is a
    // {buffer, byte offset} so projections can write straight into a KV cache row.
    void encode(id<MTLComputeCommandEncoder> enc, const char *name,
                std::initializer_list<std::pair<id<MTLBuffer>, NSUInteger>> buffers,
                const Params &p, size_t count) {
        if (!count)
            return;
        auto pipeline = pipelines_.at(name);
        [enc setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (auto &b : buffers)
            [enc setBuffer:b.first offset:b.second atIndex:index++];
        [enc setBytes:&p length:sizeof(p) atIndex:index];
        NSUInteger width = std::min(NSUInteger(128), pipeline.maxTotalThreadsPerThreadgroup);
        [enc dispatchThreads:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
    }
    void ensure_resident(const ForwardDims &d, size_t layers) {
        bool same = rbuffers_ && rdims_.hidden == d.hidden && rdims_.intermediate == d.intermediate &&
                    rdims_.heads == d.heads && rdims_.kv_heads == d.kv_heads &&
                    rdims_.head_dim == d.head_dim && rdims_.vocab == d.vocab &&
                    rdims_.capacity == d.capacity && key_cache_.size() == layers;
        if (same)
            return;
        size_t kvw = d.kv_heads * d.head_dim;
        x_ = output(d.hidden);
        norm_ = output(d.hidden);
        q_ = output(d.heads * d.head_dim);
        att_ = output(d.hidden);
        proj_ = output(d.hidden);
        gate_ = output(d.intermediate);
        up_ = output(d.intermediate);
        scores_ = output(d.heads * d.capacity);
        probs_ = output(d.heads * d.capacity);
        logits_ = output(d.vocab);
        key_cache_.assign(layers, nil);
        val_cache_.assign(layers, nil);
        for (size_t l = 0; l < layers; ++l) {
            key_cache_[l] = output(d.capacity * kvw);
            val_cache_[l] = output(d.capacity * kvw);
        }
        rdims_ = d;
        rbuffers_ = true;
    }
    void resident_forward(Input embedding, std::span<const ResidentLayer> layers, Input final_norm,
                          const TensorView &head, const ForwardDims &d, Input inv_freq,
                          size_t position, Output logits) override {
        if (position >= d.capacity)
            throw std::out_of_range("resident forward position exceeds capacity");
        require_size(embedding, d.hidden, "resident embedding");
        require_size(logits, d.vocab, "resident logits");
        require_size(inv_freq, d.head_dim / 2, "resident inv_freq");
        @autoreleasepool {
            ensure_resident(d, layers.size());
            std::memcpy([x_ contents], embedding.data(), d.hidden * sizeof(float));
            id<MTLBuffer> freq = norm_buffer(inv_freq);
            const size_t h = d.hidden, inter = d.intermediate, hd = d.head_dim;
            const size_t kvw = d.kv_heads * hd, tokens = position + 1;
            const NSUInteger row = NSUInteger(position) * kvw * sizeof(float);
            auto begin = diagnostics.enabled ? Profiler::Clock::now() : Profiler::Clock::time_point{};
            auto command = [queue_ commandBuffer];
            auto enc = [command computeCommandEncoder];
            if (!command || !enc)
                throw std::runtime_error("Metal command allocation failed");
            auto mat = [&](const TensorView &w, id<MTLBuffer> xb, NSUInteger xoff, id<MTLBuffer> yb,
                          NSUInteger yoff, size_t rows) {
                Params p;
                p.rows = u32(w.shape()[0]);
                p.cols = u32(w.shape()[1]);
                p.batch = 1;
                encode(enc, "matmul", {{device_weight(w), 0}, {xb, xoff}, {yb, yoff}}, p, rows);
            };
            for (size_t l = 0; l < layers.size(); ++l) {
                const auto &w = layers[l];
                id<MTLBuffer> kc = key_cache_[l], vc = val_cache_[l];
                Params pn;
                pn.n = u32(h);
                pn.epsilon = d.epsilon;
                encode(enc, "rmsnorm", {{x_, 0}, {norm_buffer(w.input_norm), 0}, {norm_, 0}}, pn, h);
                mat(*w.q, norm_, 0, q_, 0, h);
                mat(*w.k, norm_, 0, kc, row, kvw);
                mat(*w.v, norm_, 0, vc, row, kvw);
                Params pr;
                pr.dim = u32(hd);
                pr.position = u32(position);
                pr.heads = u32(d.heads);
                encode(enc, "rope", {{q_, 0}, {q_, 0}, {freq, 0}}, pr, d.heads * hd / 2);
                pr.heads = u32(d.kv_heads);
                encode(enc, "rope", {{kc, row}, {kc, row}, {freq, 0}}, pr, d.kv_heads * hd / 2);
                Params pa;
                pa.heads = u32(d.heads);
                pa.kv_heads = u32(d.kv_heads);
                pa.dim = u32(hd);
                pa.tokens = u32(tokens);
                pa.rows = u32(d.heads);
                pa.n = u32(tokens);
                encode(enc, "attention_scores", {{q_, 0}, {kc, 0}, {scores_, 0}}, pa,
                       d.heads * tokens);
                encode(enc, "softmax", {{scores_, 0}, {probs_, 0}}, pa, d.heads);
                encode(enc, "attention_values", {{probs_, 0}, {vc, 0}, {att_, 0}}, pa, h);
                mat(*w.o, att_, 0, proj_, 0, h);
                Params pe;
                pe.n = u32(h);
                encode(enc, "elementwise", {{x_, 0}, {proj_, 0}, {x_, 0}}, pe, h); // residual add
                encode(enc, "rmsnorm", {{x_, 0}, {norm_buffer(w.post_norm), 0}, {norm_, 0}}, pn, h);
                mat(*w.gate, norm_, 0, gate_, 0, inter);
                mat(*w.up, norm_, 0, up_, 0, inter);
                Params ps;
                ps.n = u32(inter);
                ps.operation = 2;
                encode(enc, "elementwise", {{gate_, 0}, {gate_, 0}, {gate_, 0}}, ps, inter); // SiLU
                ps.operation = 1;
                encode(enc, "elementwise", {{gate_, 0}, {up_, 0}, {gate_, 0}}, ps, inter); // gate*up
                mat(*w.down, gate_, 0, proj_, 0, h);
                encode(enc, "elementwise", {{x_, 0}, {proj_, 0}, {x_, 0}}, pe, h); // residual add
            }
            Params pf;
            pf.n = u32(h);
            pf.epsilon = d.epsilon;
            encode(enc, "rmsnorm", {{x_, 0}, {norm_buffer(final_norm), 0}, {norm_, 0}}, pf, h);
            mat(head, norm_, 0, logits_, 0, d.vocab);
            [enc endEncoding];
            [command commit];
            auto submitted = diagnostics.enabled ? Profiler::Clock::now() : Profiler::Clock::time_point{};
            [command waitUntilCompleted];
            if (diagnostics.enabled) {
                auto done = Profiler::Clock::now();
                diagnostics.record("Encoding and submission",
                                   std::chrono::duration<double, std::milli>(submitted - begin).count());
                diagnostics.record("Waiting",
                                   std::chrono::duration<double, std::milli>(done - submitted).count());
            }
            if (command.status == MTLCommandBufferStatusError)
                throw std::runtime_error("Metal execution failed: " + error_text(command.error));
            std::memcpy(logits.data(), [logits_ contents], d.vocab * sizeof(float));
        }
    }
};
std::unique_ptr<Backend> make_metal_backend() {
    return std::make_unique<MetalBackend>();
}
} // namespace kestrel
