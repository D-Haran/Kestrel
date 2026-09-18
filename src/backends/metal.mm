#include "kestrel/backend.hpp"
#include "metal_source.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
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
    size_t resident_ = 0;
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
    void rope(Output x, size_t heads, size_t dim, size_t position, float theta) override {
        require_size(x, checked_mul(heads, dim), "rope");
        if (!dim || dim % 2 || !(theta > 0) || !std::isfinite(theta))
            throw std::invalid_argument("invalid RoPE");
        Params p;
        p.heads = u32(heads);
        p.dim = u32(dim);
        p.position = u32(position);
        p.theta = theta;
        u32(x.size());
        auto xb = buffer(x), yb = output(x.size());
        run("rope", {xb, yb}, p, x.size() / 2);
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
};
std::unique_ptr<Backend> make_metal_backend() {
    return std::make_unique<MetalBackend>();
}
} // namespace kestrel
