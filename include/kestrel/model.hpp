#pragma once
#include "kestrel/cache.hpp"
#include "kestrel/profiler.hpp"
#include "kestrel/safetensors.hpp"
#include <functional>
namespace kestrel {
using Trace = std::function<void(const std::string &, size_t, Input)>;
class Model {
  public:
    Model(const std::filesystem::path &directory, Backend &backend);
    ~Model() {
        backend_.release_weights();
    }
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    const Config &config() const {
        return config_;
    }
    Profiler profiler;
    size_t weight_bytes() const {
        return weights_.bytes();
    }
    // One causal transformer step; position must equal every layer's cache length.
    Vec forward(int token, size_t position, KVCache &cache, const Trace &trace = {});
    // Process a whole prompt with batched (matrix-matrix) projections; fills the cache for
    // positions [0, prompt.size()) and returns the last token's logits. No trace support.
    Vec forward_prefill(std::span<const int> prompt, KVCache &cache);
    bool backend_resident() const {
        return backend_.supports_resident_forward();
    }
    void reset_backend_state() {
        backend_.reset_resident_state();
    }

  private:
    Config config_;
    Weights weights_;
    Backend &backend_;
    struct Layer {
        const TensorView *q, *k, *v, *o, *gate, *up, *down;
        Vec input_norm, post_norm;
    };
    std::vector<Layer> layers_;
    Vec final_norm_;
    const TensorView *embedding_, *head_;
    const TensorView &weight(const std::string &name, std::vector<size_t> shape) const;
    Vec vector_weight(const std::string &name, size_t size) const;
    // Whole-pass forward encoded into one device command buffer (no host readback mid-pass).
    Vec forward_resident(int token, size_t position, size_t capacity);
    Vec inv_freq_; // Precomputed RoPE inverse frequencies (head_dim/2), with any llama3 scaling.
};
class Session {
  public:
    Session(Model &model, size_t capacity) : model_(model), cache_(model.config(), capacity) {}
    Vec prefill(std::span<const int> prompt, const Trace &trace = {});
    Vec decode(int token, const Trace &trace = {});
    Profiler &profiler() { return model_.profiler; }
    void reset() {
        cache_.clear();
        position_ = 0;
        model_.reset_backend_state();
    }
    size_t position() const {
        return position_;
    }
    size_t cache_bytes() const {
        return cache_.bytes();
    }

  private:
    Model &model_;
    ContiguousKVCache cache_;
    size_t position_ = 0;
};
} // namespace kestrel
