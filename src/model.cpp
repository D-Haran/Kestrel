#include "kestrel/model.hpp"
#include <cmath>
#include <numbers>
#include <stdexcept>
namespace kestrel {
// RoPE inverse frequencies (head_dim/2), applying llama3 wavelength-based scaling if enabled.
// Mirrors Hugging Face's _compute_llama3_parameters.
static Vec compute_inv_freq(const Config &c) {
    const size_t dim = c.head_dim(), half = dim / 2;
    Vec inv(half);
    const double low_wl = double(c.rope_original_ctx) / c.rope_low_freq;
    const double high_wl = double(c.rope_original_ctx) / c.rope_high_freq;
    for (size_t i = 0; i < half; ++i) {
        double freq = std::pow(double(c.theta), -2.0 * double(i) / double(dim));
        if (c.rope_llama3) {
            double wl = 2.0 * std::numbers::pi / freq;
            if (wl > low_wl)
                freq /= c.rope_factor;
            else if (wl >= high_wl) {
                double smooth = (double(c.rope_original_ctx) / wl - c.rope_low_freq) /
                                (c.rope_high_freq - c.rope_low_freq);
                freq = (1.0 - smooth) * (freq / c.rope_factor) + smooth * freq;
            }
        }
        inv[i] = float(freq);
    }
    return inv;
}
const TensorView &Model::weight(const std::string &name, std::vector<size_t> shape) const {
    const auto &w = weights_.get(name);
    if (w.shape() != shape)
        throw std::runtime_error("unexpected weight shape: " + name);
    return w;
}
Vec Model::vector_weight(const std::string &name, size_t size) const {
    const auto &w = weight(name, {size});
    Vec out(size);
    for (size_t i = 0; i < size; ++i)
        out[i] = w.at(i);
    return out;
}
Model::Model(const std::filesystem::path &dir, Backend &backend)
    : config_(Config::load(dir)), weights_(dir), backend_(backend) {
    auto &c = config_;
    size_t kv = c.kv_heads * c.head_dim();
    embedding_ = &weight(c.tied && !weights_.contains("model.embed_tokens.weight")
                             ? "lm_head.weight"
                             : "model.embed_tokens.weight",
                         {c.vocab, c.hidden});
    head_ = c.tied ? embedding_ : &weight("lm_head.weight", {c.vocab, c.hidden});
    final_norm_ = vector_weight("model.norm.weight", c.hidden);
    inv_freq_ = compute_inv_freq(c);
    for (size_t i = 0; i < c.layers; ++i) {
        std::string p = "model.layers." + std::to_string(i) + ".";
        layers_.push_back({&weight(p + "self_attn.q_proj.weight", {c.hidden, c.hidden}),
                           &weight(p + "self_attn.k_proj.weight", {kv, c.hidden}),
                           &weight(p + "self_attn.v_proj.weight", {kv, c.hidden}),
                           &weight(p + "self_attn.o_proj.weight", {c.hidden, c.hidden}),
                           &weight(p + "mlp.gate_proj.weight", {c.intermediate, c.hidden}),
                           &weight(p + "mlp.up_proj.weight", {c.intermediate, c.hidden}),
                           &weight(p + "mlp.down_proj.weight", {c.hidden, c.intermediate}),
                           vector_weight(p + "input_layernorm.weight", c.hidden),
                           vector_weight(p + "post_attention_layernorm.weight", c.hidden)});
    }
    try {
        backend_.prepare_weight(*head_);
        for (const auto &layer : layers_)
            for (auto w : {layer.q, layer.k, layer.v, layer.o, layer.gate, layer.up, layer.down})
                backend_.prepare_weight(*w);
    } catch (...) {
        backend_.release_weights();
        throw;
    }
}
Vec Model::forward(int token, size_t pos, KVCache &cache, const Trace &trace) {
    auto total = profiler.scope("__forward");
    auto &c = config_;
    auto &b = backend_;
    if (token < 0 || size_t(token) >= c.vocab)
        throw std::out_of_range("token ID outside vocabulary");
    if (pos >= c.max_positions || pos >= cache.capacity())
        throw std::out_of_range("context capacity exceeded");
    // Fast path: encode the whole step into one device command buffer with the KV cache and
    // activations resident. Tracing needs per-op host values, so it keeps the op-by-op path.
    if (b.supports_resident_forward() && !trace)
        return forward_resident(token, pos, cache.capacity());
    for (size_t l = 0; l < c.layers; ++l)
        if (cache.length(l) != pos)
            throw std::logic_error("inconsistent KV cache position; reset the session");
    Vec x(c.hidden), norm(c.hidden), q(c.hidden), k(c.kv_heads * c.head_dim()), v(k.size()),
        att(c.hidden), projected(c.hidden), gate(c.intermediate), up(c.intermediate);
    profiler.measure("Embedding", [&] {
        for (size_t i = 0; i < c.hidden; ++i)
            x[i] = embedding_->at(size_t(token) * c.hidden + i);
    });
    if (trace)
        trace("embedding", pos, x);
    for (size_t l = 0; l < c.layers; ++l) {
        const auto &w = layers_[l];
        profiler.measure("RMSNorm", [&] { b.rmsnorm(x, w.input_norm, norm, c.epsilon); });
        profiler.measure("Q projection", [&] { b.matmul(*w.q, norm, q); });
        profiler.measure("K projection", [&] { b.matmul(*w.k, norm, k); });
        profiler.measure("V projection", [&] { b.matmul(*w.v, norm, v); });
        profiler.measure("RoPE", [&] {
            b.rope(q, c.heads, c.head_dim(), pos, inv_freq_);
            b.rope(k, c.kv_heads, c.head_dim(), pos, inv_freq_);
        });
        profiler.measure("KV append", [&] { cache.append(l, pos, k, v); });
        profiler.measure("Attention", [&] {
            b.attention(q, cache.keys(l), cache.values(l), att, c.heads, c.kv_heads, c.head_dim(),
                        pos + 1);
        });
        profiler.measure("Output projection", [&] { b.matmul(*w.o, att, projected); });
        profiler.measure("Residual", [&] { b.add(x, projected, x); });
        profiler.measure("RMSNorm", [&] { b.rmsnorm(x, w.post_norm, norm, c.epsilon); });
        profiler.measure("MLP gate", [&] { b.matmul(*w.gate, norm, gate); });
        profiler.measure("MLP up", [&] { b.matmul(*w.up, norm, up); });
        profiler.measure("Activation gating", [&] {
            b.silu(gate, gate);
            b.multiply(gate, up, gate);
        });
        profiler.measure("MLP down", [&] { b.matmul(*w.down, gate, projected); });
        profiler.measure("Residual", [&] { b.add(x, projected, x); });
        if (trace)
            trace("layer_" + std::to_string(l), pos, x);
    }
    profiler.measure("Final norm", [&] { b.rmsnorm(x, final_norm_, norm, c.epsilon); });
    if (trace)
        trace("final_hidden", pos, norm);
    Vec logits(c.vocab);
    profiler.measure("LM head", [&] { b.matmul(*head_, norm, logits); });
    if (trace)
        trace("logits", pos, logits);
    return logits;
}
Vec Model::forward_resident(int token, size_t pos, size_t capacity) {
    auto &c = config_;
    Vec embedding(c.hidden);
    for (size_t i = 0; i < c.hidden; ++i)
        embedding[i] = embedding_->at(size_t(token) * c.hidden + i);
    std::vector<ResidentLayer> layers;
    layers.reserve(layers_.size());
    for (const auto &w : layers_)
        layers.push_back({w.q, w.k, w.v, w.o, w.gate, w.up, w.down, w.input_norm, w.post_norm});
    ForwardDims dims{c.hidden, c.intermediate, c.heads,    c.kv_heads, c.head_dim(),
                     c.vocab,  capacity,       c.epsilon,  c.theta};
    Vec logits(c.vocab);
    backend_.resident_forward(embedding, layers, final_norm_, *head_, dims, inv_freq_, pos, logits);
    return logits;
}
Vec Model::forward_prefill(std::span<const int> prompt, KVCache &cache) {
    auto total = profiler.scope("__prefill");
    auto &c = config_;
    auto &b = backend_;
    const size_t M = prompt.size(), h = c.hidden, inter = c.intermediate, hd = c.head_dim(),
                 kvw = c.kv_heads * hd;
    if (M == 0 || M > cache.capacity() || M > c.max_positions)
        throw std::out_of_range("prefill prompt length exceeds capacity");
    for (int token : prompt)
        if (token < 0 || size_t(token) >= c.vocab)
            throw std::out_of_range("token ID outside vocabulary");
    Vec X(M * h), norm(M * h), Q(M * h), K(M * kvw), V(M * kvw), att(M * h), proj(M * h),
        gate(M * inter), up(M * inter);
    auto row = [](Vec &v, size_t i, size_t w) { return Output(v).subspan(i * w, w); };
    for (size_t i = 0; i < M; ++i)
        for (size_t j = 0; j < h; ++j)
            X[i * h + j] = embedding_->at(size_t(prompt[i]) * h + j);
    for (size_t l = 0; l < c.layers; ++l) {
        const auto &w = layers_[l];
        for (size_t i = 0; i < M; ++i)
            b.rmsnorm(row(X, i, h), w.input_norm, row(norm, i, h), c.epsilon);
        b.matmul(*w.q, norm, Q, M);
        b.matmul(*w.k, norm, K, M);
        b.matmul(*w.v, norm, V, M);
        for (size_t i = 0; i < M; ++i) {
            b.rope(row(Q, i, h), c.heads, hd, i, inv_freq_);
            b.rope(row(K, i, kvw), c.kv_heads, hd, i, inv_freq_);
        }
        for (size_t i = 0; i < M; ++i) {
            cache.append(l, i, row(K, i, kvw), row(V, i, kvw));
            b.attention(row(Q, i, h), cache.keys(l), cache.values(l), row(att, i, h), c.heads,
                        c.kv_heads, hd, i + 1);
        }
        b.matmul(*w.o, att, proj, M);
        b.add(X, proj, X);
        for (size_t i = 0; i < M; ++i)
            b.rmsnorm(row(X, i, h), w.post_norm, row(norm, i, h), c.epsilon);
        b.matmul(*w.gate, norm, gate, M);
        b.matmul(*w.up, norm, up, M);
        b.silu(gate, gate);
        b.multiply(gate, up, gate);
        b.matmul(*w.down, gate, proj, M);
        b.add(X, proj, X);
    }
    // Only the final position is needed to sample the next token.
    Vec last_norm(h), logits(c.vocab);
    b.rmsnorm(row(X, M - 1, h), final_norm_, last_norm, c.epsilon);
    b.matmul(*head_, last_norm, logits);
    return logits;
}
Vec Session::prefill(std::span<const int> prompt, const Trace &trace) {
    if (prompt.empty())
        throw std::invalid_argument("prompt must contain at least one token");
    if (prompt.size() > cache_.capacity())
        throw std::out_of_range("prompt exceeds context capacity");
    for (int token : prompt)
        if (token < 0 || size_t(token) >= model_.config().vocab)
            throw std::out_of_range("invalid prompt token ID");
    reset();
    Vec logits;
    try {
        if (!trace && !model_.backend_resident()) {
            // Batched matrix-matrix prefill (CPU path).
            logits = model_.forward_prefill(prompt, cache_);
            position_ = prompt.size();
        } else {
            // Resident per-token path (Metal) or the traced reference path.
            for (int token : prompt) {
                logits = model_.forward(token, position_, cache_, trace);
                ++position_;
            }
        }
    } catch (...) {
        reset();
        throw;
    }
    return logits;
}
Vec Session::decode(int token, const Trace &trace) {
    if (position() == 0)
        throw std::logic_error("decode requires prefill");
    try {
        auto logits = model_.forward(token, position_, cache_, trace);
        ++position_;
        return logits;
    } catch (...) {
        reset();
        throw;
    }
}
} // namespace kestrel
