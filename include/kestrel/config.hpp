#pragma once
#include "kestrel/json.hpp"
namespace kestrel {
struct Config {
    size_t hidden, intermediate, layers, heads, kv_heads, vocab, max_positions;
    float epsilon, theta;
    int bos = 1, eos = 2;
    bool tied = false;
    // Llama 3 ("llama3") RoPE frequency scaling; inactive by default (plain RoPE).
    bool rope_llama3 = false;
    float rope_factor = 8.f, rope_low_freq = 1.f, rope_high_freq = 4.f;
    size_t rope_original_ctx = 8192;
    size_t head_dim() const {
        return hidden / heads;
    }
    static Config parse(const Json &j);
    static Config load(const std::filesystem::path &dir) {
        return parse(read_json(dir / "config.json"));
    }
};
} // namespace kestrel
