#pragma once
#include "kestrel/json.hpp"
namespace kestrel {
struct Config {
    size_t hidden, intermediate, layers, heads, kv_heads, vocab, max_positions;
    float epsilon, theta;
    int bos = 1, eos = 2;
    bool tied = false;
    size_t head_dim() const {
        return hidden / heads;
    }
    static Config parse(const Json &j);
    static Config load(const std::filesystem::path &dir) {
        return parse(read_json(dir / "config.json"));
    }
};
} // namespace kestrel
