#pragma once
#include "kestrel/model.hpp"
#include <functional>
namespace kestrel {
int greedy(Input logits);
struct GenerationResult {
    std::vector<int> tokens;
    double prefill_ms = 0, first_token_ms = 0;
    std::vector<double> inter_token_ms;
};
using SelectToken = std::function<int(Input)>;
GenerationResult generate(Session &session, std::span<const int> prompt, size_t max_tokens, int eos,
                          const SelectToken &select = greedy,
                          const std::function<void(int)> &emit = {}, bool stop_on_eos = true);
} // namespace kestrel
