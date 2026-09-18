#include "kestrel/generation.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace kestrel {
int greedy(Input logits) {
    if (logits.empty())
        throw std::invalid_argument("empty logits");
    for (float f : logits)
        if (!std::isfinite(f))
            throw std::runtime_error("nonfinite logits");
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
GenerationResult generate(Session &session, std::span<const int> prompt, size_t max_tokens, int eos,
                          const SelectToken &select, const std::function<void(int)> &emit,
                          bool stop_on_eos) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto total = session.profiler().scope("__generation");
    GenerationResult r;
    auto start = Clock::now();
    auto logits = session.prefill(prompt);
    auto prefilled = Clock::now();
    r.prefill_ms = ms(start, prefilled);
    auto previous = start;
    for (size_t i = 0; i < max_tokens; ++i) {
        if (i)
            logits = session.decode(r.tokens.back());
        int token;
        session.profiler().measure("Sampling", [&] { token = select(logits); });
        auto now = Clock::now();
        if (!i)
            r.first_token_ms = ms(start, now);
        else
            r.inter_token_ms.push_back(ms(previous, now));
        previous = now;
        r.tokens.push_back(token);
        if (emit)
            emit(token);
        if (stop_on_eos && token == eos)
            break;
    }
    return r;
}
} // namespace kestrel
