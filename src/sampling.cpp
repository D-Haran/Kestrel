#include "kestrel/sampling.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace kestrel {
Sampler::Sampler(Sampling o) : options_(o), random_(o.seed) {
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1)
        throw std::invalid_argument("temperature must be >=0; top-p must be in (0,1]");
}
int Sampler::operator()(Input logits) {
    int best = greedy(logits);
    if (options_.temperature == 0)
        return best;
    std::vector<size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&](size_t a, size_t b) { return logits[a] > logits[b]; });
    if (options_.top_k && options_.top_k < indices.size())
        indices.resize(options_.top_k);
    std::vector<double> probabilities;
    double sum = 0;
    for (auto i : indices) {
        double p = std::exp((double(logits[i]) - logits[best]) / options_.temperature);
        probabilities.push_back(p);
        sum += p;
    }
    double cumulative = 0;
    size_t keep = 0;
    do {
        cumulative += probabilities[keep++];
    } while (keep < indices.size() && cumulative / sum < options_.top_p);
    double target = (double(random_()) / 4294967296.0) * cumulative;
    for (size_t i = 0; i < keep; ++i) {
        target -= probabilities[i];
        if (target < 0)
            return int(indices[i]);
    }
    return int(indices[keep - 1]);
}
} // namespace kestrel
