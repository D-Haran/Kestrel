#pragma once
#include "kestrel/generation.hpp"
#include <random>
namespace kestrel {
struct Sampling {
    float temperature = 0;
    size_t top_k = 0;
    float top_p = 1;
    uint32_t seed = 42;
};
class Sampler {
  public:
    explicit Sampler(Sampling options);
    int operator()(Input logits);

  private:
    Sampling options_;
    std::mt19937 random_;
};
} // namespace kestrel
