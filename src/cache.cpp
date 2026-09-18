#include "kestrel/cache.hpp"
#include <algorithm>
#include <stdexcept>
namespace kestrel {
ContiguousKVCache::ContiguousKVCache(const Config &c, size_t capacity)
    : capacity_(capacity), width_(checked_mul(c.kv_heads, c.head_dim())), lengths_(c.layers, 0) {
    if (!capacity || capacity > c.max_positions)
        throw std::invalid_argument("KV capacity must be within model context limit");
    size_t elements = checked_mul(checked_mul(c.layers, capacity), width_);
    checked_mul(elements, 2 * sizeof(float));
    keys_.resize(elements);
    values_.resize(elements);
}
void ContiguousKVCache::append(size_t layer, size_t position, Input key, Input value) {
    if (position != lengths_.at(layer) || position >= capacity_)
        throw std::out_of_range("KV append must be sequential and within capacity");
    require_size(key, width_, "KV key");
    require_size(value, width_, "KV value");
    size_t offset = (layer * capacity_ + position) * width_;
    std::copy(key.begin(), key.end(), keys_.begin() + offset);
    std::copy(value.begin(), value.end(), values_.begin() + offset);
    ++lengths_[layer];
}
Input ContiguousKVCache::keys(size_t layer) const {
    auto n = lengths_.at(layer);
    return Input(keys_).subspan(layer * capacity_ * width_, n * width_);
}
Input ContiguousKVCache::values(size_t layer) const {
    auto n = lengths_.at(layer);
    return Input(values_).subspan(layer * capacity_ * width_, n * width_);
}
void ContiguousKVCache::clear() {
    std::fill(lengths_.begin(), lengths_.end(), 0);
}
} // namespace kestrel
