#pragma once
#include "kestrel/backend.hpp"
#include "kestrel/config.hpp"
namespace kestrel {
// Replaceable cache interface: attention receives a contiguous prefix today.
class KVCache {
  public:
    virtual ~KVCache() = default;
    virtual void append(size_t layer, size_t position, Input key, Input value) = 0;
    virtual Input keys(size_t layer) const = 0;
    virtual Input values(size_t layer) const = 0;
    virtual size_t length(size_t layer) const = 0;
    virtual size_t capacity() const = 0;
    virtual size_t bytes() const = 0;
    virtual void clear() = 0;
};
class ContiguousKVCache final : public KVCache {
  public:
    ContiguousKVCache(const Config &config, size_t capacity);
    void append(size_t layer, size_t position, Input key, Input value) override;
    Input keys(size_t layer) const override;
    Input values(size_t layer) const override;
    size_t length(size_t layer) const override {
        return lengths_.at(layer);
    }
    size_t capacity() const override {
        return capacity_;
    }
    size_t bytes() const override {
        return (keys_.size() + values_.size()) * sizeof(float);
    }
    void clear() override;

  private:
    size_t capacity_, width_;
    Vec keys_, values_;
    std::vector<size_t> lengths_;
};
} // namespace kestrel
