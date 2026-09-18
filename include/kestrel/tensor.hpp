#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace kestrel {
enum class DType { f32, f16, bf16 };
size_t dtype_size(DType dtype);
DType parse_dtype(std::string_view name);
size_t checked_mul(size_t a, size_t b);
size_t element_count(std::span<const size_t> shape);
float decode_scalar(const std::byte *bytes, DType dtype);

// Non-owning row-major view. Owner must outlive the view. All strides are in elements.
class TensorView {
  public:
    TensorView(std::span<const std::byte> data, DType dtype, std::vector<size_t> shape);
    const std::vector<size_t> &shape() const {
        return shape_;
    }
    const std::vector<size_t> &strides() const {
        return strides_;
    }
    DType dtype() const {
        return dtype_;
    }
    size_t size() const {
        return count_;
    }
    size_t bytes() const {
        return data_.size();
    }
    const std::byte *data() const {
        return data_.data();
    }
    float at(size_t flat_index) const;
    float at(std::span<const size_t> indices) const;

  private:
    std::span<const std::byte> data_;
    DType dtype_;
    std::vector<size_t> shape_, strides_;
    size_t count_;
};
class Tensor {
  public:
    explicit Tensor(std::vector<size_t> shape);
    TensorView view() const;
    std::span<float> values() {
        return values_;
    }
    std::span<const float> values() const {
        return values_;
    }
    const std::vector<size_t> &shape() const {
        return shape_;
    }

  private:
    std::vector<size_t> shape_;
    std::vector<float> values_;
};
} // namespace kestrel
