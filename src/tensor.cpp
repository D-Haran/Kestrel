#include "kestrel/tensor.hpp"
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kestrel {
size_t checked_mul(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b)
        throw std::overflow_error("tensor size overflow");
    return a * b;
}
size_t dtype_size(DType d) {
    return d == DType::f32 ? 4 : 2;
}
DType parse_dtype(std::string_view name) {
    if (name == "F32")
        return DType::f32;
    if (name == "F16")
        return DType::f16;
    if (name == "BF16")
        return DType::bf16;
    throw std::invalid_argument("unsupported tensor dtype: " + std::string(name));
}
size_t element_count(std::span<const size_t> shape) {
    size_t count = 1;
    for (auto n : shape)
        count = checked_mul(count, n);
    return count;
}
float decode_scalar(const std::byte *p, DType d) {
    const auto byte = [&](int i) { return uint32_t(std::to_integer<unsigned char>(p[i])); };
    uint32_t bits = byte(0) | (byte(1) << 8);
    if (d == DType::f32)
        return std::bit_cast<float>(bits | (byte(2) << 16) | (byte(3) << 24));
    if (d == DType::bf16)
        return std::bit_cast<float>(bits << 16);
    const uint32_t sign = (bits & 0x8000) << 16;
    const uint32_t exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    if (exponent == 31)
        return std::bit_cast<float>(sign | 0x7f800000 | (mantissa << 13));
    if (exponent == 0)
        return std::copysign(std::ldexp(float(mantissa), -24), sign ? -1.f : 1.f);
    return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}
TensorView::TensorView(std::span<const std::byte> data, DType dtype, std::vector<size_t> shape)
    : data_(data), dtype_(dtype), shape_(std::move(shape)), strides_(shape_.size()),
      count_(element_count(shape_)) {
    if (data.size() != checked_mul(count_, dtype_size(dtype)))
        throw std::invalid_argument("tensor byte size does not match shape");
    size_t stride = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        strides_[i] = stride;
        stride = checked_mul(stride, shape_[i]);
    }
}
float TensorView::at(size_t index) const {
    if (index >= count_)
        throw std::out_of_range("tensor flat index out of range");
    return decode_scalar(data_.data() + index * dtype_size(dtype_), dtype_);
}
float TensorView::at(std::span<const size_t> indices) const {
    if (indices.size() != shape_.size())
        throw std::invalid_argument("tensor index rank mismatch");
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i])
            throw std::out_of_range("tensor index out of range");
        offset += indices[i] * strides_[i];
    }
    return at(offset);
}
Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape)), values_(element_count(shape_)) {}
TensorView Tensor::view() const {
    static_assert(std::endian::native == std::endian::little,
                  "Kestrel currently requires little-endian hosts");
    return {std::as_bytes(std::span(values_)), DType::f32, shape_};
}
} // namespace kestrel
