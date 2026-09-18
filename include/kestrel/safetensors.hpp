#pragma once
#include "kestrel/tensor.hpp"
#include <filesystem>
#include <map>
#include <memory>
namespace kestrel {
class MappedFile {
  public:
    explicit MappedFile(const std::filesystem::path &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    std::span<const std::byte> bytes() const {
        return {data_, size_};
    }

  private:
    const std::byte *data_ = nullptr;
    size_t size_ = 0;
};
class SafeTensors {
  public:
    explicit SafeTensors(const std::filesystem::path &path);
    const std::map<std::string, TensorView> &tensors() const {
        return tensors_;
    }

  private:
    MappedFile file_;
    std::map<std::string, TensorView> tensors_;
};
class Weights {
  public:
    explicit Weights(const std::filesystem::path &directory);
    const TensorView &get(const std::string &name) const;
    bool contains(const std::string &name) const {
        return tensors_.contains(name);
    }
    size_t bytes() const;

  private:
    std::vector<std::unique_ptr<SafeTensors>> files_;
    std::map<std::string, TensorView> tensors_;
};
} // namespace kestrel
