#include "kestrel/safetensors.hpp"
#include "kestrel/json.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
namespace kestrel {
MappedFile::MappedFile(const std::filesystem::path &path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("cannot open weights: " + path.string() + ": " +
                                 std::strerror(errno));
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        throw std::runtime_error("invalid weights file");
    }
    size_ = size_t(st.st_size);
    void *p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED)
        throw std::runtime_error("mmap failed for " + path.string());
    data_ = static_cast<const std::byte *>(p);
}
MappedFile::~MappedFile() {
    if (data_)
        munmap(const_cast<std::byte *>(data_), size_);
}
SafeTensors::SafeTensors(const std::filesystem::path &path) : file_(path) {
    auto bytes = file_.bytes();
    if (bytes.size() < 8)
        throw std::runtime_error("truncated Safetensors prefix");
    uint64_t length = 0;
    for (int i = 0; i < 8; ++i)
        length |= uint64_t(std::to_integer<unsigned char>(bytes[i])) << (8 * i);
    if (length < 2 || length > 100'000'000 || length > bytes.size() - 8)
        throw std::runtime_error("invalid Safetensors header length");
    if (bytes[8] != std::byte{'{'})
        throw std::runtime_error("Safetensors header must start with {");
    const auto header =
        parse_json({reinterpret_cast<const char *>(bytes.data() + 8), size_t(length)});
    if (!header.is_object())
        throw std::runtime_error("Safetensors header must be an object");
    auto data = bytes.subspan(8 + size_t(length));
    std::vector<std::pair<size_t, size_t>> ranges;
    for (auto it = header.begin(); it != header.end(); ++it) {
        const auto &meta = it.value();
        if (it.key() == "__metadata__") {
            if (!meta.is_object())
                throw std::runtime_error("invalid Safetensors metadata");
            for (auto &v : meta)
                if (!v.is_string())
                    throw std::runtime_error("metadata values must be strings");
            continue;
        }
        if (!meta.at("shape").is_array())
            throw std::runtime_error("tensor shape must be an array");
        std::vector<size_t> shape;
        for (auto &n : meta.at("shape"))
            shape.push_back(json_size(n));
        auto dtype = parse_dtype(meta.at("dtype").get<std::string>());
        auto offsets = meta.at("data_offsets");
        if (!offsets.is_array() || offsets.size() != 2)
            throw std::runtime_error("invalid data_offsets");
        auto begin = json_size(offsets[0]), end = json_size(offsets[1]);
        if (begin > end || end > data.size() ||
            end - begin != checked_mul(element_count(shape), dtype_size(dtype)))
            throw std::runtime_error("invalid tensor byte range: " + it.key());
        tensors_.emplace(it.key(),
                         TensorView(data.subspan(begin, end - begin), dtype, std::move(shape)));
        ranges.emplace_back(begin, end);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t cursor = 0;
    for (auto [begin, end] : ranges) {
        if (begin != cursor)
            throw std::runtime_error("Safetensors contains overlap or unclaimed bytes");
        cursor = end;
    }
    if (cursor != data.size())
        throw std::runtime_error("Safetensors has trailing unclaimed data");
}
Weights::Weights(const std::filesystem::path &dir) {
    std::set<std::string> names;
    Json mapping;
    if (std::filesystem::exists(dir / "model.safetensors.index.json")) {
        mapping = read_json(dir / "model.safetensors.index.json").at("weight_map");
        if (!mapping.is_object() || mapping.empty())
            throw std::runtime_error("invalid shard index");
        for (auto &v : mapping) {
            auto name = v.get<std::string>();
            if (std::filesystem::path(name).filename() != name || name == "." || name == "..")
                throw std::runtime_error("unsafe shard filename");
            names.insert(name);
        }
    } else
        names.insert("model.safetensors");
    for (const auto &name : names) {
        auto file = std::make_unique<SafeTensors>(dir / name);
        for (const auto &[key, value] : file->tensors()) {
            if (!mapping.is_null() && (!mapping.contains(key) || mapping.at(key) != name))
                throw std::runtime_error("tensor/shard index mismatch: " + key);
            if (!tensors_.emplace(key, value).second)
                throw std::runtime_error("duplicate tensor: " + key);
        }
        files_.push_back(std::move(file));
    }
    if (!mapping.is_null() && mapping.size() != tensors_.size())
        throw std::runtime_error("shard index references missing tensors");
}
const TensorView &Weights::get(const std::string &name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("missing weight: " + name);
    return it->second;
}
size_t Weights::bytes() const {
    size_t n = 0;
    for (auto &[_, t] : tensors_)
        n += t.bytes();
    return n;
}
} // namespace kestrel
